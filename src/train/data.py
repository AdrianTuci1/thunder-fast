"""Streaming, weighted, multi-source text dataloader packed to a fixed sequence length.

Roughly follows the base-model + Romanian-heavy mix described in research/notes.md §6.
Each source is a Hugging Face dataset (+ optional split/config) with a weight; samples are
drawn proportionally to weight, tokenized with the diffusion model's tokenizer, and packed
into sequences of `seq_len` tokens. A small fraction of documents is prefixed with BOS to
learn natural text starts (the diffusion model sees real content rather than pure noise).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


class PackedDataset(IterableDataset):
    def __init__(
        self,
        sources: Sequence[dict],
        tokenizer,
        seq_len: int = 256,
        bos_ratio: float = 0.045,
        column: str = "text",
    ):
        self.sources = list(sources)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.bos_ratio = bos_ratio
        self.column = column
        weights = [s["weight"] for s in sources]
        norm = sum(weights)
        self.weights = [w / norm for w in weights]

    def _row_text(self, row) -> str | None:
        """Extract text from a row; handles mono (`text`) and parallel (`translation`) sources."""
        col = self.column
        if col in row and isinstance(row[col], str):
            return row[col]
        # Parallel (e.g. OPUS): join the source/target languages so the LM sees both.
        trans = row.get(col) or row.get("translation")
        if isinstance(trans, dict):
            parts = [v for v in trans.values() if isinstance(v, str)]
            if parts:
                return " ".join(parts)
        return None

    def _open_source(self, source: dict):
        """Stream a source a fresh (HF streaming is a single pass per iterator)."""
        name = source["name"]
        config = source.get("config", None)  # .get on source is a plain dict
        # Support the "repo_id:config" shorthand used in the config file (e.g. "allenai/c4:all").
        # load_dataset needs the config as its own argument, not embedded in the repo id.
        if config is None and ":" in name:
            name, config = name.rsplit(":", 1)
        # NOTE: no `.filter()` on the streaming dataset - it can stall shard streaming in
        # `datasets` 5.x. Empty/None rows are skipped in _iter_texts instead.
        ds = load_dataset(name, config, split=source.get("split", "train"), streaming=True)
        return iter(ds)

    def _iter_texts(self):
        import random

        dead: set[int] = set()
        while True:
            alive = [i for i in range(len(self.sources)) if i not in dead]
            if not alive:
                # Every source is unavailable (e.g. gated) -> stop iterating; the training
                # loop treats StopIteration as the end of an epoch.
                return
            alive_weights = [self.weights[i] for i in alive]
            idx = random.choices(alive, weights=alive_weights, k=1)[0]
            source = self.sources[idx]
            try:
                stream = self._open_source(source)
            except Exception as e:  # noqa: BLE001
                # A gated or broken source must not kill training: skip it this cycle.
                print(f"[data] source {source.get('name')} unavailable, skipping: {e}", flush=True)
                dead.add(idx)
                continue
            for row in stream:
                text = self._row_text(row)
                if not text:
                    continue
                yield text
            # Source exhausted -> pick the next weighted source, so a small source (e.g.
            # OPUS) is effectively oversampled rather than stalling the stream.

    def __iter__(self):
        buffer: list[int] = []
        for text in self._iter_texts():
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if self.bos_ratio > 0:
                import random

                if random.random() < self.bos_ratio:
                    bos = self.tokenizer.bos_token_id
                    if bos is not None:
                        ids = [bos] + ids
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)


def build_loader(config: dict, tokenizer, world_size: int = 1, rank: int = 0) -> DataLoader:
    ds = PackedDataset(
        config["data"]["sources"],
        tokenizer,
        seq_len=config["diffusion"]["seq_len"],
        bos_ratio=config["data"].get("bos_ratio", 0.045),
    )
    # num_workers > 0 moves streaming + tokenization off the main thread; this is what keeps
    # GPU-bound training fed. On Linux (Modal/RunPod) the workers inherit the tokenizer via
    # fork, so no pickling is needed. Workers persist so the streaming iterators stay warm.
    num_workers = int(config["training"].get("num_workers", 4))
    return DataLoader(
        ds,
        batch_size=config["training"]["batch_size_seq"],
        num_workers=num_workers,
        collate_fn=lambda b: torch.stack(b, dim=0),
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
