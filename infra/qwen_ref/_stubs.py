"""Minimal in-repo stubs so the bundled Qwen2 MDM modeling imports cleanly for single-process GPU inference.

The bundled modeling imports a handful of symbols from distributed/data/utils packages that only
matter for multi-GPU training. On a single GPU those code paths are never entered
(``get_parallel_state().ulysses_enabled`` and ``.sp_enabled`` are False), and
``is_liger_kernel_available()`` is False so the torch kernels are used. We supply no-op stand-ins
so the module loads inside the Modal container without pulling in the full package (which would drag
in torchdata and friends).
"""

from __future__ import annotations

IGNORE_INDEX = -100


class _Logger:
    def get_logger(self, name):
        return _NoopLogger()


class _NoopLogger:
    def warning(self, *a, **k):
        pass

    def warning_once(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def info_rank0(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


logging = _Logger()


def is_liger_kernel_available() -> bool:
    return False


class _ParallelState:
    ulysses_enabled = False
    sp_enabled = False


_PS = _ParallelState()


def get_parallel_state() -> _ParallelState:
    return _PS


def gather_heads_scatter_seq(x, **kw):
    return x


def gather_seq_scatter_heads(x, **kw):
    return x


def reduce_sequence_parallel_loss(x, n):
    return x
