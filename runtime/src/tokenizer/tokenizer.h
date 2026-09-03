// tokenizer/tokenizer.h — byte-level BPE (Qwen3 style), self-contained.
//
// Loads Qwen3's vocab.json + merges.txt and encodes/decodes text. This is a
// self-contained runtime tokenizer (no PyTorch/HF). v1 note: pre-tokenization splits
// on whitespace; the exact re2 regex Qwen3 ships is a later refinement.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dlm {

class BPETokenizer {
public:
    // Loads from a directory containing vocab.json and merges.txt.
    bool load(const std::string& dir);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t vocab_size() const { return (int32_t)id_to_token_.size(); }
    int32_t token_to_id(const std::string& tok) const;
    int32_t add_token(const std::string& tok, int32_t id);  // used by adapter for [MASK]
    const std::string& id_to_token(int32_t id) const;
    int32_t mask_token_id() const { return mask_id_; }
    void set_mask_id(int32_t id) { mask_id_ = id; }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "a b" -> rank (lower = first)
    int32_t mask_id_ = -1;
};

} // namespace dlm
