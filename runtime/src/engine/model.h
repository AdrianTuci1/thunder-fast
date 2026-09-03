// engine/model.h — runtime model (Qwen2 backbone over ggml) with bidirectional attention.
//
// Loads the converted diffusion weights (GGUF) and runs the denoising forward pass.
// The custom piece vs llama.cpp is the ALL-ONES (non-causal) attention mask, which
// turns a causal Qwen2 into a bidirectional MDM denoiser. Everything else maps to the
// standard Qwen2/Qwen3 ggml builder path. There is no KV-cache: the whole window is one
// forward pass (matching training + the reference DiffusionLM.forward).
//
// Exposes the exact `Forward` contract the diffusion core expects:
//     vector<float> forward(const vector<int32_t>& ids)  // returns logits [L * V]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dlm {

// Hyperparameters read from the GGUF at load time. None are hardcoded, so the runtime
// adapts to whichever backbone size the checkpoint was trained as (0.6B, 7B, ...).
struct RuntimeConfig {
    int32_t n_layer = 1;
    int32_t n_embd = 1;
    int32_t n_head = 1;
    int32_t n_head_kv = 1;
    int32_t n_ff = 1;
    int32_t vocab_size = 1;
    int32_t context_len = 256;
    int32_t mask_token_id = 0;
    int32_t infer_steps = 24;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    float rms_norm_eps = 1e-6f;
    std::string ffn_act = "silu";
};

class RuntimeModel {
public:
    // `model_path` is the converted GGUF produced by tools/convert_to_gguf.py.
    explicit RuntimeModel(const std::string& model_path);
    ~RuntimeModel();

    RuntimeModel(const RuntimeModel&) = delete;
    RuntimeModel& operator=(const RuntimeModel&) = delete;

    std::vector<float> forward(const std::vector<int32_t>& ids);

    const RuntimeConfig& config() const { return cfg_; }
    int32_t mask_token_id() const { return cfg_.mask_token_id; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    RuntimeConfig cfg_;
};

} // namespace dlm
