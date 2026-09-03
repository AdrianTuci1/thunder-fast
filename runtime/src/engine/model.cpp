// engine/model.cpp — Qwen2 backbone over ggml with BIDIRECTIONAL attention.
//
// The one real divergence from llama.cpp's Qwen2/Qwen3 builder is the attention mask:
// we build an ALL-ONES (non-causal) mask instead of a lower-triangular causal mask, so
// every position attends to every position — the masked-diffusion (MDM) denoise mode.
// No KV-cache: the whole window is a single forward pass, matching the reference
// DiffusionLM.forward and the training forward.
//
// REFERENCE, NOT YET BUILT: this is ported from llama.cpp's llm_build_qwen2 and is
// validated on the Modal build (see .github/workflows/build-runtime.yml). ggml's API
// surface changes between versions; any renamed symbol should be fixed against the ggml
// checked out under runtime/third_party at build time.
#include "engine/model.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dlm {

namespace {

struct Layer {
    std::vector<float> q;      // [n_embd, n_embd]
    std::vector<float> k;      // [n_embd, n_embd_kv]
    std::vector<float> v;      // [n_embd, n_embd_kv]
    std::vector<float> o;      // [n_embd, n_embd]
    std::vector<float> ffn_g;  // [n_embd, n_ff]
    std::vector<float> ffn_u;  // [n_embd, n_ff]
    std::vector<float> ffn_d;  // [n_ff, n_embd]
};

} // namespace

struct RuntimeModel::Impl {
    // Metadata + weights decoded into host buffers (v1: plain fp32 loads; quantization
    // is applied downstream on the ggml side / a later iteration).
    RuntimeConfig cfg;
    std::vector<float> tok_embd;          // [vocab, n_embd]
    std::vector<float> output_norm;       // [n_embd]
    std::vector<float> lm_head;           // [n_embd, vocab]
    std::vector<float> per_layer_norm;    // layer input norm
    std::vector<float> per_layer_post;    // layer output norm
    std::vector<float> mask_token_row;    // the added [MASK] embedding row
    std::vector<Layer> layers;
};

RuntimeModel::RuntimeModel(const std::string& model_path) : impl_(new Impl) {
    gguf_context* gctx = nullptr;
    {
        gguf_init_params params = {};
        params.no_alloc = true;
        params.ctx = nullptr;
        gctx = gguf_init_from_file(model_path.c_str(), params);
        if (!gctx) throw std::runtime_error("failed to open GGUF: " + model_path);
    }

    auto i32 = [&](const char* k, int32_t dflt) {
        int idx = gguf_find_key(gctx, k);
        return idx < 0 ? dflt : gguf_get_val_i32(gctx, idx);
    };
    auto f32 = [&](const char* k, float dflt) {
        int idx = gguf_find_key(gctx, k);
        return idx < 0 ? dflt : gguf_get_val_f32(gctx, idx);
    };

    impl_->cfg.n_layer       = i32("qwen2.n_layer", 28);
    impl_->cfg.n_embd        = i32("qwen2.n_embd", 1024);
    impl_->cfg.n_head        = i32("qwen2.n_head", 16);
    impl_->cfg.n_head_kv     = i32("qwen2.n_head_kv", 8);
    impl_->cfg.n_ff          = i32("qwen2.n_ff", 3072);
    impl_->cfg.vocab_size    = i32("qwen2.vocab_size", 151936);
    impl_->cfg.context_len   = i32("qwen2.context_length", 256);
    impl_->cfg.mask_token_id = i32("dlm.mask_token_id", 151665);
    impl_->cfg.infer_steps   = i32("dlm.infer_steps", 24);
    impl_->cfg.rms_norm_eps  = f32("qwen2.rms_norm_eps", 1e-6f);
    impl_->cfg.rope_freq_base= f32("qwen2.rope_freq_base", 10000.0f);

    // TODO(build): read the weight tensors out of the gguf context (gguf_get_tensor →
    //    ggml_backend_tensor_get into host buffers) keyed by the standard names
    //    (token_embd.weight, blk.N.attn_q/k/v/output.weight, blk.N.ffn_gate/up/down,
    //    blk.N.attn_norm.weight, blk.N.ffn_norm.weight, output_norm.weight, output.weight).
    //    This is the mechanical part a ggml build with the right tensor-mapping header
    //    fills in; it is validated on the Modal build so the exact names/shapes can be
    //    checked against the converter that wrote the GGUF.
    gguf_free(gctx);
}

RuntimeModel::~RuntimeModel() { delete impl_; }

std::vector<float> RuntimeModel::forward(const std::vector<int32_t>& ids) {
    // Build the ggml graph: token_embd → per-layer [attn(bidir) + ffn] → output_norm →
    // lm_head → logits. The attention subgraph uses an ALL-ONES KQ mask:
    //
    //   q = gemm(attn_q, x_norm); k = gemm(attn_k, x_norm); v = gemm(attn_v, x_norm)
    //   q = rope(q, pos); k = rope(k, pos)
    //   kq = dot(q, k)                       // [n_head, L, n_head_kv] after reshape/transpose
    //   kq = mask_ones + softmax(kq)          // full attention (non-causal)
    //   out = dot(v, kq) → gemm(attn_o) → residual
    //
    // The generated window is short (context_len tokens) so a plain (non-flash) softmax
    // path is correct and simple; a flash / fused kernel is a later optimization.
    //
    // TODO(build): materialize this graph with the loaded tensors. Returned logits are
    //    [L * vocab_size] in row-major (position-major) order, matching denoise.h.
    throw std::runtime_error("RuntimeModel::forward needs the Modal ggml build (ADR 0012)");
}

} // namespace dlm
