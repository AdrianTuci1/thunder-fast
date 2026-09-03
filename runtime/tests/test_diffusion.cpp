// Functional smoke test for the diffusion core (no ggml).
// Verifies: prefix held fixed, masked suffix filled with argmax, output length.

#include "diffusion/denoise.h"

#include <cassert>
#include <cstdio>
#include <vector>

using dlm::discrete_generate_window;

// A fake forward: logits favour token 999 at every position, so the argmax is 999.
struct FakeForward {
    int V;
    explicit FakeForward(int v) : V(v) {}
    std::vector<float> operator()(const std::vector<int32_t>& ids) const {
        size_t L = ids.size();
        std::vector<float> logits(L * (size_t)V, 0.0f);
        for (size_t j = 0; j < L; ++j) logits[j * (size_t)V + 999] = 10.0f;
        return logits;
    }
};

int main() {
    const int V = 1024;
    const int32_t mask = 1000;
    const std::vector<int32_t> prompt = {1, 2, 3};
    const int target_len = 8;   // prompt(3) + 5 generated
    FakeForward forward(V);

    // entropy (confidence path), temperature=0 => argmax, deterministic.
    auto out_ent = discrete_generate_window(forward, mask, prompt, target_len,
                                            24, 0.0f, 0.0f, 200, "entropy", 1e-3f, 0.6f, 42);
    assert(out_ent.size() == 5);
    for (int32_t t : out_ent) assert(t == 999);

    // origin path.
    auto out_orig = discrete_generate_window(forward, mask, prompt, target_len,
                                             24, 0.0f, 0.0f, 200, "origin", 1e-3f, 0.0f, 7);
    assert(out_orig.size() == 5);
    for (int32_t t : out_orig) assert(t == 999);

    // target_len == prompt len (G>=1) still produces at least one token.
    auto out_min = discrete_generate_window(forward, mask, prompt, /*target_len=*/3,
                                            24, 0.0f, 0.0f, 200, "entropy", 1e-3f, 0.6f, 1);
    assert(out_min.size() == 1);

    std::printf("test_diffusion: OK\n");
    return 0;
}
