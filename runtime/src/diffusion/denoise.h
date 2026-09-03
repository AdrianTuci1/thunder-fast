// diffusion/denoise.h — progressive discrete un-masking over one fixed window.
//
// Mirrors src/train/diffusion._discrete_generate_window for a single sequence
// (B = 1, matching the reference eval path). Prompt tokens are held fixed; the
// remaining positions start from `mask_token_id` and are revealed most-confident-first.
//
// `Forward` is any callable:  const vector<int32_t>& ids -> vector<float>& logits of L*V.
#pragma once

#include "schedule.h"
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace dlm {

template <typename Forward>
std::vector<int32_t> discrete_generate_window(
    Forward&& forward,
    int32_t mask_token_id,
    const std::vector<int32_t>& prompt_ids,
    int target_len,
    int steps,
    float temperature,
    float top_p,
    int top_k,
    const std::string& alg,
    float eps,
    float alg_temp,
    uint64_t seed)
{
    const int P = (int)prompt_ids.size();
    const int G = std::max(1, target_len - P);
    const int L = P + G;

    // x = [prompt..., mask * G]
    std::vector<int32_t> x(L);
    std::vector<char> fix_mask(L, 0);
    for (int j = 0; j < L; ++j) {
        if (j < P) { x[j] = prompt_ids[j]; fix_mask[j] = 1; }
        else x[j] = mask_token_id;
    }

    std::vector<float> ts = make_timesteps(steps, eps);
    std::mt19937 rng(seed);

    for (int i = 0; i < steps; ++i) {
        std::vector<char> mask_index(L);
        bool any_masked = false;
        for (int j = 0; j < L; ++j) { mask_index[j] = x[j] == mask_token_id; any_masked |= mask_index[j]; }
        if (!any_masked) break;

        const std::vector<float>& logits = forward(x);
        const int V = (int)(logits.size() / L);

        // Next-token alignment (matches training/reference): position j reads logits[j-1].
        std::vector<float> shifted((size_t)L * V);
        for (int j = 0; j < L; ++j) {
            int src = (j == 0) ? j : j - 1;
            for (int v = 0; v < V; ++v) shifted[(size_t)j * V + v] = logits[(size_t)src * V + v];
        }

        const float t = ts[(size_t)i];
        const float s = ts[(size_t)i + 1];

        if (alg == "origin") {
            const float pf = transfer_frac(i, steps, t, s);
            for (int j = 0; j < L; ++j) {
                if (!mask_index[j]) continue;
                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) >= pf) continue;
                Sample smp = sample_one(&shifted[(size_t)j * V], V, temperature, top_p, top_k, "origin", rng);
                x[j] = smp.token;
            }
        } else {
            // Confidence-based (entropy / topk_margin / maskgit_plus).
            std::vector<float> conf((size_t)L);
            std::vector<int32_t> x0((size_t)L);
            for (int j = 0; j < L; ++j) {
                Sample smp = sample_one(&shifted[(size_t)j * V], V, temperature, top_p, top_k, alg, rng);
                conf[j] = smp.confidence;
                x0[j] = smp.token;
            }

            int num_masked = 0;
            for (int j = 0; j < L; ++j) num_masked += mask_index[j];
            int n_transfer = (i < steps - 1)
                ? (int)(num_masked * (1.0f - s / t))
                : num_masked;
            n_transfer = std::clamp(n_transfer, 0, num_masked);

            if (n_transfer <= 0) continue;

            // Gather masked positions with their confidence.
            std::vector<int> masked;
            for (int j = 0; j < L; ++j) if (mask_index[j]) masked.push_back(j);
            std::sort(masked.begin(), masked.end(), [&](int a, int b) { return conf[a] > conf[b]; });

            if (alg_temp > 0.0f && n_transfer < (int)masked.size()) {
                // Soften confidence into a distribution over masked positions, then sample
                // n_transfer without replacement.
                std::vector<float> w(masked.size());
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t k = 0; k < masked.size(); ++k) {
                    w[k] = conf[masked[k]] / (alg_temp);
                    mx = std::max(mx, w[k]);
                }
                double sum = 0.0;
                for (float wv : w) sum += std::exp((double)(wv - mx));
                std::vector<double> prob(masked.size());
                for (size_t k = 0; k < masked.size(); ++k) prob[k] = std::exp((double)(w[k] - mx)) / sum;
                std::discrete_distribution<int> dd(prob.begin(), prob.end());
                std::vector<char> used(masked.size(), 0);
                int placed = 0;
                while (placed < n_transfer) {
                    int k = dd(rng);
                    if (used[k]) continue;
                    used[k] = 1;
                    x[masked[k]] = x0[masked[k]];
                    ++placed;
                }
            } else {
                for (int c = 0; c < n_transfer; ++c) {
                    int j = masked[c];
                    x[j] = x0[j];
                }
            }
        }
    }

    return std::vector<int32_t>(x.begin() + P, x.begin() + L);
}

} // namespace dlm
