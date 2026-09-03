// diffusion/sampler.h — token sampling + a confidence metric over one position.
//
// Mirrors src/train/diffusion._sample_tokens for a single position: temperature
// scaling, top_k / top_p filtering, then either argmax (temperature <= 0) or a
// multinomial draw. The returned confidence drives the progressive-unmask ordering.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dlm {

struct Sample {
    float confidence;
    int32_t token;
};

inline void softmax_inplace(std::vector<float>& p) {
    float mx = *std::max_element(p.begin(), p.end());
    if (!std::isfinite(mx)) mx = 0.0f;              // guard against all -inf
    double sum = 0.0;
    for (float v : p) { double e = std::exp((double)(v - mx)); sum += e; }
    float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (float& v : p) v = v == -std::numeric_limits<float>::infinity() ? 0.0f : v * inv;
}

// Sample one position from raw logits [V]. Follows the reference order:
// temperature -> top_k -> top_p -> softmax.
inline Sample sample_one(
    const float* logits,
    int V,
    float temperature,
    float top_p,
    int top_k,
    const std::string& alg,
    std::mt19937& rng)
{
    std::vector<float> p(logits, logits + V);
    if (temperature > 0.0f) for (float& v : p) v /= temperature;

    const float neg_inf = -std::numeric_limits<float>::infinity();

    if (top_k > 0 && top_k < V) {
        std::vector<float> copy = p;
        std::nth_element(copy.begin(), copy.begin() + top_k - 1, copy.end(), std::greater<float>());
        float kth = copy[top_k - 1];
        for (float& v : p) if (v < kth) v = neg_inf;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (p[a] == p[b]) return a < b;
            return p[a] > p[b];
        });
        float mx = *std::max_element(p.begin(), p.end());
        double sum = 0.0;
        for (float v : p) sum += std::exp((double)(v - mx));
        double cum = 0.0;
        for (int k = 0; k < V; ++k) {
            double prob = std::exp((double)(p[idx[k]] - mx)) / sum;
            if (k == 0) { cum += prob; continue; }
            if (cum > (double)top_p) p[idx[k]] = neg_inf;
            cum += prob;
        }
    }

    softmax_inplace(p);

    int32_t x0;
    if (temperature > 0.0f) {
        std::discrete_distribution<int> dd(p.begin(), p.end());
        x0 = dd(rng);
    } else {
        x0 = (int32_t)(std::max_element(p.begin(), p.end()) - p.begin());
    }

    float conf = p[x0];
    if (alg == "topk_margin") {
        float largest = 0.0f, second = 0.0f;
        for (float v : p) {
            if (v > largest) { second = largest; largest = v; }
            else if (v > second) second = v;
        }
        conf = largest - second;
    } else if (alg == "entropy") {
        double ent = 0.0;
        for (float v : p) if (v > 0.0f) ent += (double)v * std::log((double)v);
        conf = (float)ent;
    }
    return {conf, x0};
}

} // namespace dlm
