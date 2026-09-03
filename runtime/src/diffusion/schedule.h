// diffusion/schedule.h — mask/timestep schedule for discrete MDM.
//
// Mirrors src/train/diffusion._discrete_generate_window: a linear timestep grid
// from 1.0 down to `eps`, and the per-step fraction of masked positions to reveal.
#pragma once

#include <vector>

namespace dlm {

// steps+1 values linearly spaced from 1.0 (fully masked) down to `eps` (clean).
inline std::vector<float> make_timesteps(int steps, float eps) {
    std::vector<float> ts(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        ts[static_cast<size_t>(i)] = 1.0f + (eps - 1.0f) * (static_cast<float>(i) / static_cast<float>(steps));
    }
    return ts;
}

// Fraction of currently-masked positions to reveal at step `i`.
// On the final step everything remaining is revealed (p_transfer = 1).
inline float transfer_frac(int i, int steps, float t, float s) {
    (void)t; (void)s;
    return (i < steps - 1) ? (1.0f - s / t) : 1.0f;
}

} // namespace dlm
