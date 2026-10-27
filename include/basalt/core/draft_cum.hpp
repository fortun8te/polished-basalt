// include/basalt/core/draft_cum.hpp - the cumulative draft-length rule.
//
// The MTP chain keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above the
// threshold: the product over i <= j of acc(p_i), where acc maps a draft probability to the observed acceptance rate of drafts in
// that probability bin.  A middling first draft is cheap to verify and still often right, so it is proposed; a long
// run of merely high probabilities stops once the joint chance decays.  The host (the window size, the draft loop)
// and the device (the conditional draft graph) evaluate the same products in the same order, so they agree exactly.
// Only what is proposed changes: acceptance and outputs stay exact.
#pragma once

#include "basalt/core/diag.hpp"
#include "basalt/platform/system.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define DRAFT_CUM_HD __host__ __device__
#else
#define DRAFT_CUM_HD
#endif

namespace basalt::core {

/// The acceptance rate of a sampled draft by its probability bin.
DRAFT_CUM_HD inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// Whether draft j stays in the window: prod_{i <= j} acc(probs[i]) >= th.
DRAFT_CUM_HD inline bool draft_keep(const float* probs, int j, float th) {
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= th;
}

/// The rule's base threshold: what a request starts from and the device graph's default.
constexpr float kDraftCumBase0 = 0.35f;

/// The CPU-served expert bytes per expert slot, in MB, set once when the expert tiers are final.  Each extra verified
/// token adds distinct experts the CPU must read from RAM, so a pack that leaves more bytes in RAM pays more per extra
/// token; the rule's threshold rises with it.  Fixed by the pack and its placement, never by timing.
inline float& draft_cum_cpu_mb() { static float v = 0.0f; return v; }
inline float draft_cum_scale() {
    constexpr float kLoMb = 1.1f, kHiMb = 3.5f, kMax = 1.5f;
    const float t = (draft_cum_cpu_mb() - kLoMb) / (kHiMb - kLoMb);
    // BASALT_FORK_DRAFT_CUM_MULT: a factor on the scale (1 = as is; below 1 proposes longer windows, above 1 shorter ones)
    static const float mult = (float) basalt::platform::fork_number("DRAFT_CUM_MULT", 1.0, 0.25, 4.0);
    return mult * (1.0f + (kMax - 1.0f) * (t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t));
}
inline float draft_cum_base() { return kDraftCumBase0 * draft_cum_scale(); }

/// The CPU expert misses of recent verify windows against this request's own history, by window size.  Every extra
/// verified token adds experts the CPU must read, so when windows of a given size are missing more than they used to,
/// extra tokens cost more and the threshold should rise; when they miss less (the VRAM tier has warmed to the
/// request), it can fall.  Each size is compared with itself, so a change in which sizes get chosen does not move it.
struct MissTrend {
    static constexpr int kSizes = 16;
    static constexpr int kWarm = 16;        ///< windows of one size before its trend counts
    double fast[kSizes + 1] = {}, slow[kSizes + 1] = {};
    int n[kSizes + 1] = {};
    float scale = 1.0f;                     ///< the factor on the threshold

    /// One verified window of `t` tokens in which the CPU computed `per_layer` distinct experts per layer.
    void observe(int t, double per_layer) {
        constexpr double kFast = 0.1, kSlow = 0.01;
        constexpr float kLoScale = 0.85f, kHiScale = 1.2f;
        t = t < 1 ? 1 : t > kSizes ? kSizes : t;
        if (n[t] == 0) fast[t] = slow[t] = per_layer;
        fast[t] += kFast * (per_layer - fast[t]);
        slow[t] += kSlow * (per_layer - slow[t]);
        ++n[t];
        double lr = 0.0, w = 0.0;
        for (int u = 1; u <= kSizes; ++u) {
            if (n[u] < kWarm || slow[u] < 1e-6 || fast[u] < 1e-6) continue;
            const double wu = n[u] < 32 ? (double) n[u] : 32.0;
            lr += wu * std::log(fast[u] / slow[u]);
            w += wu;
        }
        if (w <= 0.0) return;
        const float r = (float) std::exp(0.5 * lr / w);   // the square root of the geometric mean ratio
        scale = r < kLoScale ? kLoScale : r > kHiScale ? kHiScale : r;
    }
};

/// Per-request threshold.  The threshold falls from the base toward kLo as the
/// request's drafts are accepted more often than the bin table predicts: when they run well, longer windows pay.  They
/// rise from the base toward kHi times the base when drafts are accepted less often than it predicts: when they run
/// badly, the extra verified tokens cost more than they return.  The observed and expected accepted counts decay per
/// window; the ratio sets the threshold along a piecewise-linear ramp.  A window with no draft offered carries no
/// evidence and moves the ratio back to the one that gives the base, so a raised threshold cannot hold itself up.
/// The window's value is fixed before its drafts are made and shared by the host rule and the device graph.
struct DraftCumAdapt {
    float base = draft_cum_base();   ///< the threshold while nothing is known
    bool ramp = platform::fork_switch("THRESH_RAMP");   ///< the threshold may rise above the base; false: it only falls
    bool miss_aware = platform::fork_opt_in("MISS_THRESH");   ///< recent CPU expert misses scale the threshold
    double obs = 0.0, exp = 0.0;
    float cur = base;
    MissTrend trend;

    float th() const { return cur * trend.scale; }

    /// One verified window of `t` tokens in which the CPU computed `misses` distinct (layer, expert) pairs over
    /// `layers` layers.  Not for windows that read prompt tokens.
    void note_misses(int t, int64_t misses, int layers) {
        if (miss_aware && layers > 0 && misses >= 0) trend.observe(t, (double) misses / (double) layers);
    }

    /// A batch slot's verified window; suffix proposals do not update the MTP acceptance rule.
    void observe_batch_window(const float* probs, int tokens, int accepted, bool from_suffix) {
        if (tokens > 0 && !from_suffix) observe(probs, tokens - 1, accepted);
    }

    /// One verified window of the MTP chain: `n` drafts were offered with probabilities probs[0..n), `acc` accepted.
    void observe(const float* probs, int n, int acc) {
        constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1, kRmin = 0.7, kIdle = 0.95;
        constexpr float kHi = 1.3f;
        const float kLo = 0.25f * draft_cum_scale();
        if (n <= 0) {
            if (!ramp || exp < 0.5) return;
            obs = obs * kIdle + kRlo * exp * (1.0 - kIdle);   // the ratio drifts to the one that gives the base
        } else {
            double c = 1.0, e = 0.0;
            for (int j = 0; j < n; ++j) { c *= (double) draft_cum_acc(probs[j]); e += c; }
            obs = obs * kDecay + (double) acc;
            exp = exp * kDecay + e;
        }
        if (exp < 0.5) return;
        const double r = obs / exp;
        if (ramp && r < kRlo) {
            const double u = std::fmin(std::fmax((kRlo - r) / (kRlo - kRmin), 0.0), 1.0);
            cur = base + (float) u * (kHi * base - base);
            return;
        }
        const double t = std::fmin(std::fmax((r - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::fmin(kLo, base) - base);
    }
};

}  // namespace basalt::core
