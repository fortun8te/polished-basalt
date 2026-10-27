// tests/spec/draft_cum_test.cpp - the cumulative draft rule's per-request threshold (no model, no GPU).
//
//   1. with the ramp off, the threshold is bit for bit the one-way rule's (a copy of it is the oracle);
//   2. with the ramp on, a request whose drafts are accepted as predicted or better gets that same threshold;
//   3. drafts accepted less often than predicted raise the threshold, never beyond 1.3 times the base, and shorten
//      the windows the rule keeps;
//   4. drafts accepted more often than predicted still lower it to kLo;
//   5. a raised threshold comes back when no draft is offered, and when acceptance recovers;
//   6. the same holds at the heavier CPU-expert scale.
#include "basalt/core/draft_cum.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

using basalt::core::DraftCumAdapt;
using basalt::core::draft_cum_acc;
using basalt::core::draft_cum_cpu_mb;
using basalt::core::draft_cum_scale;
using basalt::core::draft_keep;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the rule before the ramp: the threshold only falls from the base
struct OneWay {
    float base = basalt::core::draft_cum_base();
    double obs = 0.0, exp = 0.0;
    float cur = base;
    void observe(const float* probs, int n, int acc) {
        if (n <= 0) return;
        constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1;
        const float kLo = 0.25f * draft_cum_scale();
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double) draft_cum_acc(probs[j]); e += c; }
        obs = obs * kDecay + (double) acc;
        exp = exp * kDecay + e;
        if (exp < 0.5) return;
        const double t = std::fmin(std::fmax((obs / exp - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::fmin(kLo, base) - base);
    }
};

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double) (s >> 11) / 9007199254740992.0;
    }
};

// Drafts of the given probabilities, each accepted with `skill` times the table's rate (one draft past the first miss
// is not tested): the count a window would report.
int accepted_of(const float* probs, int n, double skill, Rng& r) {
    int a = 0;
    while (a < n && r.next() < std::fmin(1.0, skill * (double) draft_cum_acc(probs[a]))) ++a;
    return a;
}

// The window the rule keeps for these probabilities at this threshold.
int window_len(const float* probs, int max_t, float th) {
    int T = 1;
    while (T < max_t && draft_keep(probs, T - 1, th)) ++T;
    return T;
}

// Run the threshold through windows of `skill`; the offered drafts are those the rule keeps at the current threshold.
template <class A>
void run(A& a, double skill, int windows, Rng& r, const float* probs, int max_t) {
    for (int w = 0; w < windows; ++w) {
        const int n = window_len(probs, max_t, a.cur) - 1;
        a.observe(probs, n, accepted_of(probs, n, skill, r));
    }
}
}  // namespace

int main() {
    std::printf("draft_cum_test\n");
    const float probs[8] = {0.97f, 0.93f, 0.88f, 0.85f, 0.8f, 0.75f, 0.7f, 0.6f};
    {
        Rng r1;
        DraftCumAdapt a;
        a.ramp = false;
        OneWay o;
        bool same = true, up = false;
        for (double skill : {1.0, 0.5, 0.8, 1.2, 1.5, 0.3}) {
            for (int w = 0; w < 200; ++w) {
                const int n = window_len(probs, 8, o.cur) - 1;
                const int acc = accepted_of(probs, n, skill, r1);
                a.observe(probs, n, acc);
                o.observe(probs, n, acc);
                same = same && a.cur == o.cur && a.obs == o.obs && a.exp == o.exp;
                up = up || a.cur > a.base;
            }
        }
        check(same, "ramp off: the threshold is the one-way rule's, bit for bit");
        check(!up, "ramp off: the threshold never rises above the base");
    }
    {
        Rng r1;
        DraftCumAdapt a;
        OneWay o;
        bool same = true;
        for (double skill : {1.0, 1.15, 1.4}) {
            a = DraftCumAdapt();
            o = OneWay();
            for (int w = 0; w < 300; ++w) {
                const int n = window_len(probs, 8, o.cur) - 1;
                const int acc = accepted_of(probs, n, skill, r1);
                a.observe(probs, n, acc);
                o.observe(probs, n, acc);
                // while the evidence says "as predicted or better" the two agree
                if (a.exp >= 0.5 && a.obs / a.exp >= 0.9) same = same && a.cur == o.cur;
                else break;
            }
        }
        check(same, "ramp on: acceptance as predicted or better gives the one-way threshold");
    }
    {
        Rng r;
        DraftCumAdapt a;
        const float base = a.base;
        run(a, 0.75, 400, r, probs, 8);
        const float mid = a.cur;
        check(mid > base, "drafts accepted less than predicted: the threshold rises above the base");
        DraftCumAdapt z;
        float top = z.cur;
        for (int w = 0; w < 400; ++w) {
            z.observe(probs, 5, 0);   // nothing is ever accepted
            top = std::max(top, z.cur);
        }
        check(top <= 1.3f * base * 1.0001f && top >= 1.3f * base * 0.999f, "it tops out at 1.3 times the base");
        const float flat[8] = {0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f};
        bool mono = true;
        for (float th = 0.2f; th < 0.7f; th += 0.01f)
            mono = mono && window_len(flat, 8, th + 0.01f) <= window_len(flat, 8, th);
        check(mono, "a higher threshold keeps no longer a window");
        check(window_len(flat, 8, top) < window_len(flat, 8, base), "and for confident drafts a shorter one at the top");
    }
    {
        Rng r;
        DraftCumAdapt a;
        run(a, 1.35, 400, r, probs, 8);
        check(a.cur < a.base, "drafts accepted more than predicted: the threshold falls below the base");
        check(std::fabs(a.cur - 0.25f * draft_cum_scale()) < 0.02f, "and reaches kLo when they are accepted far more");
    }
    {
        Rng r;
        DraftCumAdapt a;
        run(a, 0.5, 300, r, probs, 8);
        const float high = a.cur;
        check(high > a.base * 1.2f, "a poor stretch raises the threshold well above the base");
        run(a, 1.2, 300, r, probs, 8);
        check(a.cur < a.base, "acceptance recovering brings it back below the base");
        DraftCumAdapt b;
        run(b, 0.5, 300, r, probs, 8);
        const float stuck = b.cur;
        for (int w = 0; w < 60; ++w) b.observe(probs, 0, 0);   // windows with no draft offered
        check(stuck > b.base && b.cur < stuck, "windows with no draft offered pull a raised threshold back");
        for (int w = 0; w < 400; ++w) b.observe(probs, 0, 0);
        check(std::fabs(b.cur - b.base) < 1e-3f, "and after enough of them it is the base again");
    }
    {
        draft_cum_cpu_mb() = 3.5f;   // the heaviest CPU-served expert share
        Rng r;
        DraftCumAdapt a;
        const float base = a.base;
        run(a, 0.4, 500, r, probs, 8);
        check(std::fabs(base - 0.525f) < 1e-6f && a.cur <= 1.3f * base * 1.0001f && a.cur > base,
              "scaled base 0.525: the raised threshold stays within 1.3 times it");
        draft_cum_cpu_mb() = 0.0f;
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
