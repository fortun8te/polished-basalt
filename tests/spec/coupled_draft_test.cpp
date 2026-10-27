// tests/spec/coupled_draft_test.cpp - coupled draft sampling on a synthetic model (no GPU, no weights).
//
// The target and the drafter are hash-seeded logit tables over a small vocabulary; the sampler chain, the Philox
// counters and the window loop mirror the engine's (sampler.cu tail, verify.cpp run_end, mtp.cpp draft_rows).  Checks
//   1. the text is the plain sampler's, token for token, whatever the drafts are: greedy, coupled, drafts from an
//      independent uniform, and drafts with a wrong counter - for the sampler's inverse-CDF pick and for a Gumbel-max
//      pick (the argmax of p_i / E_i, E_i exponential, keyed by seed, counter and token id);
//   2. at temperature 1 the coupled draft is accepted more often than the argmax draft and than a draft from an
//      independent uniform, and not above the bound sum(min(p, q)) that any coupling has;
//   3. the counter arithmetic matters: one position off, the gain is gone;
//   4. the inverse-CDF coupling gains little once the drafter's ranking drifts from the target's, where the Gumbel-max
//      coupling keeps most of the bound (it needs the target's own pick to be Gumbel-max too);
//   5. a cold request does not lose much to coupling (the floor in coupled_draft_enabled sends those to the argmax).
// The printed rates are for the synthetic model only; they say nothing about the real drafter.
#include "basalt/core/coupled_draft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using basalt::core::coupled_draft_cell;
using basalt::core::coupled_draft_counter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int V = 48;

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// the sampler's uniform: a function of (seed, counter) only
double uni(uint64_t seed, uint64_t counter) { return (double) (mix(seed * 0x2545f4914f6cdd1dull + counter) >> 11) / 9007199254740992.0; }
double gauss(uint64_t key, int i) {
    const double u1 = ((double) (mix(key + 2 * (uint64_t) i) >> 11) + 1.0) / 9007199254740993.0;
    const double u2 = (double) (mix(key + 2 * (uint64_t) i + 1) >> 11) / 9007199254740992.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

uint64_t ctx_key(const std::vector<int>& c) {
    const size_t n = c.size();
    return mix((uint64_t) c[n - 1] * 1000003ull + (uint64_t) c[n - 2] * 7919ull + 17);
}
constexpr double kSigma = 3.0;
void target_logits(const std::vector<int>& c, double* l) {
    const uint64_t k = ctx_key(c);
    for (int i = 0; i < V; ++i) l[i] = kSigma * gauss(k, i);
}
void draft_logits(const std::vector<int>& c, double* l, double noise) {
    const uint64_t k = ctx_key(c);
    for (int i = 0; i < V; ++i) l[i] = kSigma * gauss(k, i) + noise * gauss(k ^ 0xabcdef12345ull, i);
}

struct Params { int top_k = 20; double top_p = 0.95, temp = 1.0; };
struct Dist { int n = 0; int id[V]; double p[V]; };

// sampler.cu: top_k by (logit desc, id asc); top_p over the raw softmax of those; temperature; renormalise
Dist chain(const double* logit, const Params& sp) {
    std::vector<int> ord(V);
    for (int i = 0; i < V; ++i) ord[(size_t) i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return logit[a] > logit[b] || (logit[a] == logit[b] && a < b); });
    const int k = std::min(sp.top_k, V);
    double mx = logit[ord[0]], sum = 0.0, e[V];
    for (int i = 0; i < k; ++i) sum += (e[i] = std::exp(logit[ord[(size_t) i]] - mx));
    int keep = k;
    double cum = 0.0;
    for (int i = 0; i < k; ++i) {
        cum += e[i] / sum;
        if (cum >= sp.top_p) { keep = i + 1; break; }
    }
    Dist d;
    d.n = keep;
    double s2 = 0.0;
    for (int i = 0; i < keep; ++i) {
        d.id[i] = ord[(size_t) i];
        s2 += (d.p[i] = std::exp((logit[ord[(size_t) i]] - mx) / sp.temp));
    }
    for (int i = 0; i < keep; ++i) d.p[i] /= s2;
    return d;
}
int pick(const Dist& d, double u) {
    double cum = 0.0;
    for (int i = 0; i < d.n; ++i) { cum += d.p[i]; if (u < cum) return d.id[i]; }
    return d.id[d.n - 1];
}

enum class Draft { Greedy, Coupled, Independent, WrongCounter };
enum class Pick { Cdf, Gumbel };

// a Gumbel-max pick: the argmax of p_i / E_i with E_i = -log U(seed, counter, i); a token keeps its noise whatever
// the other tokens of the list are
int pick_gumbel(const Dist& d, uint64_t seed, uint64_t counter) {
    int best = d.id[0];
    double bv = -1.0;
    for (int i = 0; i < d.n; ++i) {
        const double e = -std::log(uni(seed ^ (0x1234567ull * (uint64_t) (d.id[i] + 1)), counter) + 1e-300);
        if (d.p[i] / e > bv) { bv = d.p[i] / e; best = d.id[i]; }
    }
    return best;
}
int pick_with(Pick how, const Dist& d, uint64_t seed, uint64_t counter) {
    return how == Pick::Cdf ? pick(d, uni(seed, counter)) : pick_gumbel(d, seed, counter);
}

struct Result { std::vector<int> text; long drafted = 0, accepted = 0, windows = 0, first_ok = 0; };

// the engine's window loop: window of T tokens, drafts made after the previous window's commit
Result run(Draft how, int T, int n_tokens, uint64_t seed, const Params& sp, double noise, Pick mode = Pick::Cdf) {
    Result r;
    std::vector<int> ctx = {3, 11, 5};
    r.text.clear();
    // the first window is the last prompt token alone
    int64_t P = (int64_t) ctx.size() - 1;
    std::vector<int> drafts;
    bool first = true;
    int a = 0;
    int64_t prev_p = 0;
    while ((int) r.text.size() < n_tokens) {
        if (!first) {
            drafts.clear();
            std::vector<int> dctx = ctx;
            for (int j = 0; j < T - 1; ++j) {
                double l[V];
                draft_logits(dctx, l, noise);
                const Dist q = chain(l, sp);
                const uint64_t ctr = coupled_draft_counter(coupled_draft_cell(prev_p, a, j));
                int d;
                switch (how) {
                    case Draft::Greedy: d = q.id[0]; break;
                    case Draft::Coupled: d = pick_with(mode, q, seed, ctr); break;
                    case Draft::Independent: d = pick_with(mode, q, seed ^ 0x77777ull, ctr); break;
                    default: d = pick_with(mode, q, seed, ctr + 1); break;
                }
                drafts.push_back(d);
                dctx.push_back(d);
            }
        }
        const int Tw = first ? 1 : T;
        const int64_t pos0 = (int64_t) ctx.size() - 1;
        std::vector<int> vctx = ctx, out;
        for (int t = 0; t < Tw; ++t) {
            double l[V];
            target_logits(vctx, l);
            out.push_back(pick_with(mode, chain(l, sp), seed, (uint64_t) (pos0 + t)));
            if (t + 1 < Tw) vctx.push_back(drafts[(size_t) t]);
        }
        int acc = 0;
        while (acc < Tw - 1 && drafts[(size_t) acc] == out[(size_t) acc]) ++acc;
        if (!first) { r.drafted += Tw - 1; r.accepted += acc; ++r.windows; r.first_ok += acc >= 1; }
        for (int i = 0; i <= acc; ++i) { ctx.push_back(out[(size_t) i]); r.text.push_back(out[(size_t) i]); }
        prev_p = pos0;
        a = acc;
        first = false;
        (void) P;
    }
    r.text.resize((size_t) n_tokens);
    return r;
}

std::vector<int> plain(int n_tokens, uint64_t seed, const Params& sp, Pick mode = Pick::Cdf) {
    std::vector<int> ctx = {3, 11, 5}, text;
    while ((int) text.size() < n_tokens) {
        double l[V];
        target_logits(ctx, l);
        const int t = pick_with(mode, chain(l, sp), seed, (uint64_t) ctx.size() - 1);
        ctx.push_back(t);
        text.push_back(t);
    }
    return text;
}

// the share of windows whose first draft is accepted
double rate(Draft how, const Params& sp, double noise, Pick mode = Pick::Cdf, int T = 4) {
    long w = 0, a = 0;
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        const Result r = run(how, T, 3000, seed, sp, noise, mode);
        w += r.windows;
        a += r.first_ok;
    }
    return (double) a / (double) w;
}

// the first draft's bound over the plain text's contexts: sum over tokens of min(p, q), the best any coupling does
double min_bound(const Params& sp, double noise) {
    double tot = 0.0;
    int n = 0;
    std::vector<int> ctx = {3, 11, 5};
    for (int i = 0; i < 4000; ++i) {
        double lt[V], ld[V];
        target_logits(ctx, lt);
        draft_logits(ctx, ld, noise);
        const Dist p = chain(lt, sp), q = chain(ld, sp);
        double s = 0.0;
        for (int a = 0; a < p.n; ++a)
            for (int b = 0; b < q.n; ++b)
                if (p.id[a] == q.id[b]) s += std::min(p.p[a], q.p[b]);
        tot += s;
        ++n;
        ctx.push_back(pick(p, uni(9, (uint64_t) i)));
    }
    return tot / n;
}
}  // namespace

int main() {
    std::printf("coupled_draft_test\n");
    check(coupled_draft_counter(coupled_draft_cell(100, 2, 0)) == 103 && coupled_draft_counter(coupled_draft_cell(100, 2, 3)) == 106,
          "counter of a draft = the next window's pos0 + j (p + a + 1 + j)");
    const Params hot;
    Params cold = hot;
    cold.temp = 0.3;
    for (Pick mode : {Pick::Cdf, Pick::Gumbel}) {
        bool same = true;
        for (uint64_t seed = 1; seed <= 4; ++seed) {
            const std::vector<int> ref = plain(1500, seed, hot, mode);
            for (Draft how : {Draft::Greedy, Draft::Coupled, Draft::Independent, Draft::WrongCounter})
                for (int T : {2, 4, 6}) same = same && run(how, T, 1500, seed, hot, 0.8, mode).text == ref;
        }
        check(same, mode == Pick::Cdf ? "inverse-CDF pick: text = the plain sampler's for every draft rule, window size, seed"
                                      : "Gumbel-max pick: text = the plain sampler's for every draft rule, window size, seed");
    }
    const double close = 0.3, far = 0.8;
    const double greedy = rate(Draft::Greedy, hot, close), coupled = rate(Draft::Coupled, hot, close),
                 indep = rate(Draft::Independent, hot, close), wrong = rate(Draft::WrongCounter, hot, close);
    const double bound = min_bound(hot, close);
    std::printf("  first draft accepted, T=1, close drafter: greedy %.3f  coupled %.3f  independent %.3f  wrong counter %.3f"
                "  (bound %.3f)\n", greedy, coupled, indep, wrong, bound);
    check(coupled > greedy + 0.05, "close drafter, T=1: coupled draft accepted more often than the argmax draft");
    check(coupled > indep + 0.05, "... and than a draft from an independent uniform");
    check(coupled > wrong + 0.05, "... and than the same draft with the counter one position off");
    check(coupled <= bound + 0.02, "... and not above the bound sum(min(p, q))");
    const double gf = rate(Draft::Greedy, hot, far), cf = rate(Draft::Coupled, hot, far),
                 gumf = rate(Draft::Coupled, hot, far, Pick::Gumbel), gumg = rate(Draft::Greedy, hot, far, Pick::Gumbel);
    std::printf("  first draft accepted, T=1, far drafter: greedy %.3f  coupled(CDF) %.3f  coupled(Gumbel) %.3f  (Gumbel target, greedy draft %.3f)"
                "  (bound %.3f)\n", gf, cf, gumf, gumg, min_bound(hot, far));
    check(gumf > cf + 0.1 && gumf > gf + 0.1, "far drafter: the Gumbel-max coupling keeps well above the CDF coupling and the argmax draft");
    const double cg = rate(Draft::Greedy, cold, close), cc = rate(Draft::Coupled, cold, close);
    std::printf("  first draft accepted, T=0.3, close drafter: greedy %.3f  coupled %.3f\n", cg, cc);
    check(cc >= cg - 0.1, "T=0.3: coupled is not far below the argmax draft");
    const double g2 = rate(Draft::Greedy, hot, 0.0), c2 = rate(Draft::Coupled, hot, 0.0);
    std::printf("  first draft accepted, T=1, drafter equal to the target: greedy %.3f  coupled %.3f\n", g2, c2);
    check(c2 > 0.99, "a drafter equal to the target: coupled drafts are always accepted");
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
