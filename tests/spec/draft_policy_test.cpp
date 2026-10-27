// tests/spec/draft_policy_test.cpp - DraftPolicy: when does a lookup window beat the MTP's?
//
// Simulated rounds with measured window costs (about 10 ms more per token) check that
//   1. with no lookup proposal the MTP window is kept;
//   2. lookup drafts that are mostly rejected stop being taken (their bucket's rate falls);
//   3. lookup drafts that are always accepted are taken, and the window grows with them;
//   4. match-length buckets learn separately (short matches failing does not stop long ones);
//   5. the policy never proposes a window beyond its cap;
//   6. with separate costs per source, a lookup round is priced by lookup rounds: cheaper rounds make a marginal
//      lookup worth taking, dearer ones make it not (a single shared table prices both by whichever was seen last);
//   7. with no lookup round measured yet, the separate costs choose exactly as the single table does.
#include "basalt/spec/draft_policy.hpp"

#include <cstdio>

using basalt::spec::DraftPolicy;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double cost(int t) { return 19.0 + 10.5 * (t - 1); }   // ms per round, measured shape

// Lookup rounds of 6 seen first (4 of 5 drafts accepted, or 5 and 4 alternately when `sure`), then MTP rounds of every
// size, which accept two drafts at most; the lookup rounds cost `lk_extra` ms more (or less) than the MTP rounds of
// the same size. Returns the choice for a 5-token proposal after a 20-token match.
DraftPolicy::Pick marginal(bool split, double lk_extra, bool sure, DraftPolicy* out = nullptr) {
    DraftPolicy p(6);
    p.set_split_cost(split);
    for (int i = 0; i < 30; ++i) p.observe(true, 6, sure ? (i % 2 ? 5 : 4) : 4, 20, 38.0 + 10.5 * 5 + lk_extra);
    for (int t = 1; t <= 6; ++t)
        for (int i = 0; i < 30; ++i) p.observe(false, t, t > 1 ? (t - 1 < 2 ? t - 1 : 2) : 0, 0, 38.0 + 10.5 * (t - 1));
    if (out != nullptr) *out = p;
    return p.choose(4, 5, 20);
}
}  // namespace

int main() {
    std::printf("draft_policy_test\n");
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));   // MTP windows of 4: 3 tokens each
        const DraftPolicy::Pick k = p.choose(4, 0, 0);
        check(!k.lookup && k.t == 4, "no proposal: the MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 2, 0, cost(4));
        for (int t = 2; t <= 6; ++t) p.observe(false, t, 0, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 0, 4, cost(6));      // short matches, all rejected
        check(p.lookup_rate(4) < 0.15, "rejected short-match drafts: their rate falls below 0.15");
        check(!p.choose(4, 5, 4).lookup, "rejected short-match drafts: no longer taken");
        for (int i = 0; i < 40; ++i) p.observe(true, 6, 5, 30, cost(6));     // long matches, all accepted
        check(p.lookup_rate(30) > 0.9, "accepted long-match drafts: their rate rises above 0.9");
        const DraftPolicy::Pick k = p.choose(4, 5, 30);
        check(k.lookup && k.t == 6, "accepted long matches: the full lookup window is taken");
        check(!p.choose(4, 5, 4).lookup, "buckets are separate: short matches still not taken");
        check(p.choose(4, 20, 30).t <= 6, "never beyond the window cap");
    }
    {
        DraftPolicy p(8);
        for (int i = 0; i < 50; ++i) p.observe(false, 3, 2, 0, cost(3));      // a very good MTP: 3 of 3 tokens
        for (int t = 2; t <= 8; ++t) p.observe(false, t, t - 1, 0, cost(t));
        for (int i = 0; i < 40; ++i) p.observe(true, 4, 2, 8, cost(4));       // lookup at q ~ 0.67
        check(!p.choose(3, 7, 8).lookup, "a mediocre lookup does not replace a strong MTP window");
    }
    {
        DraftPolicy p(6);
        for (int i = 0; i < 50; ++i) p.observe(false, 4, 3, 0, cost(4));      // a near-perfect MTP, only size 4 seen
        const DraftPolicy::Pick k = p.choose(4, 5, 40);
        check(k.lookup && k.t == 6, "an unmeasured size is probed for a confident lookup");
        for (int i = 0; i < 3; ++i) p.observe(true, 6, 5, 40, 3.0 * cost(6));   // it turns out very expensive
        check(!p.choose(4, 5, 40).lookup, "after the probes, the measured cost decides");
    }
    {
        DraftPolicy p(6);
        const DraftPolicy::Pick shared = marginal(false, -8.0, false), split = marginal(true, -8.0, false);
        check(!shared.lookup && split.lookup, "cheaper lookup rounds: taken with separate costs, not with one table");
        const DraftPolicy::Pick shared_dear = marginal(false, 20.0, true), split_dear = marginal(true, 20.0, true);
        check(shared_dear.lookup && !split_dear.lookup, "dearer lookup rounds: declined with separate costs, taken with one table");
        marginal(true, -8.0, false, &p);
        check(p.cost_ms(6) > 90.0 && p.cost_ms(6) < 91.0, "MTP cost of a size is not moved by lookup rounds");
        check(p.lookup_cost_ms(6) > 82.0 && p.lookup_cost_ms(6) < 83.0, "lookup cost of a size is its own rounds' time");
        check(p.lookup_cost_ms(4) < p.cost_ms(4), "an unmeasured lookup size follows the lookup/MTP ratio");
    }
    {
        // nothing measured for lookup: both modes choose the same, over every match length and a spread of MTP yields
        bool same = true;
        for (int ma = 0; ma <= 4; ++ma) {
            DraftPolicy a(6), b(6);
            a.set_split_cost(false);
            b.set_split_cost(true);
            for (int t = 1; t <= 6; ++t)
                for (int i = 0; i < 10; ++i) {
                    a.observe(false, t, t - 1 < ma ? t - 1 : ma, 0, cost(t));
                    b.observe(false, t, t - 1 < ma ? t - 1 : ma, 0, cost(t));
                }
            for (int m : {3, 8, 15, 40})
                for (int k : {1, 3, 5}) {
                    const DraftPolicy::Pick x = a.choose(3, k, m), y = b.choose(3, k, m);
                    same = same && x.lookup == y.lookup && x.t == y.t;
                }
        }
        check(same, "no lookup round measured yet: separate costs choose as the single table does");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
