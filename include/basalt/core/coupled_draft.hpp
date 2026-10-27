// include/basalt/core/coupled_draft.hpp - the host side of coupled draft sampling (BASALT_FORK_COUPLED_DRAFT, off).
//
// A sampled request's verify row t of a window at pos0 draws Philox(seed, pos0 + t) and keeps draft t only when it
// equals that row's pick, so the text is a function of the seed alone whatever the drafts are.  A drafter that samples
// its draft with the target's chain and that same uniform agrees with the pick more often than its argmax does; the
// text does not change.  Design: docs/coupled-draft.md.
//
// Positions are sequence indices.  After a window at p with `a` drafts accepted the next window starts at p + a + 1.
// Chain step j runs at MTP cell p + a + j and is verified by row j of that window, whose counter is p + a + 1 + j.
#pragma once

#include "basalt/platform/system.hpp"

#include <cstdint>

namespace basalt::core {

/// The MTP cell chain step j runs at, after a window at `p` with `a` drafts accepted.
inline int64_t coupled_draft_cell(int64_t p, int a, int j) { return p + a + j; }

/// The Philox counter the verify row that checks the draft made at `cell` will draw.
inline uint64_t coupled_draft_counter(int64_t cell) { return (uint64_t) (cell + 1); }

/// Whether a request's drafts are sampled: the switch is on, the request samples, and its temperature is at least the
/// floor (a colder request's argmax draft is the better guess).
inline bool coupled_draft_enabled(bool greedy, float temperature) {
    static const bool on = platform::fork_opt_in("COUPLED_DRAFT");
    static const double floor_t = platform::fork_number("COUPLED_MIN_TEMP", 0.9, 0.0, 4.0);
    return on && !greedy && temperature > 0.0f && (double) temperature >= floor_t;
}

}  // namespace basalt::core
