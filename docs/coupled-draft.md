# Coupled draft sampling (design, not wired into the engine)

Status: the host arithmetic (`include/basalt/core/coupled_draft.hpp`) and a synthetic-model test
(`tests/spec/coupled_draft_test.cpp`) exist. The GPU side below is not written: it needs `.cu` edits and graph
capture changes that cannot be compiled or run without the GPU box. Switch when wired: `BASALT_FORK_COUPLED_DRAFT=1`
(default off), `BASALT_FORK_COUPLED_MIN_TEMP` (default 0.9).

## Idea
A sampled request's verify row t of a window at `pos0` draws `philox_uniform(seed, pos0 + t)` and keeps draft t only
when it equals the row's pick (`verify.cpp` run_end, `sampler.cu` tail). The MTP draft is its argmax, so a draft is
accepted with probability p_target(argmax). Let the draft layer instead pick from its own distribution, through the
same chain (top_k, top_p, min_p, temperature) and **the same uniform the verifying row will draw**. Two inverse-CDF
picks from one uniform agree more often than an argmax and a sample do.

## Why the text does not change
Verification is untouched, and the row's pick depends only on (seed, counter, the target's distribution given the
accepted prefix). The token at position s is therefore the same whichever drafts were offered; drafts only change how
many tokens a window yields. `coupled_draft_test` checks this against a plain sampler for greedy, coupled,
independent-uniform and wrong-counter drafts, and for window sizes 2, 4, 6. (Batch-shape numerics: the verify kernels
are not batch-invariant across widths, so a near tie can still flip a token, as for any change of window width.)

## Counter arithmetic
After a window at `p` with `a` drafts accepted, the next window starts at `p + a + 1`. Chain step j runs at MTP cell
`p + a + j` and is verified by row j of that window: counter `p + a + 1 + j` = cell + 1. This holds after rejections,
for any window size, when a lookup window is chosen instead (its drafts are simply unused), and for `draft_first` /
`draft_more` (they stage their cell the same way). `mtp.cpp` `draft_rows` already writes the cell of every step into
the step record (`put()`, h_step_ word 0); the kernel can read the counter from there, so the captured graphs replay
with each round's own counter and need no extra host plumbing.

## What it is worth
Synthetic model (48 tokens, top_k 20, top_p 0.95, T = 1; first draft accepted): drafter close to the target
0.52 argmax -> 0.66 coupled (bound 0.91); drafter far from it 0.47 -> 0.48 (bound 0.76). The inverse-CDF coupling is
fragile: when the drafter's ranking drifts from the target's, one displaced entry shifts every later interval. A
Gumbel-max pick on both sides (argmax of p_i / E_i, E_i keyed by seed, counter, token id) keeps 0.74 in the far case,
but then the verifier's own pick changes too: the text per seed changes (the distribution stays exact) and every
sampled request is affected. Strata (the parent project) implements both (`STRATA_SPEC_COUPLED`,
`STRATA_SPEC_GUMBEL`) and states that sampled drafting is "not faster in a way that holds up" and gains at T = 1.0,
loses below about 0.7. Strata PR 3908 reports +22% / +15% at T = 0.7, but against a baseline that did not draft
sampled requests at all, and with rejection sampling (below); it does not isolate coupling.
Estimate for T = 1.0 thinking mode: GUESS +0 to 8% decode tok/s, depending on how close the MTP head is to the target.
The cost side: the fused head + argmax kernels (`q5k_head_top1`, `q5k_proxy_top1`) do not write logits, so a coupled
step needs the logits head plus a top-k split and merge: GUESS 30 to 80 us per chain step.

## Calibration caveat
`draft_cum_acc` (`draft_cum.hpp`, mirrored on the device in `verify_kernels.cu` `mtp_draft_cond_kernel` and
`mtp_step_end_kernel`) maps a draft's probability to the observed acceptance of **argmax** drafts in that bin. A coupled
draft is accepted more often than the bin says when the head is confident and less predictable when it is not, and its
token need not be the top one. Feed the rule the same number as before (the head's top-1 probability) and let
`DraftCumAdapt` absorb the shift: its ratio rises above 1 and the threshold falls, but only to `kLo`. A proper fix
refits the bins from `--draft-log` windows taken with the switch on, in the host table and both device kernels
together (they must stay the same products in the same order).

## Code locations
- `src/core/mtp.cpp:944-972` `record_forward` head: the fused `T == 1` top-1 paths (944-954) must be skipped when
  coupled; the logits paths (955-962) already fill `head_logits_`; the greedy pick (966-972) is replaced by the new
  call. `row_top_prob`/`map_ids` for the subset vocabulary (`dvocab_`, `n_tier1_`) are the mapping to reproduce.
- `mtp.cpp:1024-1075` `record_round` / `record_step`, `1142-1260` `capture_round` / `capture_step` / `capture_cond`:
  the new kernels are plain nodes in these captures (the cond graphs' `ctail_` hook after `step_end` is unchanged,
  because `probs_` and `out_ids_` keep their meaning). Parameters live in device memory, so no per-request recapture.
- `mtp.cpp:1404-1470` `draft_rows`: `put()` (1443-1445) writes the cell; `h_row_` slot 2 is reserved, slot 3 is the threshold.
- `src/kernels/cuda/sampler.cu`: `sampled_tail_warp<kProb>` (374, already takes `prob_out`), `warp_merge_lists` (542),
  the split part and `sampler_split_merge_kernel` (587-690), `apply_penalties` / `history_count` (70-76).
- `include/basalt/kernels/sampler.hpp:15` `SamplerParams`; `include/basalt/core/verify.hpp:160` `set_sampling` (its
  comment says the drafter never sees the request's sampling: that changes).
- `src/program/generate.cpp`: next to `ver.set_sampling(req_sp)` (9918, 9938) and the CLI's (11081); the batch path
  (`draft_seq`, 8020) and the pipelined `draft_from` (10534) stay on argmax in a first version.
- Unchanged on purpose: `verify.cpp:4094-4110` (the exact-match rule), which is what keeps the text.

## Port map (Strata, MIT, same lineage; read only)
`src/kernels/cuda/sampler.cu` 798-935 (`coupled_stage_kernel`: request params and penalty-ring base from mapped host
memory into device copies at the start of each round graph; `coupled_penalize_kernel`; `coupled_merge_kernel`: merge
the split lists, then the tail with `counter = cell + 1`, map the pick to its id, write id and probability, append to
the ring) and 1301-1345 (launchers, scratch size); `src/core/mtp.cpp` 558-619 (`setup_coupled`, `set_sampling`), 1199-1208
(the branch in the forward), 1270-1345 (separate coupled round and step graphs), 1576-1598 (cells);
`include/strata/core/coupled_draft.hpp`.

## Minimal skeleton
```cpp
// sampler.hpp
size_t coupled_draft_scratch_bytes(int nv);
void coupled_draft_stage(const SamplerParams* mapped, SamplerParams* dev, void* stream);   // each round graph
void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const SamplerParams* dev,
                          const int32_t* step_rec, void* scratch, int32_t* out_id, float* out_prob, void* stream);
// mtp.cpp, record_forward, after the logits are written (fused top-1 paths skipped when coupled_rec_):
if (coupled_rec_) { coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, cparams_, step_, cscratch_,
                                         out_ids_, probs_, cs); return true; }
// the kernel: p = *dev; p.counter = coupled_draft_counter(step_rec[0]); top_k lists -> sampled_tail_warp<true>
```
First version: skip requests with penalties or logit bias (the target applies them per row over the window's own
history; Strata keeps a device ring for that), skip batch slots and the two-stage pipeline.

## Rejection sampling instead
Strata's `STRATA_SPEC_PROB` and PR 3908's `q` draw the draft from q with its own stream and accept it with
min(1, p/q), resampling from the residual on a reject. Acceptance reaches sum(min(p, q)), the most any scheme can,
and the distribution stays exact, but the verifier changes (the accept rule, the residual draw, q lists of up to 64
entries per draft passed from drafter to verifier) and the text per seed then depends on the drafts. Not proposed here.

## A/B on the box
Same seed and prompt, switch off and on: the text should match (modulo near ties). Compare `acc=` and `dl=` on the
server's `done` line and tok/s; `--draft-log` `W` lines for the window size and per-draft probabilities;
`--decode-timing` for the draft ms per window (the cost side). `set_draft_probe` gives the target's probability of the
draft token, which separates acceptance from window shape.
