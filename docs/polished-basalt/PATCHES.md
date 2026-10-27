# Polished Basalt: every change on top of upstream, with provenance

Base: upstream [jesdga95/basalt](https://github.com/jesdga95/basalt) `main` at `b6125f60` (27 Oct 2026, includes the Windows
RAM-guard fix `d148f19b`). Everything below sits on top of it as separate commits, so any one can be dropped with `git revert`.

**Status words.** *Upstream-reported*: someone else measured it on hardware. *Fork-written*: written here, never run.
*Proved*: has a test that ran on the Mac. Nothing has run on the 5080 PC yet.

## A. Carried from your Codex build

| Commit | Change | Why |
|---|---|---|
| `7709f608` | `--ple-gguf`: read the n-gram (PLE) table straight from the original GGUF shard; compact `.basalt` pack without the table | Saves a second 87 GB pack on a nearly full SSD. Written by your Codex session, reviewed there; carries its own hash-verification and tests. |
| `41b3865a`, `97bcf381` | `/chat` route and `serve/web/chat.html` | The private chat page your launcher opens. |

## B. Upstream fixes that are not in upstream `main` (open PRs and issue patches)

| Commit | Source | What | Numerics | Kill switch | Status |
|---|---|---|---|---|---|
| `42e22cdc` | basalt #82 (arytek), refined by tarhue | CPU-missed experts are sent over PCIe and computed on the GPU, for single requests and for mixed-quant packs like GSQ-RCO IQ3_S. Stops the `pcie=0.0 %` you see today. | A staged expert runs the GPU kernel instead of the CPU one: tiny drift, bitwise parity test passes upstream, KLD gate not run | `BASALT_FORK_STAGING=0` | Upstream-reported +14 % (5070 Ti, 119K context), +2.4 % when every expert is in RAM |
| `8b1d33de` | basalt #73 comment (tarhue) | If Windows refuses to page-lock the whole RAM copy, lock it in ~1 GiB slices | none | none (only runs on refusal) | Upstream-reported on 5080 + 64 GB. A safety net for you: your 46 GiB copy locked whole in the Codex log |
| `aca5019e` | fork-written | Hold 1 GiB of page-locked headroom across the whole-copy path too, so later pinned buffers still fit | none | `BASALT_FORK_PIN_HEADROOM=0` | Fork-written |
| `e85b1ef9`, `db01a98c` | basalt PR #89 (Marco Muratori, oblq) | Retries and edited conversation tails resume from the right checkpoint instead of re-reading the whole prompt (6,058 of 6,063 tokens reused in their test) | none | none | Upstream-reported; 599 serve tests passed there; passes here (260 tests, 3 skipped for a missing module) |
| `1eb480b5` | basalt PR #67 (michmill1970) | `--conversation-cache-min-tokens N`: helper/one-shot calls no longer evict your long conversation from the RAM cache | none | flag defaults to 0 (off); profiles set 4096 | Upstream-reported |
| `10ac690c` | basalt #95 (arytek), from Strata 0.1.45 | "Answer room": if thinking would use up `max_tokens`, close it early so an answer still fits. Without it, 4 of 4 coding answers at `max_tokens 8192` came back empty | none | `--no-answer-room` | Upstream-reported; its own tests pass |
| `4c5f2feb` | basalt #86 (mitalbert) | Sign masks for the AVX-512 IQ2_XXS/IQ2_XS/IQ3_XXS CPU rows from `_pdep_u64` instead of a table lookup | bit-identical; proved by emulation on every input (3.25e9 comparisons, 0 mismatches) | `BASALT_FORK_PDEP_SIGNS=0` | Written here from the issue text. Reported 15–29 % per thread, ~1.6 % per window. The x86 path has never run |
| `bc503477` | basalt #93 | Opt the process and each pool worker out of Windows power throttling (EcoQoS) | none | `BASALT_FORK_ECOQOS=0` | Fork-written; Windows bodies checked against stubs only. Expect ~0 % in the foreground |
| `59163c8b` + `ab80ba94` | basalt #85 (mitalbert), re-implemented | `dec_gdn_rec` with two value heads per block, so it fits one wave on 84 SMs | bit-identical by construction | **opt-in**: `BASALT_FORK_GDN_REC2=1` | Fork-written CUDA, never compiled or run. Reported ~0.3–1 % |
| `c06e5f22` | fork-written | `BASALT_FORK_DRAFT_CUM_MULT` scales the speculative draft-length threshold (1.0 = unchanged) | none (only which windows are drafted) | the variable itself | Tuning knob for A/B (try 0.8, 1.2) |

## B2. Speculation tuning (fork-written, host-side policy; tests pass on the Mac)

| Commit | Switch | Default | What | Expected |
|---|---|---|---|---|
| `d5e0c238` | `BASALT_FORK_SPLIT_COST` | on | Separate round-cost tables for MTP and lookup windows | 0–2 % on lookup-heavy text (guess) |
| `05fcd039`, `e3474559` | `BASALT_FORK_THRESH_RAMP` | on | Draft-length threshold may rise to 1.3x base when acceptance is poor, not only fall | +1–3 % at T=1.0 (guess) |
| `b92c4829` | `BASALT_FORK_LOOKUP_OVERRIDE` | **off** | Let a trusted lookup replace the MTP's first draft | 0–3 % (guess); `generate.cpp` edit never compiled |
| `1626ac58` | `BASALT_FORK_MISS_THRESH` | **off** | Scale the draft threshold by the trend of CPU expert misses | 0–2 % (guess); `generate.cpp` edit never compiled |
| `f76d8e73`, `58f4c207`, `1b86edcb` | none | n/a | `docs/coupled-draft.md`, `coupled_draft.hpp`, `coupled_draft_test.cpp`: design and host arithmetic for coupled sampled drafting. **Not wired into the engine.** | 0–8 % at T=1.0 if ever ported (guess) |
| `2f1f7f32` | n/a | n/a | `docs/fork.md` lists these switches | |

All of these only change which drafts or how many are verified; the verifier still produces every token. Window width changes, so
a near-tie can resolve differently (the precision page already allows this).

## C. Helpers

| Commit | Change |
|---|---|
| `97d08ac2`, `ad3f9894`, `ab80ba94` | `platform::fork_switch` (on unless `=0`), `fork_number`, `fork_opt_in` (on only if `=1`). One place for every `BASALT_FORK_*` variable. |
| `73c2f89a`–`120d57b7` | `fork/scheduler/`: a stdlib-only proxy in front of the single slot (priority classes, prefix-aware ordering, cancellation, optional thinking policy and cloud fallback). 83 tests pass on Python 3.9/3.11/3.14. See `fork/scheduler/README.md`. |
| `97bcf381` | `fork/windows/` build/start/stop scripts and `fork/profiles/` A/B arms |

## D. Not included on purpose

| Item | Why |
|---|---|
| PR #28 Windows memory seams (large pages, DXGI budgets) | 2,000 lines that change how the expert cache is sized; applies cleanly but is untested. Parked on branch `opt/pr28-windows-memory-seams`. |
| PR #73 | Overlaps #28 and tarhue's slice pinning, 1,100 lines, stale. |
| PR #66 | Does not apply to current main; ~6 ms per turn. |
| PRs #74, #87, #13 | NVFP4/Gyro paths; your pack is IQ3_S. |
| PRs #68–#70, #75 | Vision only; vision is off. |
| `STRATA_ROUTE_TAIL_SKIP`-style routing shortcuts | Not in Basalt; trades quality for speed (KL +0.01 per position measured on IQ3_S). You run full routing on purpose. |
| Relaxed drafting | Changes the output distribution. |
| NVFP4 / Q4_K_XL packs | Experts exceed the Windows pinned-memory cap or cost 25–30 % speed; revisit only if quality failures appear. |

## E. Kill-switch reference

| Variable | Default | Effect |
|---|---|---|
| `BASALT_FORK_STAGING` | on | `0`: no PCIe-staged share |
| `BASALT_FORK_PIN_HEADROOM` | on | `0`: no extra pinned headroom on the whole-copy path |
| `BASALT_FORK_PDEP_SIGNS` | on | `0`: table-based sign masks |
| `BASALT_FORK_ECOQOS` | on | `0`: leave Windows power-throttling alone |
| `BASALT_FORK_GDN_REC2` | **off** | `1`: two-heads-per-block GDN recurrence |
| `BASALT_FORK_DRAFT_CUM_MULT` | 1.0 | draft-length threshold multiplier, 0.25–4 |
| `BASALT_FORK_SPLIT_COST` | on | `0`: one shared draft cost table |
| `BASALT_FORK_THRESH_RAMP` | on | `0`: threshold only falls, as upstream |
| `BASALT_FORK_LOOKUP_OVERRIDE` | **off** | `1`: trusted lookups may replace the MTP's first draft |
| `BASALT_FORK_MISS_THRESH` | **off** | `1`: miss-trend scaling of the draft threshold |

`CONTRIBUTING.md` upstream forbids environment-variable switches, so none of this is suitable to send upstream as is; individual
changes (PCIe staging, pdep, answer room) can be split and offered separately.

## F. Credits

Patch authors: **tarhue** (Windows slice pinning, staging refinement), **arytek** (PCIe staging, answer room, #93),
**mitalbert** (#85, #86), **Marco Muratori** and **oblq** (#89), **michmill1970** (#67), **dkaynarc** (#57 analysis). Basalt itself is
by **jesdga95**; Strata by **Niko1221**. Both are MIT licensed and this fork keeps the licence.
