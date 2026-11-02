# Polished Basalt on upstream 0.2.0

Branch `polished-basalt-0.2-merge`: upstream `main` at `f0bb833c` (0.2.0, 2026-11-01) with the fork's patches
cherry-picked onto it one by one (`git cherry-pick -x`, so every commit names its original). A plain merge of
`polished-basalt` gave 30 conflicts because upstream split `src/program/generate.cpp` into `src/program/{cli,generate,serve}/`
and `src/core/{mtp,verify}.cpp` and `src/prefill/prefill.cpp` into directories; each fork hunk on those files was moved to
the file that now holds its code.

## Dropped: upstream now does it

| Fork commit | What | Upstream replacement |
|---|---|---|
| `4c5f2feb` (+ test `3af495aa`) | AVX-512 sign masks via pdep (#86) | #111 `bacebb52`: pdep sign masks and gathered IQ3_S / IQ2_S grids, same bits |
| `bc503477` | EcoQoS opt-out (#93) | #109 `23c6c941`: `keep_full_speed()` at startup (process-wide; the fork's per-worker call is gone) |
| `1eb480b5` | `--conversation-cache-min-tokens` (#67) | `463d92e8`, same flag |
| `10ac690c` | answer room (#95) | #97 `e229e7d9`: `answer-room`, on by default; the thinking budget closes with the template's own tag |
| `92269016` (budget part) | explicit budgets beat answer room | upstream's #97 logic kept; `test_hardening_budgets.py` removed |
| `269783cc`, `0fc1eae9`, `e56b8458` | pool: batch ends with its last job, parked worker yields | #98 `c1850c40` and later pool commits |
| `d8ef97b2` | concurrent prompt-cache improvements | already in upstream `src/program/serve/` (`move_routed_below`, `partial_chunk`) |
| `997e931c` (ordering part) | strict tool-call ID rejection | upstream orders results by call id and keeps received order when IDs are ambiguous; the fork's truncated/undeclared/duplicate parameter refusal is kept |
| 9 commits whose subject is in upstream verbatim | cherry-picks of upstream work that later landed | the upstream commits |

## Ported to the new layout

PLE table from the GGUF shard (`--ple-gguf`, now threaded through `generate::Context::ple_path`); Windows slice pinning
(`complement_sliced_` kept beside upstream's sleep/wake `unregister_complement`); pinned headroom; PCIe staging for single
requests (`stage_solo` in `verify/expert_plan.hpp`, `staged` in `verify/record_layer.cpp` / `record_moe.cpp`) and its
`BASALT_FORK_STAGING` switch; `fork_switch` / `fork_number` / `fork_opt_in`; GDN two-heads (opt-in); `DRAFT_CUM_MULT`,
`SPLIT_COST`, `THRESH_RAMP`; batch-window threshold recovery (`serve/batch.cpp`); the CPU warm-up's per-layer layout and
three-expert read (`verify/cpu_warm.cpp`); HTTP, vision-source, image-size, artifact and tokenizer hardening; the API key file;
`/chat`; the scheduler, profiles, Windows scripts and checks.

## Not ported yet (need a hand port onto `src/program/serve/`)

| Fork commit | Why it waits |
|---|---|
| `e85b1ef9`, `db01a98c` (#89 checkpoint routing) | the engine's `SCACHE` report and the BDONE/DONE ordering live in `serve/batch.cpp`, `request_finish.cpp`; upstream's own "stored conversation keeps its system prompt and last three turn states" covers part of the need |
| `9d8f4052`, `df27de1c` (terra_scheduler, engine-frame faults) | built on the #89 reader loop in `serve/engine/process.py` |
| `bab2edbc` (stop on a failed adaptive transfer) | the adaptation loop moved to `serve/tiers.cpp` and changed (adaptive swaps) |
| `b92c4829`, `1626ac58` (`LOOKUP_OVERRIDE`, `MISS_THRESH`) | off by default; their decode hooks moved; switches removed from `fork_launch.py` |

## Checked here

No CUDA on the Mac, so CMake (`LANGUAGES C CXX CUDA`) does not configure and nothing CUDA-side was compiled. Host-only
`-fsyntax-only` passes for every changed C++ file that does not include `cuda_runtime.h` (cli, model_file, verified_file,
draft_policy, vision preprocess/mmproj, `tools/vision/vision.cpp`, the spec/platform/vision/artifact tests); `coupled_draft_test`
builds and passes. Python: serve + fork suites pass (926+ tests) after the adjustments above.

## Expected speed

Upstream 0.2.0 changelog: decode about +13% (prose 297 to 330-337 tok/s, struct 558 to about 600) on an RTX 5090 + 5060 Ti
with the NVFP4 pack, greedy output identical; levers are the 2-bit draft-head screen, the two-stage Q8_0 output head,
earlier adaptation rounds and better starting placement. The community table (`docs/community/benchmarks.json`) has one
RTX 5080 run on 0.1.x (WSL, IQ3_S): decode 137.9 tok/s, prefill 3180 tok/s at 16K and 4352 at 64K. No 0.2.0 run on a 5080 or
with IQ3_S exists. Applied to the fork's 115 tok/s (thinking off, short coding) the upstream lever gives roughly 125-130 tok/s,
and 105 at 255K roughly 112-118; both are estimates to be measured with `basalt-bench --community`, not results.

## Q2_0

0.2.0 fixes loading packs with small expert blobs (Q2_0): the bulk read staging buffers are aligned to the direct-read
alignment. An official `Qwen3.8-FN-GSQ-RCO-Q2_0.basalt` is on Hugging Face (`jesdga/Qwen3.8-Flash-Next-Basalt`). Another
Q2_0 GGUF (e.g. an abliterated one) must be converted: `python3 tools/convert/convert.py --gguf <shard 1> --mmproj <mmproj>
--out X.basalt [--compat-bf16]`; without the official tuned drafter it decodes markedly slower, so pass
`--mtp-dense` with the `mtp/dense` section of the official Q2_0 pack when the shapes match.

## Deploying

Rebuild (format v2 changed), migrate or re-download v1 packs (`tools/migrations/basalt_v1_to_v2.py`), and drop removed flags
(`--dump-hc8`, converter `--profile`/`--profile-from`/`--hc8`, server `repeat_stop_tokens`) from scripts and profiles.
