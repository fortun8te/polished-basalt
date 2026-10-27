# How much faster is Polished Basalt, honestly?

Nothing here has been run on the 5080 PC. Every number is an estimate built from your own Codex measurements and other people's
reports. The first real A/B (see `TEST_PLAN.md`) replaces this page.

## Baseline (your measurements, upstream commit b2fe96bf, tuned profile)

| Workload | Tokens/s |
|---|---:|
| Short coding, median of 3 | 115.1 |
| Prose | 84.2 |
| Near-full 255K context | 104.9 |
| Fresh prompt ingestion | ~3,500 |

## What your own logs say about where the time goes

From the Codex runs (`linux-comparison-20261027` logs, native-Windows arm):

- The 46.3 GiB expert copy **was** page-locked whole (`pinned 46.32 GiB in 12 s`). The Windows pin cap only bites on 64 GB
  machines, so the Windows slice-pinning patch (tarhue, basalt #73) is a safety net for you, not a speed-up.
- `pcie=0.0 %` on every request. The PCIe staging path never ran, because it is off for single requests and for mixed-type
  packs like GSQ-RCO IQ3_S (basalt #82). **That is the one engine change with real upside on your box.**
- Expert-cache hit rate was only 70–79 %: roughly one expert in four is computed by the CPU every token.
- MTP draft acceptance 63–68 % at the sampled settings (T=1.0, top_p 0.95, top_k 20).
- n-gram table read with the default `direct` route; the matched `ple-io ram` test was +8 % decode.

## Estimated effect per change (decode, your box)

| Change | Source | Estimate | Confidence | Numerics |
|---|---|---:|---|---|
| `ple-io ram` (profile only) | Codex test, basalt #57 | +5–8 % decode; prefill up to several x on non-repetitive text | medium (one pass) | none |
| PCIe staging for single requests / mixed packs | basalt #82 | +2–8 % (reporter: +14 % when many experts are on file, +2.4 % when all are in RAM; yours are all in RAM) | medium-low | staged experts run GPU kernels, tiny drift |
| AVX-512 pdep sign masks | basalt #86 | +1–2 % | medium | bit-identical (proved by emulation) |
| Adaptive-tier tuning (`adapt-swaps 160`, `adapt-decay 0.97`) | Strata #922 | 0–5 % | low | tier placement changes |
| Windows power-throttling opt-out | basalt #93 | ~0 % in foreground, protects background runs | medium | none |
| Sampled-MTP drafting (spec agent, default OFF) | Strata #3908 | +10–20 % on sampled (T=1.0) decode if it ports cleanly | low (unreviewed port) | output distribution unchanged by design |

**Central estimate: +8 to +15 % decode (115 → roughly 125–132 tok/s), +20 % with everything landing and sampled drafting working.**
Prefill gains come from `ple-io ram` only. Reaching 140–150 tok/s on engine work alone is possible but not likely; 200 tok/s is not
reachable on one 5080 with this engine (published 5080 results of 186.8 tok/s are native Linux *aggregate* numbers on a 16-core
CPU).

## Why the ChatGPT "+30–40 %" headline does not transfer

The 105.7 → 141.5 tok/s result on a 5080 + 9800X3D (basalt #82 comment) was on a **64 GB** PC where the whole expert copy could not
be page-locked (pinned 0 GiB, then 36.81 GiB after slice pinning), plus four patches at once (pinning, staging, #86, PLE fix). On
128 GB the pinning half of that gain is already yours.

## Where the big wins are instead (time to a finished task, not tok/s)

| Lever | Why it matters | Estimate |
|---|---|---|
| Fewer thinking tokens (Swift 1.5 IQ3_S, or per-step reasoning effort) | thinking tokens dominate wall time; 1,000 tokens = ~10 s | −10 to −45 % tokens; ~1.4–1.8x faster per task in small independent tests |
| Keep prefix caches warm (client settings, retry routing #89, `conversation-cache-min-tokens`) | a cold 100K re-read is ~29 s | seconds per turn |
| Priority queue in front of the single slot | background calls stop evicting your interactive conversation | latency, not tok/s |
| Second GPU as a peer expert tier (Linux) | upstream 2x5080 went ~102 → ~200 tok/s | +30–70 %, costs hardware and a Linux boot |

## What would change these numbers

`pcie=` in the Basalt "done" lines after the staging patch; `hit=` after the adapt tuning; `acc=`/`dl=` for sampled drafting;
`ram_gbps` from `basalt-bench --hardware` (DRAM bandwidth is the ceiling for the CPU-computed fraction).
