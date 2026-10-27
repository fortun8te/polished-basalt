# Research index: what the upstream trackers and the community say

Collected 27 Oct 2026 by reading every Basalt issue and PR (96 items) and the Strata items on performance, prefill/KV, quality and
hardware reports (Strata has ~4,300 items; coverage there is by theme, not exhaustive). Numbers are self-reported unless marked.

## Closest hardware to ours (16 GB Blackwell/Ada, Windows, X3D, 64–128 GB)

| Hardware | Pack, context | Decode tok/s | Prefill tok/s | Source |
|---|---|---:|---:|---|
| RTX 5080 / 9800X3D / 64 GB / Win 11 / 610.62 | IQ3_XXS, 131K | 114.1 chat, 138.1 agent | 3,273 (4K), 3,863 (16K) | Strata #2383 |
| RTX 5080 / 7800X3D / 64 GB / Win 11 | IQ3_XXS, 262K | 91.9–95.5 | 3,061 (4K), 3,607 (128K) | Strata #1967 |
| RTX 5080 / Core Ultra 9 285K / 64 GB / Fedora | IQ3_XXS, 64K | 150.0 (126.7–170.2) | 3,755 (4K), 4,396 (16K) | Strata #3620 |
| RTX 5080 / WSL2 / 88 GiB | IQ3_S | 137.9 | 3,180 (16K), 4,352 (64K) | Basalt community bench |
| RTX 5070 Ti / 9950X / 128 GB / Win | IQ3_S | 83.7 real (94.3 calibrated) | n/a | Strata #922 |
| RTX 5080 / 9950X3D / 60 GiB / Linux | IQ3_S | 186.8 aggregate | 4,877 (16K), 4,961 (64K) | Basalt PR #90 |
| **Ours (Codex measurement)** | **IQ3_S, 256K** | **115 coding, 84 prose, 105 at 255K** | **~3,500** | local |

Reading: our number is in the normal range for this class. IQ3_XXS packs are ~10–20 % faster (smaller, higher cache hit) at a
higher quality cost (KLD ~0.24 vs 0.155 against Q8_0).

## Upstream items that shaped the fork

| Item | Takeaway |
|---|---|
| Basalt #82 | PCIe staging off for single requests and mixed-type packs: +14 % at 119K on a 16 GB card, +2.4 % with all experts in RAM |
| Basalt #73 comment | On 64 GB Windows the RAM copy could not be page-locked whole; 1 GiB slices fix it (5080 + 9800X3D: 105.7 → 141.5 with four patches) |
| Basalt #57 | Windows `FILE_FLAG_NO_BUFFERING` reads of the n-gram table: prefill 697 → 3,860 tok/s on non-repetitive text with a cached/RAM route |
| Basalt #83 / #62 | RAM guard stopped long prompts on Windows; fixed on main (`d148f19b`), already in our base |
| Basalt #86 / #85 / #93 | AVX-512 pdep rows; GDN two-wave launch on <96 SMs; Windows power throttling (hybrid Intel CPUs only) |
| Basalt #88 | Same PC, WSL2 vs native Windows: h2d 29.6 vs 40.9 GB/s; decode 126 vs 159 tok/s (WSL slower); prefill 16K 3,430 vs 1,782 (Windows slower, the #57 problem) |
| Basalt #95 | Thinking can eat `max_tokens` and return nothing; "answer room" fixes it (ported from Strata 0.1.45) |
| Basalt PR #89 / #67 | Retry/edited-tail checkpoint routing; keep helper calls out of the conversation cache |
| Strata #3908 / #3907 / #3905 | Sampled-MTP drafting (+15–22 % on a 5080 at T=0.7), pool warm-up, barrier removal; see `docs/coupled-draft.md` for why the Basalt port is design-only |
| Strata #965 | Pins Claude Code's per-request `cch` stamp; Basalt already has it |

## Hazards to avoid

- **CUDA 13.2** miscompiles IQ3_S kernels (Strata #892). Stay on 13.0/13.1 (the Codex build uses 13.1).
- **Non-finite logits / "!!!!" output** on Windows CUDA 13 with IQ3_S (Strata #3602, open). If it ever happens, try another driver before blaming the model.
- **TDR resets** around the verify window on some drivers (Strata #961, #267). No safe RTX 50 driver version is documented; 610.47 is what you run now.
- **`--kv q4_0` with MTP and KV streaming**: launch failure (Strata #2403). Keep int8 KV.
- **Tuners can lose speed** (Strata #3978: an adapt trio cut decode 56 → 45 tok/s on 2x5060 Ti). Every adapt change needs an A/B.
- **Vision on Windows** is broken upstream (Basalt #94). Vision is off.
- **Coder-pruned packs**: tool-call loops and CJK failures (Strata #438, #558). Not used.

## Quality notes

- IQ3_S vs Q8_0: top-1 87.3 %, KLD 0.155 (Basalt model card); UD-Q4_K_XL 92.2 % / 0.057; NVFP4 91.5 % / 0.068.
- Int8 KV is the floor: q4_0 KV measured 2.7–3.5x the KL of int8 (Strata #729).
- Route tail-skip (Strata) costs ~+0.01 KL per position on IQ3_S; not in Basalt and not wanted here.
- Sampling: use the model card's thinking-mode settings (T 1.0, top_p 0.95, top_k 20); greedy defaults cause loops (Strata #728).

## Hardware levers outside software

| Lever | Evidence | Estimate |
|---|---|---|
| Second GPU as a peer expert tier (native Linux) | Basalt #52: 2x5080 ~102 → ~200 tok/s; Windows runs one GPU only | +30–70 % |
| Bigger-VRAM card (5090) | Strata docs: ~700 more cached experts per GB; #52 5090 alone 214.6 | +50–90 % |
| DRAM speed/config | One controlled test (DDR4): prefill +3–11 %, decode unchanged; no 2-vs-4-DIMM test exists | 0–5 % |
| Native Linux dual boot | removes the WSL costs and the Windows pin cap | 0–20 %, unmeasured |
