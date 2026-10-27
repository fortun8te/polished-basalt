# Test plan for the 5080 PC (nothing here has been run)

Order matters: stop at the first failure. Nothing touches the production server; the fork runs on its own port (18892) and
refuses to start if anything answers on 18891/18892 or less than 14 GiB of VRAM is free.

## 0. Preconditions

- GPU idle and no other Codex job running (they share the GPU and RAM; a running job invalidates both measurements).
- Close the browser/Discord if the monitor is on the 5080 (desktop VRAM matters for the expert cache).
- Driver 610.47 now, 621.01 downloaded but not installed. Avoid CUDA 13.2 (Strata #892: wrong IQ3_S results); keep 13.1.
- Free disk: the fork build needs ~6 GB, nothing else is downloaded.

## 1. Build (CPU only, ~4–6 min)

1. Copy the repo working tree (without `.git`) to `C:\Users\pc-agent\AgentWorkspace\basalt-5080\src`.
2. Run `fork\windows\Build-Fork.cmd`. It builds `basalt`, `basalt-probe`, `basalt-bench` and the CPU test targets into
   `...\basalt-5080\build`, separate from `ai200-20261026`.
3. `ctest --test-dir ...\build -R "iq_signs_test|iq_avx512_signs_test|platform_system_test|file_expert_source_test|external_ple_test|verified_file_test|pool_native_test|model_file_test"` must be 100 % green.
   `iq_avx512_signs_test` has never run on real AVX-512 hardware before this step.

## 2. Start-up checks (read the log, no measurements yet)

Start arm `a0-daily-control`:

```
.\Start-ForkBasalt.ps1 -Profile a0-daily-control
```

Look for, in order:

| Log line | Good | If not |
|---|---|---|
| `power throttling opt-out` | "process opted out" | warning with an OS error: harmless, note it |
| `FileExpertSource: ... page-locked and mapped` | whole copy pinned (46 GiB in ~12 s) | slice message is also fine |
| `the staged share: a window's layer with 3+ CPU misses ...` | staging armed | message says why not |
| `mid_chain: device not covered` | **absent** | present means the fused decode path is off: stop and tell us |
| `large pages` note | "2097152 B" | 4 KiB: needs "Lock pages in memory" and a fresh sign-in; skip |
| `PLE table ... (--ple-io ram)` | only on arms a1+ | |

## 3. Correctness before speed

- Greedy identity: fixed 300-token greedy run with `BASALT_FORK_*=0` for every switch vs defaults. Differences are expected only
  from the PCIe-staging patch and tier placement (see the patch table); everything else must be bit-identical.
- Your frozen 12-case suite and the 391 arithmetic check on arm a0, then a1.
- Answer-room: one request with `max_tokens 8192` and a hard prompt. The log must show `thinking limited to ... so the answer
  has room` and the reply must contain a final answer.

## 4. A/B arms (profiles in `fork/profiles/`)

Interleave ABAB, 3+ runs per arm, same frozen requests, discard warm-up, record `tg`, `acc`, `hit`, `pcie`, `dl`.

| Arm | Changes vs previous | Question it answers |
|---|---|---|
| a0-daily-control | today's profile on the fork binary | do fork patches change anything with defaults? |
| a1-ple-ram | + `ple-io ram` | +8 % decode / prefill on fresh text? |
| a2-adapt-slow | + `adapt-swaps 160`, `adapt-decay 0.97` | hit-rate gain? |
| a3-reserve-lean | + reserve 384 MiB, saves expert profile | VRAM headroom vs stability |

Per-patch kill switches (`Start-ForkBasalt.ps1 -Disable name,name`) isolate each patch: see `PATCHES.md`.

## 5. Quality gates (only for changes that alter numerics)

`basalt-bench --quality` (100 chunks) with a Q8_0 reference when available, or the frozen 12-case suite repeated 3x. Acceptance:
no new failed case, KLD change within noise (±0.01).

## 6. Promotion

Only promote an arm if: all correctness checks pass, median decode over 5+ interleaved runs is ≥ 3 % better with non-overlapping
ranges, and the 12-case suite is not worse. Then copy the profile over the daily one. Production files stay untouched until then.
