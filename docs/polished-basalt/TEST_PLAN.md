# Test plan for the 5080 PC (nothing here has been run)

Order matters: stop at the first failure. The fork uses its profile's separate port and refuses an occupied production or
candidate port. Use the current identity-checked launcher and verify actual profile/candidate identity in the receipt.
Do not run during another GPU job or bypass an approval rejection through another entrypoint.

## 0. Preconditions

- GPU idle and no other Codex job running (they share the GPU and RAM; a running job invalidates both measurements).
- Record desktop GPU use and free VRAM without closing unrelated work automatically.
- Record actual driver, CUDA compiler and build identity. Build-specific reports do not establish that every CUDA 13.2 build
  is faulty. Keep the accepted toolchain for the first comparison and test compiler changes separately.
- Check current free disk and temporary build requirements. No implicit weight downloads or file retirement.

## 1. Build (CPU only, ~4–6 min)

1. Copy the repo working tree (without `.git`) to `C:\Users\pc-agent\AgentWorkspace\basalt-5080\src`.
2. Run `fork\windows\Build-Fork.cmd`. CMake build files are under `...\basalt-5080\build`; runtime executables are under the
   source tree's `bin` directory. The build records candidate identity. Neither path replaces `ai200-20261026`.
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

- Freeze control/candidate executable and profile identities and actual expert placement. Greedy identity applies to
  bitwise-neutral changes. Draft widths, reduction order and placement can change rounding; those need numerical-quality
  checks rather than an assumption that verification guarantees identical tokens.
- Your frozen 12-case suite and the 391 arithmetic check on arm a0, then a1.
- Answer-room: one request with `max_tokens 8192` and a hard prompt. The log must show `thinking limited to ... so the answer
  has room` and the reply must contain a final answer.

## 4. A/B arms (profiles in `fork/profiles/`)

Interleave ABAB, 3+ runs per arm, same frozen requests, discard warm-up, record `tg`, `acc`, `hit`, `pcie`, `dl`.

| Arm | Changes vs previous | Question it answers |
|---|---|---|
| a0-daily-control | today's profile on the fork binary | do fork patches change anything with defaults? |
| a1-ple-ram | + `ple-io ram` | Does it reduce completed-task time on fresh text? |
| a2-adapt-slow | + `adapt-swaps 160`, `adapt-decay 0.97` | hit-rate gain? |
| a3-reserve-lean | + reserve 384 MiB, saves expert profile | VRAM headroom vs stability |

Per-patch kill switches (`Start-ForkBasalt.ps1 -Disable name,name`) isolate each patch: see `PATCHES.md`.

## 5. Quality gates (only for changes that alter numerics)

Use the Q8_0 decode-path gate required by CONTRIBUTING.md, with complete coverage and the previous build alongside it.
Large numerical changes need the full 100-chunk run. The frozen 12-case behavioral suite is additional evidence, not a
substitute when the reference is unavailable. Record missing tests as unrun. Use measured baseline variation and the
repository's acceptance criteria, not an invented universal KLD tolerance.

## 6. Promotion

Promote only after relevant correctness/quality checks and repeated matched performance checks pass. Record correct-task
time, first text, prompt processing, output rate, token count and power separately. Preserve rollback and verify restoration.
Include startup and identity-check cost. Source tests alone do not establish a speed gain or justify a daily-profile swap.
