# Fork switches

Changes carried only by this fork are switched by the environment variable `BASALT_FORK_<NAME>`
(`basalt::platform::fork_switch`): a change is on unless the variable is `0`. `basalt::platform::fork_number` reads a
number the same way. `fork/windows/Start-ForkBasalt.ps1 -Disable a,b` sets the variables to `0`.

| switch | what it does |
| --- | --- |
| `GDN_REC2` | `dec_gdn_rec` / `dec_gdn_rec_seg` run two value heads per 256-thread block (96 blocks for 48 heads, one wave on 84 SMs, where a cluster launch keeps two blocks per SM). Bitwise the same outputs; `dec_gdn_parity` checks it with the switch on and off. Used by windows of 5-8 tokens (the mid chain takes 1-4) |
| `PIN_HEADROOM` | Windows: 1 GiB of page-locked memory (down to 64 MiB) is held across the whole-copy `cudaHostRegister` of the resident expert copy, as the sliced path already holds it, so a copy the driver takes to its ceiling is taken in slices instead of leaving the buffers allocated after it no page-locked room. Bitwise neutral; startup gains one 1 GiB `cudaHostAlloc` |
| `DRAFT_CUM_MULT` | A number in [0.25, 4], default 1: a factor on the draft-length rule's threshold scale (base 0.35 times 1.0-1.5 by the CPU-served bytes per expert). Below 1 proposes longer windows, above 1 shorter ones. Only what is proposed changes; verified output does not. `=0` is out of range, so `-Disable` leaves it at 1 |
| `STAGING` | `0` turns the PCIe-staged share of CPU misses off (single requests and batch windows), so `Start-ForkBasalt.ps1 -Disable staging` has an effect. Tuning without a rebuild: `--exp stage_solo=0`, `stage_pool_gbs=N`, `stage_link_gbs=N`, `stage_fix_us=N`, `stage_blocks=N` (cost model: pool 45 GB/s, link 18 GB/s, 35 us per copy, 8 blocks) |
| `SPLIT_COST` | The draft policy prices lookup windows from lookup rounds and MTP windows from MTP rounds (two cost tables); `=0` is the single shared table. Only what is proposed changes |
| `THRESH_RAMP` | The per-request draft threshold may rise to 1.3 times its base when drafts are accepted less often than the bin table predicts, and drifts back to the base while no draft is offered; `=0` lets it only fall. Window widths change, verified output does not (see [precision](precision/index.md)) |
| `LOOKUP_OVERRIDE` | Opt-in (`=1`): a prompt-lookup window whose first token differs from the MTP's first draft is still offered to the policy, when lookups of that match length were accepted at 0.9 or more and the rate of disagreeing lookups (counted apart) beats the MTP window |
| `MISS_THRESH` | Opt-in (`=1`): the draft threshold is scaled by the trend of the CPU expert misses per layer, per window size against the request's own slow average (square root of the ratio, 0.85 to 1.2). Single-GPU serve loop and the bench loop |
| `COUPLED_DRAFT`, `COUPLED_MIN_TEMP` | Design only, nothing reads them yet: sampled requests would draft with the verifier's own Philox uniform ([coupled-draft.md](coupled-draft.md)). `COUPLED_MIN_TEMP` is a number in [0, 4], default 0.9 |
