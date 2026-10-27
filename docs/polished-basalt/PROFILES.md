# Launch profiles

All profiles are in `fork/profiles/`. The `a*` arms are the A/B ladder for the 5080 PC (see `TEST_PLAN.md`); `b0` is a
generic starting point supplied from another review. Paths in the `a*` files are placeholders for the Windows machine; edit the
`model` and `ple-gguf` paths before use.

| File | Port | Context | Slots | What it is for |
|---|---:|---:|---:|---|
| `a0-daily-control.json` | 18892 | 262,144 | 1 | Today's daily profile on the fork binary: the control arm |
| `a1-ple-ram.json` | 18893 | 262,144 | 1 | a0 + `ple-io ram` (n-gram table locked in RAM) |
| `a2-adapt-slow.json` | 18894 | 262,144 | 1 | a1 + `adapt-swaps 160`, `adapt-decay 0.97`, `adapt-every 1` |
| `a3-reserve-lean.json` | 18895 | 262,144 | 1 | a2 + VRAM reserve 384 MiB and a saved expert profile |
| `b0-agents-parallel2-128k.json` | 8080 | 131,072 | **2** | Generic agent profile from a separate review (see below) |

## b0: what to know before using it

- **Two slots.** On this PC two or four slots gave *lower* aggregate throughput than one (about 67–69 vs 76 tok/s in the Codex
  concurrency runs), and slots cost expert-cache VRAM on a 16 GB card. Use `"parallel": 1` for a single agent; use 2 only if two
  clients must make progress at the same time and you accept the lower speed. The review's own README says the same: try
  1 vs 2 vs 4 with your real load.
- **131,072 context** is half of the 256K target in the daily setup. Raise `max-context` to 262144 and keep `kv-resident` set
  (the daily profile uses 32768) if you need the long window.
- **`vision: off`** closes the image-fetch exposure even without the image-source patches.
- **`expert-profile-save`** records which experts your work uses. Basalt's converter can bake that profile into a pack
  (`tools/convert/convert.py --profile <file>`), so the hot experts start in VRAM. That writes a whole new pack (tens of GB),
  so it only makes sense after a week of real use and when the disk has room; the daily compact pack is 58 GB.
- Start: `bin/basalt serve --config <profile> --api-key <key>`. Always give a key if `host` is not 127.0.0.1.
- Then tune on the machine: `basalt-probe` and apply its `suggested:` lines; `basalt-bench --community` before and after; keep
  only wins.
