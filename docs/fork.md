# Fork switches

Changes carried only by this fork are switched by the environment variable `BASALT_FORK_<NAME>`
(`basalt::platform::fork_switch`): a change is on unless the variable is `0`. `basalt::platform::fork_number` reads a
number the same way. `fork/windows/Start-ForkBasalt.ps1 -Disable a,b` sets the variables to `0`.

| switch | what it does |
| --- | --- |
| `GDN_REC2` | `dec_gdn_rec` / `dec_gdn_rec_seg` run two value heads per 256-thread block (96 blocks for 48 heads, one wave on 84 SMs, where a cluster launch keeps two blocks per SM). Bitwise the same outputs; `dec_gdn_parity` checks it with the switch on and off. Used by windows of 5-8 tokens (the mid chain takes 1-4) |
