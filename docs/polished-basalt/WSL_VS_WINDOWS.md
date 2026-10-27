# Windows and WSL comparison

WSL2 runs Linux inside Windows. The recorded one-pass comparison was slower under WSL, so native Windows remains the
accepted daily setup. That result does not establish a general Windows/Linux ranking or prove the cause of slower decode.

## Recorded observations

The same RTX 5080, Ryzen 7 9800X3D, full IQ3_S model, saved requests and configured 262144 window were used. Each of three coding
requests stopped at 1024 output tokens before producing a final answer. Their reasoning and speculation differed.

| Setup | Pooled output tokens/s | Cold start |
|---|---:|---:|
| Native Windows, PLE in RAM | 94.84 | 48.7s |
| WSL2, PLE in RAM | 86.75 | 287s |

Windows recorded roughly 46.3 GiB pinned expert memory; WSL recorded 46.2 GiB. KV streaming was active in both. A generic claim
that WSL could pin only 1 GiB does not explain this run. Similar hit-rate or acceptance ranges do not prove equal computational
work. There was one ordered pass, not alternating repeats or completed-task quality acceptance.

WSL startup included creating and checking a 28.8 GB memory snapshot through the Windows filesystem. This extra work confounds
the cold-start comparison. It does not by itself establish why warm generation was slower.

## Hypotheses, not established causes

GPU virtualization, copy throughput, launch/synchronization overhead, CPU scheduling and memory contention are plausible.
Reports from other machines are not a profile of this one. Both systems pinned the observed allocation. No inference-time
clock or transfer trace isolated the local cause.

Investigate with identical completed requests and alternating runs. Record exact builds, sampling, output lengths, actual
context, CPU time, transfer bytes, GPU clocks/power and concurrent work. Separate cold loading, snapshot creation, prompt
processing and generation. Storage placement is a separate experiment requiring sufficient disk headroom.

Native Linux has not been measured on this PC. It is a separate comparison, not an assumed upgrade or prerequisite. Do not
change the user's OS or daily engine on the strength of unmatched community numbers.

Sources: [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
[WSL filesystems](https://learn.microsoft.com/en-us/windows/wsl/filesystems),
[Basalt Windows/WSL reports](https://github.com/jesdga95/basalt/issues/88).
