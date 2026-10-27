# Why the WSL "Linux" test was slower than native Windows

Short version: the test was **WSL2, not Linux**. WSL2 runs Linux inside a virtual machine whose GPU access goes through a
paravirtual layer. That layer, not Linux itself, costs speed for an engine that moves expert weights between RAM and the GPU
thousands of times per second. A native Linux boot is expected to be equal to or better than Windows, but nobody has measured
it on this PC.

## What was measured (Codex run, 27 Oct 2026, `linux-comparison-20261027`)

Same PC (RTX 5080, Ryzen 9 9800X3D, 128 GB), same saved requests, same weights, same 256K profile, `ple-io ram` on both.

| Setup | Decode | Prefill (6.4K tokens) | Cold start |
|---|---:|---:|---:|
| Windows native, PLE in RAM | 87–101 tok/s (pooled 94.84) | 4,293–4,424 tok/s | 49 s |
| WSL2, PLE in RAM | 77–90 tok/s (pooled 86.75) | 3,865–4,269 tok/s | 4 min 47 s |

Engine counters were the same on both sides, which rules out "different work":

| Counter | Windows | WSL2 |
|---|---:|---:|
| Expert-cache hit rate | 70.8–78.6 % | 72.2–79.0 % |
| MTP draft acceptance | 63–66 % | 58–68 % |
| CPU expert-pool workers | 5 + host | 5 + host |
| Expert RAM copy | 46.3 GiB page-locked | 46.2 GiB page-locked |
| KV streaming | on (32768 resident) | on (99.5 % of reads hit VRAM) |

So the model did the same work and the same speculation; WSL2 simply moved bytes and launched kernels more slowly.

## Evidence for each cause

1. **Host-to-device copies are slower under WSL2.** Upstream Basalt issue #88 benchmarked one RTX 5090 + i7-13700K on both:
   `h2d_gbps` 29.6 (WSL2) vs 40.9 (native Windows) on the same PCIe 5.0 x16 slot, DRAM bandwidth equal (60.7 vs 61.3 GB/s).
   Decode was 126.4 vs 158.7 tok/s. Basalt streams experts and KV over exactly this path.
2. **Page-locking is far slower in the guest.** In the Codex logs, locking the 46 GiB expert copy took **107 s on WSL2 vs 12 s on
   Windows**. Pinned memory in a WSL2 guest has to be registered through the host's graphics kernel, so every pinned range
   costs more and the driver may cap it (the Strata docs say WSL pins only ~1 GB and turns KV streaming off in some paths).
3. **Kernel-launch and sync latency.** A decode window issues hundreds of small launches and waits on the CPU/GPU hand-off
   (`GPU-reach wait`, ~6–11 ms of a ~26 ms window in upstream traces). Under GPU paravirtualization each launch and each
   host-visible event costs extra microseconds. At 85–115 tok/s that adds up to single-digit percent.
4. **Cold start.** The WSL arm read the model from `/mnt/c` (a network-style file bridge, 200–440 MB/s in the log) and built a
   28.8 GB RAM snapshot first. This only affects startup, not decode.
5. **Not the cause:** DRAM bandwidth (equal), expert hit rate (equal), draft acceptance (equal), swap (off), memory pressure
   (no swap, ample headroom).

## What this means

- WSL2 will not beat native Windows for this engine on this PC. Do not use it for speed.
- Native Linux is a different question. It removes causes 1–4 and the Windows pinned-memory cap and EcoQoS issues, and gives
  real hugepages. Upstream's own benchmarks are all native Linux (a 5080 + 9950X3D row reports 186.8 tok/s *aggregate* and
  ~4,900 tok/s prefill, but that is a different CPU, a different benchmark and an aggregate figure).
- The cheap way to know: boot a Linux live USB or a spare NVMe, copy the packs, run the same frozen requests with
  `fork/windows/` equivalents. Expected result: 0–20 % decode, mostly prefill. If it is under 5 %, stay on Windows.

## Caveats

One ordered pass, not alternated A/B; the reasoning text differed across arms; sample sizes are 3 requests. The #88 numbers are
from a different PC. Treat the ranking of causes as informed inference, not a profiled breakdown.
