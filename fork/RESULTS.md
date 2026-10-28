# Measured results (RTX 5080 + Ryzen 7 9800X3D, 5600 MT/s RAM, Windows)

What was measured on this machine, how, and what it means for settings. Numbers are from `basalt-probe` and the local test runs, not estimates.

## Engine probe (Qwen3.8-Flash-Next IQ3_S pack, 51.1 GiB, 256k context, int8 KV)
| Measure | Result |
|---|---|
| Decode, cold | 100 tok/s prose, 115 tok/s conversation, 151 tok/s structured |
| Memory bandwidth | RAM 61 GB/s read, PCIe 57 GB/s, GPU 817 GB/s |
| Expert placement | 4,278 experts on the GPU, the rest in RAM |
| Prefill | about 4.4K tok/s; probe suggests `--prefill-cpu-share 0.45` for about +10% |

Reading: decode is limited by CPU and RAM bandwidth (the GPU sits well below its power limit while decoding); prefill is GPU-bound. Two parallel streams do not raise total throughput; they only remove queueing.

## Test loop
`python fork/tools/checks.py quick` now runs pytest in parallel (`pytest-xdist`, added to `requirements-test.txt`) when it is installed: about 43 s down to about 9.7 s locally. Without xdist it runs serially as before.
