# Performance evidence and remaining uncertainty

Polished Basalt has no measured combined speed advantage on the target RTX 5080. Host tests establish specific code behavior,
not CUDA compilation, model quality or an end-to-end improvement. Compare stock Basalt, stock Strata and the fork with the
same full IQ3_S model, high reasoning, actual prompt lengths, output allowances and GPU power limit.

## Recorded daily-setup results are not fork results

| Workload | Recorded result | Limit |
|---|---|---|
| Short coding settings screen | 115.1 output tokens/s | Thinking off; short input |
| Three tiny completed high-reasoning tasks | 70.4–75.1 output tokens/s | Instrumented small tasks; 256K configured only |
| Windows RAM comparison | 94.84 output tokens/s | Capped coding responses did not finish |
| Near-filled context | 104.9 output tokens/s at 255,419 input tokens | Separate workload; concurrent NAS copy |
| Return to a long chat | 18.68s to 0.166s first-text wait | One A/B/A comparison; adopted before this fork |

These observations cannot form one progress curve. The 115.1 result is not the high-reasoning baseline. No recorded result
establishes 140+ output tokens/s with high reasoning, a filled 256K context and checked coding quality.

## Candidate benefits

| Candidate | Evidence | What remains to prove |
|---|---|---|
| Mixed-pack CPU-to-GPU staging | Issue 82 reports about 14% with some weights on disk, about 2.4% with all weights in RAM | Target contention, rounding and complete-task time |
| RAM-backed PLE | One local comparison favored RAM; another local screen was slower | Alternating repeats on fresh prompts and loading/memory cost |
| CPU sign-mask work | Scalar checks and test infrastructure | Actual AVX-512 parity and whole-model benefit |
| Retry/cache admission | Specific repeated-chat examples and source tests | Real multi-turn, sibling, cancellation and compaction behavior |
| Draft policy | Host tests; some features are experimental | Actual acceptance, numerical quality and completed-task time |
| GPU kernel proposals | Author measurements on particular builds/machines | Target compilation, parity and whole-window timing |

Benefits overlap and can conflict. A separate 5080 fork's faster CPU path made additional staging slower in its reported
comparison. Do not sum percentages from separate patches or hardware. A large pinning recovery on a 64 GB machine does not
transfer to the recorded 128 GB run, whose entire expert allocation was already pinned.

Sources: [staging](https://github.com/jesdga95/basalt/issues/82),
[GPU follow-up](https://github.com/jesdga95/basalt/issues/85),
[CPU/GPU follow-up](https://github.com/jesdga95/basalt/issues/86),
[checkpoint routing and withdrawn decode claim](https://github.com/jesdga95/basalt/pull/89),
[separate 5080 experiments](https://github.com/xyehya/basalt/tree/decode-perf-test).

## Useful acceptance

Measure time to correct completed work first. Record prompt processing, output rate, generated tokens, retries, cache reuse
and waiting separately. Repeat against an unchanged control, including difficult high-reasoning work. A different model,
lower precision, skipped experts or reduced reasoning changes the comparison rather than proving this model is optimized.

Numerical changes require the Q8 reference quality gate in CONTRIBUTING.md. A small behavioral suite supplements that gate;
it cannot replace it. Bitwise-neutral changes need actual greedy identity with fixed placement. Keep the daily setup intact
until target checks pass. Neither 140 nor 200 tokens/s can be promised or ruled out from the available evidence. There is no
defensible combined percentage forecast yet.
