# Research backlog and evidence boundaries

These are candidates for isolated experiments, not an approved list to merge or percentages to add. Keep high reasoning,
full routing, the full IQ3_S model and the 262144 context target. Record actual used context separately.

| Candidate | Current disposition | Primary evidence |
|---|---|---|
| Mixed-pack staging | Already present; test the target's CPU/transfer contention before crediting a gain | [Basalt 82](https://github.com/jesdga95/basalt/issues/82) |
| Minimum parked-conversation size / retries | Already incorporated; real continuation, sibling and compaction checks remain | [67](https://github.com/jesdga95/basalt/pull/67), [89](https://github.com/jesdga95/basalt/pull/89) |
| Pinned checkpoint bounce buffer | Small follow-up latency candidate; reported single-digit milliseconds, not faster decoding | [Basalt 66](https://github.com/jesdga95/basalt/pull/66) |
| Incremental large tool-call scanning | Promising CPU-overhead candidate; preserve quoted-tag and strict parsing behavior | [LesterLyu fork](https://github.com/LesterLyu/basalt) |
| CPU sign/dequantization kernels | Measure the actual mixed formats and token widths; a one-token kernel percentage is not a whole-engine percentage | [Strata 3946](https://github.com/Niko1221/Strata/pull/3946), [Basalt 74](https://github.com/jesdga95/basalt/pull/74) |
| Checkpoint pin selection | Reproduce the scenario in this fork before porting the different cache layout | [Strata 3517](https://github.com/Niko1221/Strata/pull/3517) |
| Routed prompt streaming | Author reports faster short follow-up reads on 3090/IQ3_XXS, with contrary decode results; not established here | [Strata 4283](https://github.com/Niko1221/Strata/pull/4283) |
| Two-pass attention selection | Prompt-processing experiment on another GPU; requires real geometry and quality validation | [Strata 4119](https://github.com/Niko1221/Strata/pull/4119) |
| Longer lookup windows | Invasive kernel/window work; reported results concern another GPU and a narrow task | [Strata 4040](https://github.com/Niko1221/Strata/pull/4040) |
| Shared/persistent conversation storage | Memory/restart-latency experiments; check isolation, identity and restore failures | [4224](https://github.com/Niko1221/Strata/pull/4224), [4226](https://github.com/Niko1221/Strata/pull/4226) |
| Worker-message compatibility | Check the actual client/adapter failure; published end-to-end evidence used DeepSeek on Linux | [Strata 4210](https://github.com/Niko1221/Strata/pull/4210) |
| Alternative draft model | Requires extra memory and can displace experts; not consistently faster on 16 GB cards | [Strata 3965](https://github.com/Niko1221/Strata/issues/3965) |
| Gyro-L | Different quantization and quality/speed tradeoff; not an acceleration patch to this IQ3_S setup | [Basalt 13](https://github.com/jesdga95/basalt/pull/13) |

## Fork comparisons

- [Strata-xeno](https://github.com/xenodeve/Strata-xeno): headline results used two GPUs and Swift IQ2_XS. Its coding-shaped
  comparison was roughly tied with tuned upstream output rate. This is not a full IQ3_S single 5080 result.
- [StrataFast](https://github.com/eddoursul/StrataFast): dual-GPU scheduling results do not transfer directly to one GPU.
  Reports include both gains and regressions on different hardware.
- [xyehya's experiments](https://github.com/xyehya/basalt/tree/decode-perf-test): relevant 5080 work, but an Intel host and
  different workloads. Additional PCIe staging became slower after CPU improvements in reported comparisons.
- [capy-flash](https://github.com/ValerioDolci/capy-flash): useful cache/replay and CPU/DMA ideas, with substantial two-GPU
  focus. Private coding tasks, public replay fixtures and quantization comparisons have different evidentiary limits.
- [ninfer-ext](https://github.com/giveen/ninfer-ext): several advertised mechanisms already exist in Basalt. Check exact
  commits and paths before counting them as new work.
- [Infernix](https://github.com/Wallawalla47/Infernix): real-user agent reports are useful, but hardware, checkpoint,
  output limits and quality coverage differ. Do not substitute its results for a matched target comparison.

No combined speed ratio for this fork is established by these projects. In particular, an author benchmark at one power
limit, on another quantization, or with thinking off is not evidence for high-reasoning completed work on this PC.

## Larger source-review ideas

Worker-completion signalling, retained speculative tails, repeated old-context transfers, demand scheduling and 84-SM kernel
layout are profiling hypotheses. First prove the path is active and costly in the normal run. Detailed profiling may itself
change optimized execution. Keep cold/warm, single-request/aggregate, generation/prefill and complete/incomplete outputs
separate. Fix a demonstrated correctness failure before optimizing the affected path.

The current continuation prioritizes unfinished launch validation, prompt/tool fidelity, cancellation, bounded input,
quality-check integrity and adaptive-transfer failure handling. Target CUDA compilation, CPU hardware parity, Q8-reference
quality and repeated full-context coding tests remain distinct acceptance steps.
