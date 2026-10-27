# Polished Basalt

<p align="center"><img src="assets/polished-basalt/crafting-recipe.png" alt="Four Minecraft Basalt blocks craft into four Polished Basalt blocks" width="640"></p>

A personal, experimental [Basalt](https://github.com/jesdga95/basalt) fork for an RTX 5080, Ryzen 7 9800X3D and 128 GB RAM.
Basalt descends from [Strata](https://github.com/Niko1221/Strata). Both upstream projects and their contributors retain their
credits and licences.

The target is full Qwen3.8-Flash-Next IQ3_S, high reasoning and a 262,144-token window on native Windows. The fork combines
selected engine experiments with tested improvements to request handling, cancellation, input boundaries and test integrity.

**Status:** local host tests do not establish a Windows/CUDA build, target model quality or a measured speedup. The complete
fork has not demonstrated 140 tokens/s under those conditions. Existing upstream benchmark numbers are not this fork's
results. Keep the accepted daily setup until the target validation passes.

## Start here

- [Build, test and adjust on the PC](docs/polished-basalt/QUICKSTART.md)
- [Ordered target validation](docs/polished-basalt/TEST_PLAN.md)
- [What changed, provenance and switches](docs/polished-basalt/PATCHES.md)
- [Evidence and limits](docs/polished-basalt/EXPECTED_GAINS.md)
- [Windows versus WSL](docs/polished-basalt/WSL_VS_WINDOWS.md)
- [Codex integration](docs/polished-basalt/CODEX_INTEGRATION.md)
- [Upstream documentation](UPSTREAM_README.md)

## Fast local checks

With the repository's test dependencies and pytest installed:

```sh
python fork/tools/checks.py quick
python fork/tools/checks.py full
```

These use local fake engines, fixtures and test servers. They do not contact the PC or load a model. The GPU and numerical
quality gates are separate. GitHub CI runs on pushes, pull requests and manual dispatch; it compiles Linux and Windows CUDA
code and runs host tests without a GPU. Do not interpret a green CI result as a model-speed benchmark.

## Experiment controls

The recovered launcher verifies source, binary, profile, model and process identity. Each profile has an isolated endpoint
and run directory. Per-arm switches live only in the child environment. Startup refuses an occupied production/candidate
port. Stop verifies the recorded process and reports failure rather than claiming an unsuccessful shutdown worked.

Candidate manifests and rollback archives remain local and ignored by Git. A promotion plan prints differences; it does not
replace the daily profile. Full model hashes can take time to verify. Build once per source change and reuse the frozen
workloads and receipts for matched profile comparisons.

No weights, credentials, private transcripts or benchmark claims from an unrun experiment are distributed here. See
[CONTRIBUTING.md](CONTRIBUTING.md), [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
