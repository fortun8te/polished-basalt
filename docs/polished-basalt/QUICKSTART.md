# Build, test and adjust the fork

The normal model and the experimental fork are separate. Build once per source revision, then reuse the build and frozen
requests for profile comparisons. The launcher refuses occupied production/candidate ports and records the actual inputs.
It does not silently stop the normal model or promote a result.

## Local source checks

Install the repository's test dependencies and pytest in your own environment. These checks use fake engines and local
test servers; no model or PC connection is required.

```sh
python fork/tools/checks.py quick
python fork/tools/checks.py full
```

Quick checks cover the changed admission, text-boundary, image-boundary, process-pipe, launcher, identity and quality-checker
paths. Run them after a focused edit. Full checks cover the server suite too; use them before publishing an integrated change.
CUDA compilation, target CPU instructions, model quality and speed require separate checks. A skipped test is not a pass.

## Prepare on the Windows PC

Use the existing PC model files and accepted CUDA/toolchain installation. Review the paths in `fork/profiles/`; no model
download is needed. The supplied build wrapper targets the existing task account's toolchain and working directories.

```text
git clone --branch polished-basalt https://github.com/fortun8te/polished-basalt C:\Users\pc-agent\AgentWorkspace\basalt-5080\src
cd C:\Users\pc-agent\AgentWorkspace\basalt-5080\src
fork\windows\Build-Fork.cmd
```

If transferring a source archive instead of cloning, first run `python fork/tools/fork_identity.py stamp` in the Git checkout
and include the resulting stamp. The build records executable, source, profile and model identities. Full model hashing can
take noticeable time and is part of preparation; do not describe it as a measured instant-start operation.

## Start an arm during an agreed test window

Finish existing work and stop the daily model through its normal control. Do not use a different port to load a second full
model onto the same GPU. The fork's startup checks are an additional refusal mechanism, not permission to interrupt work.

```text
python fork/tools/fork_launch.py start --arm a0-daily-control --wait 240
python fork/tools/fork_launch.py status
```

The default root is the checkout. Each run records resolved configuration, process identity, logs and its candidate identity
under ignored local run directories. Use `a1-ple-ram`, `a2-adapt-slow` or `a3-reserve-lean` only as explicitly selected trials.
Their names do not mean they are faster. Parameter changes belong in a new recorded candidate, never an unrecorded inherited
environment. The launcher builds a child-specific environment so switches do not leak between arms.

Run the existing engine benchmark and the separate completed-task/quality checks in [TEST_PLAN.md](TEST_PLAN.md). Freeze
the request bodies, output allowances and acceptance criteria before comparison. Record actual input context, high reasoning,
power limit, cold/warm state and complete outputs. Alternate control/candidate repeats. Do not use synthetic throughput alone
to judge a coding assistant or silently lower reasoning/context to finish within a time budget.

## Stop, restore and decide

```text
python fork/tools/fork_launch.py stop
python fork/tools/fork_launch.py promote-plan --arm a1-ple-ram --daily PATH_TO_DAILY_PROFILE
```

Stop checks the recorded process identity and reports failure if it cannot confirm termination. Restart the accepted daily
model through its existing starter after the experimental process has ended. `promote-plan` prints differences; it does not
overwrite the daily profile. `fork_identity.py show` displays the prior candidate, and `restore --to ID` restores a recorded
fork candidate when appropriate. Neither command establishes that real hardware recovery has been tested.

Do not copy a candidate into the daily setup until its quality and repeatable performance checks pass. Keep the working
profile and restoration record. Real Windows launch, GPU execution and cold wake/reboot still require target validation.
