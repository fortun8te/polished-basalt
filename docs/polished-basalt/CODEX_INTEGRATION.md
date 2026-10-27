# Codex and other coding clients

Optimize time to correct completed work while preserving the requested reasoning, context and tools. A short answer or a
successful process exit is not proof the task finished. Short screens and larger real tasks must be reported separately.

## Established local observations

One repair trial fell from 72.46 to 56.94 seconds after execution guidance, with 120 checks passing. A separate three-task trial
fell from 72.66 to 63.86 seconds after shorter final reports. These were single comparisons, not general speed guarantees.
Returning to one long document improved first-text latency from 18.68 to 0.166 seconds with retrieval retained. That cache gain
was already adopted before this fork and is not evidence of compaction correctness or faster decoding.

High reasoning passed the small reasoning comparison more reliably than medium. Retain high; do not shorten reasoning,
shrink the context, omit experts or change models merely to reach a speed number. Fix unnecessary tool calls, failures and
repeated prompt processing when traces actually show them.

## Provider and conversation contract

The adapter maps Codex high to Qwen xhigh. Its Responses endpoint is stateless: clients must replay prior reasoning, tool
calls and paired results. Verify this through continuation, retry, resume and compaction, using synthetic private canaries.
Do not assume a one-turn reply establishes the whole contract.

The adapter's reasoning replay field uses reversible encoding. Keep captured request fixtures local and synthetic; do not
publish private conversations as supposedly encrypted diagnostics. Incomplete responses and incomplete tool arguments must
not be treated as completed work.

The context window includes input and output. Preserve enough output space for reasoning plus the final answer. A fixed
small output cap can stop the model before code or an answer appears. The answer-room policy is a behavior change that needs
quality checks; it is not proof that shorter thinking preserves difficult-task performance.

## Keep provider registration separate from selection

A user-level provider definition can register a connection without selecting it globally. For the existing Mac forwarding
setup, a registration-only example is:

```toml
[model_providers.private_pc]
name = "Private PC Qwen"
base_url = "http://127.0.0.1:18894/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
stream_idle_timeout_ms = 600000
```

Select the local provider/model with 262144 context, high effort and its own model metadata in the local thread/profile.
Do not overwrite the global OpenAI catalog or context to make one local chat work. A provider table alone does not prove
that a desktop model picker can route across providers; verify the actual start/resume route in the installed client.
A client on the PC can use its local 18891 endpoint; fork trials use the profile's separate port. None should be public.

## Changes that require evidence

- Initial skill catalog limits can omit useful skills. Inspect actual discovery and test skill-using tasks before trimming.
  Keep relevant project instructions and useful memory; casual greetings need not recap unrelated projects.
- Tool-output limits can remove error evidence and cause rereads. Measure truncation before changing them. Shorter output
  is not automatically less total work.
- Local compaction around 220000 total tokens is the recorded client setup. Changing thresholds or counting scope requires
  a real recall/cache test. Delaying compaction indiscriminately can overflow the model's actual window.
- WebSockets, remote compaction and hosted search are compatibility features, not free speed switches. Ordinary client-side
  function/MCP tools have a different contract. Do not advertise a tool the adapter cannot execute or disable all tools as
  a substitute for fixing schema errors.
- The scheduler is experimental. Mock-server tests do not establish Windows/model behavior, cancellation recovery or a
  performance benefit. Measure it before placing it in the daily path. Keep background limits separate from coding tasks.
- A repeated-call warning needs a known failing trace and legitimate retries/polling controls. Do not cancel useful work
  based on a guessed repetition threshold.

## Comparison checklist

Freeze the actual client binary, profile, model and server build. Confirm that smoke tests use the same effective settings
as the normal launcher, apart from explicit disposable-fixture exceptions. Record total task time, first correct state,
final completion, reasoning/output tokens, tool retries, prompt reuse and actual context. Repeat matched runs and require
correct artifacts plus the relevant unchanged tests. See [TEST_PLAN.md](TEST_PLAN.md).

Official references: [configuration](https://developers.openai.com/codex/config-reference),
[skills](https://developers.openai.com/codex/skills),
[Qwen model guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
