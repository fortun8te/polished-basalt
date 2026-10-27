# Making Codex (and Claude Code / OpenCode) use the local model efficiently

Evidence from three sources: your Codex/qodex measurements, the upstream Basalt/Strata issue tracker, and public write-ups.
Effect sizes marked CLAIMED come from a single user or a vendor and were not reproduced.

## Where the time goes today (your qodex measurements)

- A 3-task suite took 72.7 s, 63.9 s after the guidance changes you adopted; the cloud model Luna took 24.5 s, Sol 70.8 s.
- Local decodes faster per token than Sol but writes about twice the tokens (4,847 vs 2,189). Generation dominates; tool
  execution is ~0.1 s per task.
- 79–88 % of input was served from cache; uncached prefill (~7.7K tokens) cost ~3 s.
- A fresh 255K-token read costs ~72 s; a returning conversation costs ~0.2 s with the RAM conversation cache.
- Thinking off failed the hard tasks (0/4); thinking on passed 4/4 but those answers took 15–146 s.

So the levers are: (1) fewer tokens generated, (2) never losing the prefix cache, (3) not letting side calls block the slot.

## Ranked actions

| # | Action | Evidence | Effort |
|---|---|---|---|
| 1 | Keep the answer reachable: `answer-room` is now in the fork (thinking is closed early so `max_tokens` is never eaten). Set a server default `reasoning-budget-tokens` only well below the client's `max_tokens`. | basalt #95: 4/4 empty answers at 8,192 without it; your own run had 2 of 4 hit the cap | done in fork |
| 2 | Find what busts the prefix cache: compare `pp=<prompt>/<cached>c` in the server `done` lines across a day of real turns; the first turn whose cached count drops names the buster. | Basalt already pins Claude Code's `cch=` stamp (`serve/frontend.py`); Codex has other candidates (environment block, MCP/tool order, reasoning-item re-rendering) | S |
| 3 | Claude Code: set `CLAUDE_CODE_ATTRIBUTION_HEADER=0` in `settings.json` env. | CLAIMED (Unsloth docs: 2 s → 30 s slowdown without it) | S |
| 4 | Send `preserve_thinking: true` where the chat template supports it, so earlier turns' thinking is re-rendered identically. | Qwen3.6 model card text (VERIFIED), Qwen3.8 assumed | S |
| 5 | Avoid compaction as long as possible; when it must happen, do it once. Codex: raise `model_auto_compact_token_limit` (you use 220000) and set `model_auto_compact_token_limit_scope = "body_after_prefix"`. OpenCode: `"compaction": {"prune": false}`. | Each compaction forces a full re-read (~29 s per 100K). TraceLab: 18 % of Codex sessions compacted (secondary source) | S |
| 6 | Put `fork/scheduler/` in front of the single slot: interactive turns first, helper calls (haiku-class, titles, summaries) behind, prefix-aware ordering, cancellation. Point `ANTHROPIC_DEFAULT_HAIKU_MODEL` (Claude Code) or `small_model` (OpenCode) at the local model name so those calls are recognised. Pair it with `conversation-cache-min-tokens 4096` so a helper call can never evict your long conversation. | Basalt's queue is FIFO; multi-slot batching measured **lower** aggregate throughput (69 vs 76 tok/s) | M, built |
| 7 | Codex: `[agents] max_concurrent_threads_per_session = 1` so subagents queue instead of fighting for the slot. | Codex subagent docs (VERIFIED) | S |
| 8 | Tool output discipline: `tool_output_token_limit` around 8000 (you use 10000). Shell-output compressors such as RTK are CLAIMED to cut 20–90 % of tool tokens; each 10K tokens of tool output costs ~3 s of prefill. | CLAIMED | S |
| 9 | Per-step reasoning effort through the scheduler: background class `reasoning_budget_tokens` / `max_tokens` cap; leave interactive untouched. Medium effort on the interactive path was rejected in your trial (one code-understanding error). Low effort has never been measured on your suite: do that first. | Basalt serve: only budget and `max_tokens` leave the cached prefix intact; `enable_thinking`/`reasoning_effort` change the system block | M |
| 10 | Make the Responses endpoint tolerant of non-function tools (`web_search`, `image_generation`, `namespace`) and set a `model_catalog_json` entry so Codex stops falling back to generic metadata. | llama.cpp PR #23041 did this for the same 400 errors | M, check Basalt first |
| 11 | Local/cloud cascade: local for bulk edits, cloud for planning/review, routed by model name through the proxy. Subagents can set `model` but not `model_provider`. | aider architect/editor split (VERIFIED for aider, no local-Qwen numbers) | M |
| 12 | Wait-token logit bias to shorten chains of thought (27–51 % on R1-style models). | CLAIMED, untested on Qwen agent tasks | S, risky |

## Codex provider block (based on your `pc-qwen.config.toml`)

```toml
model_provider = "private_pc"
model = "qwen3.8-flash-next-iq3_s"
model_context_window = 262144
model_auto_compact_token_limit = 220000
model_auto_compact_token_limit_scope = "body_after_prefix"   # verify your Codex build accepts it
tool_output_token_limit = 8000
model_reasoning_effort = "high"
model_reasoning_summary = "none"

[model_providers.private_pc]
name = "private_pc"
base_url = "http://127.0.0.1:8081/v1"        # the scheduler; 18894 if you skip it
wire_api = "responses"
stream_idle_timeout_ms = 1800000            # long queues
requires_openai_auth = false

[agents]
max_concurrent_threads_per_session = 1
```

## Do not bother

- Rewriting Codex's base prompt (~8.5K tokens once, then cached; the docs discourage `model_instructions_file`).
- Multi-slot batching for this MoE.
- `wire_api = "chat"` (removed from Codex).
- LSP/semantic-search tooling: no evidence of a gain on local models.
- Fine-tunes that cut thinking (Swift 1.5, ThinkingCap): sources conflict. One review found Swift 1.5 GSQ IQ3_S for Flash-Next
  (same size as your pack, thinking tokens −10 to −45 % in small independent tests, IFBench −3 points, no 256K test); another
  found only a 27B dense variant. Verify the Flash-Next card before investing 84 GB of disk; it needs `convert.py` to become a
  `.basalt` pack. It is the strongest *time-to-answer* lever if it holds.

## Not yet measured in your suite

Low reasoning effort; an output cap for routine turns; interactive (not `exec`) timing; compaction and the tool-output cap
(never triggered); long multi-file tasks at 100K+ context; cold start and sleep-to-wake; a matched multi-agent run.
