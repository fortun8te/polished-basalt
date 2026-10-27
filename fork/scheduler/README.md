# Request scheduler

A small proxy that sits in front of one `basalt serve` and decides which request the server sees next. The server
runs one decode slot, and re-reading a long prompt costs far more than reusing a cached one, so the order matters: an
interactive turn should never queue behind a title generator, and a conversation should follow its own previous turn.
Standard library only (Python 3.9 or newer); one file, `scheduler_proxy.py`.

```
clients ──> scheduler :8081 ──(at most max_inflight)──> basalt serve :8080
```

## Run

```
python scheduler_proxy.py --config scheduler.example.json
python scheduler_proxy.py --upstream http://127.0.0.1:8080 --listen 127.0.0.1:8081 --max-inflight 1
.\Start-Scheduler.ps1            # Windows; -Stop stops only the scheduler it started
```

Status: `GET /_sched/status`. Log: one `key=value` line per event on stderr (and `--log-file`).

## Order of service

1. A background request that has waited `queue.max_wait_s` goes first (oldest first), so nothing starves.
2. Then interactive requests, then background ones.
3. Inside a class the oldest goes first, unless one of the first `queue.reorder_window` entries continues the last
   interactive request the server ran (see Prefix). The oldest can be passed `queue.max_skips` times, then it goes.

Nobody is refused: there is no 429 and no queue limit. Responses carry `X-Queue-Position` (0 = sent to the server at
once, else the 1-based place in line when the request arrived), `X-Queue-Class`, `X-Queue-Id` and `X-Queue-Wait-Ms`.
A streaming request that waits longer than `queue.keepalive_after_s` gets its `200` and `text/event-stream` headers
early, then `: queued position=N` comments every `keepalive_every_s`, so client idle timers do not fire. If the server
then answers with an error, it arrives as an error event in the client's API format.

## Classes

`classify.rules` run in order; the first match sets the class, `classify.default` otherwise. All conditions of a rule
must hold: `header` (name to regex), `model`, `path`, `system_regex` (first 4 KiB of the system prompt), `max_tokens_lte`
and `max_tokens_gte` (the request's output cap, whichever of `max_tokens`, `max_completion_tokens`,
`max_output_tokens` it sends), `tools` (true when it lists tools), `stream`. Regexes ignore case. Without
`classify.rules` the built-in rules apply: `X-Priority: background`, a model name containing `haiku`, and an output cap
of 512 or less with no tools. Only the paths in `queued_paths` are scheduled; everything else (`/v1/models`,
`/health`, `count_tokens`, `/load`) passes straight through.

## Prefix

For each request the proxy hashes the first `prefix.bytes` (8192) of system, tools and the first `prefix.messages`
message in prompt order (`prefix`), and of that first message alone (`conversation`). `cache_control` markers are left
out. Only the start of the prompt is read, so both hashes are the same on every turn of a conversation. A queued request
scores 2 when both match the last interactive request the server ran, 1 when only the prefix does (same agent, other
session), else 0; the best score inside the window goes first. Background requests never change what counts as
resident. An agent's tool list is usually longer than 8 KiB, so the prefix hash names the agent and the conversation
hash tells its sessions apart; sessions that open with the same first message cannot be told apart.

## Thinking policy (off by default)

`policy.enabled` turns on `policy.rules`. A rule has `class` and the same conditions as above, then any of:
`set_default` (fields added only when the client sent none; objects such as `chat_template_kwargs` are merged),
`force` (fields set always) and `max_tokens_cap` (lowers the request's output cap, or adds one, in the field its API
uses: `max_tokens`, `max_completion_tokens` or `max_output_tokens`).

The body is never re-serialised. A field the client already sent is replaced in place and a new one is appended after the
last top-level member, so every other byte, the messages included, reaches the server as sent.

Which fields are safe depends on the server. Basalt closes the thinking at `reasoning_budget_tokens` while decoding,
so that field leaves the prompt alone. `chat_template_kwargs.enable_thinking=false` and `reasoning_effort` are rendered by
the chat template into the system block at the head of the prompt, so they change the prompt (and so the cached prefix)
of the request that carries them. Use them only for classes that share no prefix with the resident conversation, and
never for interactive requests.

## Cancellation

The proxy watches each client socket. A client that leaves while queued is dropped from the queue. One that leaves
while running has its upstream connection closed at once, which frees the proxy's slot immediately and makes the server
stop within its own hang-up poll. A half-closed client socket counts as gone, as it does for the server.

## Cloud fallback (off by default)

With `fallback.enabled`, a request of a class in `fallback.classes` (background) that is still queued after
`fallback.after_s` is sent to `fallback.base_url` instead and no longer waits for the slot. The client's `Authorization`
and `x-api-key` are not forwarded unless `forward_client_auth` is true; `fallback.headers` (values may use
`${ENV_NAME}`) are added, and `fallback.force` fields (usually `model`) are spliced into the body. The response carries
`X-Queue-Fallback: 1`.

## Pointing clients at it

Use the scheduler's address wherever the server's was. Settings below are the clients' documented ones; none were run
against the real clients here.

- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8081`, `ANTHROPIC_AUTH_TOKEN` as before. The server accepts
  its alias and the model file's name. Point `ANTHROPIC_MODEL` at the alias and `ANTHROPIC_DEFAULT_HAIKU_MODEL`
  (the helper slot) at the file name; the `model-file-name` rule then files the helper calls as background.
- **Codex CLI**: in `config.toml` set a provider with `base_url = "http://127.0.0.1:8081/v1"`, `wire_api = "responses"`
  and a large `stream_idle_timeout_ms`. A second provider for batch work can add
  `http_headers = { "X-Priority" = "background" }`.
- **OpenCode**: set the provider's `options.baseURL` to `http://127.0.0.1:8081/v1`, and `small_model` to a model entry
  that uses the file name, or add `options.headers` with `X-Priority`.
- **Custom client**: send `X-Priority: background` on calls that can wait.

## With the server's own queue and cache

With the server's default single stream it serialises on a lock in no promised order; with the scheduler in front,
at most `max_inflight` requests reach it, so that lock is never contended. Match `max_inflight` to `--parallel`
if you change that. Set the server's `conversation-cache-min-tokens` above the size of a helper call so short
requests do not park over a long conversation; ordering cannot prevent that by itself.

## Limits

- A background call that runs between two turns of a conversation can still evict it from the server's caches;
  the proxy only orders requests and has no hold-back window.
- Non-streaming requests get nothing until their turn. Clients whose idle timer counts events and ignores SSE
  comments can still time out in a long queue.
- No upstream idle timeout: a server that stops answering without closing holds its slot.
- HTTP/1.1 on plain TCP for clients, one request per connection (`Connection: close`). Bodies with a
  `Content-Encoding` are queued as interactive and never inspected or edited.
- Position is the estimate at arrival; prefix reordering can change the real order.

## Tests

```
pip install -r requirements.txt
python -m pytest
```

The tests start the proxy and a fake slow streaming upstream on loopback ports; no model, GPU or network is used.
