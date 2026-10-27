#!/usr/bin/env python3
"""Request scheduler in front of a single-slot, OpenAI- and Anthropic-compatible server.

    python scheduler_proxy.py --config scheduler.example.json
    python scheduler_proxy.py --upstream http://127.0.0.1:8080 --listen 127.0.0.1:8081

Clients talk to the proxy instead of the server. The proxy admits at most `max_inflight` inference requests to the
upstream and queues the rest without refusing any: interactive turns go first, background calls (title and summary
helpers, small-model calls) wait, and among equals the request that continues the conversation the upstream last served
goes first so its cached prefix is reused. A client that leaves is dropped from the queue, or its upstream connection is
closed, which frees the slot. Request bodies reach the upstream byte for byte; the optional policy only splices the
fields it names. Standard library only.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import ipaddress
import itertools
import json
import os
import re
import signal
import ssl
import sys
import time
from datetime import datetime
from urllib.parse import urlsplit

INTERACTIVE, BACKGROUND = "interactive", "background"
MISSING = object()
STATUS_PATH = "/_sched/status"

# ------------------------------------------------------------------------------------------------------ configuration

CONDITIONS = ("path", "model", "header", "max_tokens_lte", "max_tokens_gte", "tools", "system_regex", "stream")

DEFAULT_RULES = [
    {"name": "x-priority", "header": {"x-priority": "^background$"}, "class": BACKGROUND},
    {"name": "small-model", "model": "haiku", "class": BACKGROUND},
    {"name": "tiny-untooled", "max_tokens_lte": 512, "tools": False, "class": BACKGROUND},
]

DEFAULTS = {
    "listen": "127.0.0.1:8081",
    "upstream": "http://127.0.0.1:8080",
    "max_inflight": 1,
    "max_body_mib": 256,
    "read_timeout_s": 120,
    "connect_timeout_s": 10,
    "allowed_hosts": [],
    "queued_paths": ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/completions"],
    "queue": {"max_wait_s": 120, "reorder_window": 4, "max_skips": 4, "keepalive_after_s": 15, "keepalive_every_s": 10},
    "prefix": {"bytes": 8192, "messages": 1},
    "classify": {"default": INTERACTIVE, "rules": DEFAULT_RULES},
    "policy": {"enabled": False, "rules": []},
    "fallback": {"enabled": False, "base_url": "", "after_s": 30, "classes": [BACKGROUND], "headers": {},
                 "force": {}, "forward_client_auth": False},
    "log": {"file": None},
}


class ConfigError(ValueError):
    pass


def merge(defaults: dict, given: dict, where: str = "") -> dict:
    """`defaults` overlaid with `given`; a name the defaults do not have is an error (typos must not pass silently)."""
    if not isinstance(given, dict):
        raise ConfigError(f"{where.rstrip('.') or 'config'}: expected an object")
    out = dict(defaults)
    for key, value in given.items():
        if key not in defaults:
            raise ConfigError(f"unknown setting {where}{key}")
        base = defaults[key]
        out[key] = merge(base, value, f"{where}{key}.") if isinstance(base, dict) and base else value
    return out


class Rule:
    """One classification or policy rule: every condition it names must hold."""

    def __init__(self, raw: dict, kind: str):
        if not isinstance(raw, dict):
            raise ConfigError(f"{kind} rule: expected an object")
        extra = set(raw) - set(CONDITIONS) - {"name", "class"}
        if kind == "policy":
            extra -= {"set_default", "force", "max_tokens_cap"}
        if extra:
            raise ConfigError(f"{kind} rule {raw.get('name', '?')}: unknown setting {sorted(extra)[0]}")
        cls = raw.get("class")
        if cls not in (INTERACTIVE, BACKGROUND, None) or (kind == "classify" and cls is None):
            raise ConfigError(f"{kind} rule {raw.get('name', '?')}: class must be interactive or background")
        if not any(c in raw for c in CONDITIONS) and kind == "classify":
            raise ConfigError(f"classify rule {raw.get('name', '?')}: needs at least one condition")
        self.raw, self.name, self.cls = raw, raw.get("name") or f"{kind}-rule", cls
        try:
            self.path = re.compile(raw["path"], re.I) if "path" in raw else None
            self.model = re.compile(raw["model"], re.I) if "model" in raw else None
            self.system = re.compile(raw["system_regex"], re.I) if "system_regex" in raw else None
            self.headers = {k.lower(): re.compile(v, re.I) for k, v in (raw.get("header") or {}).items()}
        except re.error as e:
            raise ConfigError(f"{kind} rule {self.name}: bad pattern: {e}") from None
        self.lte, self.gte = raw.get("max_tokens_lte"), raw.get("max_tokens_gte")
        self.tools, self.stream = raw.get("tools"), raw.get("stream")
        self.set_default, self.force = raw.get("set_default") or {}, raw.get("force") or {}
        self.cap = raw.get("max_tokens_cap")
        if self.cap is not None and (isinstance(self.cap, bool) or not isinstance(self.cap, int) or self.cap < 1):
            raise ConfigError(f"policy rule {self.name}: max_tokens_cap must be a positive whole number")

    def matches(self, info: "RequestInfo") -> bool:
        if self.path and not self.path.search(info.path):
            return False
        if self.model and not (info.model and self.model.search(info.model)):
            return False
        for name, rx in self.headers.items():
            if not rx.search(info.headers.get(name, "")):
                return False
        if self.lte is not None and not (info.max_tokens is not None and info.max_tokens <= self.lte):
            return False
        if self.gte is not None and not (info.max_tokens is not None and info.max_tokens >= self.gte):
            return False
        if self.tools is not None and info.has_tools != bool(self.tools):
            return False
        if self.stream is not None and info.stream != bool(self.stream):
            return False
        if self.system and not self.system.search(info.system_text):
            return False
        return True


def expand_env(value: str) -> str:
    """${NAME} in a configured header value is the environment's NAME; a missing one is a startup error."""
    def sub(m):
        if m.group(1) not in os.environ:
            raise ConfigError(f"fallback.headers: environment variable {m.group(1)} is not set")
        return os.environ[m.group(1)]
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, value)


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    given = {}
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                given = json.load(f)
        except (OSError, ValueError) as e:
            raise ConfigError(f"{path}: {e}") from None
    cfg = merge(DEFAULTS, given)
    for key, value in (overrides or {}).items():
        cfg[key] = value
    q = cfg["queue"]
    if not (isinstance(cfg["max_inflight"], int) and cfg["max_inflight"] >= 1):
        raise ConfigError("max_inflight: a whole number, 1 or more")
    if q["reorder_window"] < 1 or q["max_skips"] < 0 or q["max_wait_s"] < 0:
        raise ConfigError("queue: reorder_window >= 1, max_skips >= 0, max_wait_s >= 0")
    if cfg["prefix"]["bytes"] < 1 or cfg["prefix"]["messages"] < 1:
        raise ConfigError("prefix: bytes and messages are 1 or more")
    if cfg["classify"]["default"] not in (INTERACTIVE, BACKGROUND):
        raise ConfigError("classify.default: interactive or background")
    Target(cfg["upstream"])
    fb = cfg["fallback"]
    if fb["enabled"]:
        Target(fb["base_url"])
        for name in fb["classes"]:
            if name not in (INTERACTIVE, BACKGROUND):
                raise ConfigError("fallback.classes: interactive or background")
        for value in fb["headers"].values():
            expand_env(str(value))
    return cfg


# --------------------------------------------------------------------------------------------------------------- logs

def logfmt(evt: str, **kv) -> str:
    """One structured line: `ts=... evt=... key=value ...`; values with spaces are JSON-quoted."""
    parts = [f"ts={datetime.now().isoformat(timespec='milliseconds')}", f"evt={evt}"]
    for key, value in kv.items():
        if value is None:
            continue
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        if not text or any(c in text for c in ' ="\t\n'):
            text = json.dumps(text)
        parts.append(f"{key}={text}")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------ request inspection

_STR = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"', re.S)
_TOK = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"|[{\[]|[}\]]', re.S)
_LIT = re.compile(rb"[^,}\]\s]+")
_WS = re.compile(rb"\s*")


def _skip_value(b: bytes, i: int) -> int:
    c = b[i:i + 1]
    if c == b'"':
        m = _STR.match(b, i)
        if not m:
            raise ValueError("bad string")
        return m.end()
    if c in (b"{", b"["):
        depth = 0
        for m in _TOK.finditer(b, i):
            t = m.group()
            if t[:1] == b'"':
                continue
            depth += 1 if t in (b"{", b"[") else -1
            if depth == 0:
                return m.end()
        raise ValueError("unbalanced")
    m = _LIT.match(b, i)
    if not m:
        raise ValueError("bad value")
    return m.end()


def scan_members(body: bytes) -> tuple[list[tuple[str, int, int]], int, int]:
    """Top-level members of a JSON object as (key, value start, value end), plus the offsets of its braces. Only the
    structure is read, so the caller can splice bytes and leave everything else as sent."""
    i = _WS.match(body, 0).end()
    if body[i:i + 1] != b"{":
        raise ValueError("not an object")
    opening, members = i, []
    i = _WS.match(body, i + 1).end()
    while body[i:i + 1] != b"}":
        m = _STR.match(body, i)
        if not m:
            raise ValueError("bad key")
        key = json.loads(m.group())
        i = _WS.match(body, m.end()).end()
        if body[i:i + 1] != b":":
            raise ValueError("missing colon")
        start = _WS.match(body, i + 1).end()
        end = _skip_value(body, start)
        members.append((key, start, end))
        i = _WS.match(body, end).end()
        if body[i:i + 1] == b",":
            i = _WS.match(body, i + 1).end()
        elif body[i:i + 1] != b"}":
            raise ValueError("missing comma")
    return members, opening, i


def _dump(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def inject(body: bytes, edits: dict) -> bytes:
    """Apply {key: fn(old or MISSING) -> new or MISSING} to the top-level members. A member that exists is replaced in
    place; a new one is appended after the last. No other byte changes. ValueError when the body is not a JSON object."""
    members, opening, closing = scan_members(body)
    last = {}
    for key, start, end in members:
        last[key] = (start, end)               # a repeated key: the last one is what parsers read
    splices, added = [], []
    for key, fn in edits.items():
        if key in last:
            start, end = last[key]
            old = json.loads(body[start:end])
            new = fn(old)
            if new is not MISSING and new != old:
                splices.append((start, end, _dump(new)))
        else:
            new = fn(MISSING)
            if new is not MISSING:
                added.append(json.dumps(key).encode() + b":" + _dump(new))
    if added:
        at = members[-1][2] if members else opening + 1
        splices.append((at, at, (b"," if members else b"") + b",".join(added)))
    for start, end, text in sorted(splices, key=lambda s: s[0], reverse=True):
        body = body[:start] + text + body[end:]
    return body


class _Full(Exception):
    pass


def _walk(obj, out: list, state: list) -> None:
    """Compact JSON of `obj` appended to `out` until `state[1]` bytes are reached. `cache_control` markers are skipped:
    clients move them from turn to turn, and they are not part of the prompt."""
    if state[0] >= state[1]:
        raise _Full
    room = state[1] - state[0]
    if isinstance(obj, dict):
        out.append("{")
        state[0] += 1
        first = True
        for key, value in obj.items():
            if key == "cache_control":
                continue
            piece = ("" if first else ",") + json.dumps(key, ensure_ascii=False) + ":"
            first = False
            out.append(piece)
            state[0] += len(piece)
            _walk(value, out, state)
        out.append("}")
        state[0] += 1
    elif isinstance(obj, list):
        out.append("[")
        state[0] += 1
        for n, value in enumerate(obj):
            if n:
                out.append(",")
                state[0] += 1
            _walk(value, out, state)
        out.append("]")
        state[0] += 1
    else:
        piece = json.dumps(obj[:room] if isinstance(obj, str) else obj, ensure_ascii=False)
        out.append(piece)
        state[0] += len(piece)


def _head_bytes(parts: list, limit: int) -> bytes:
    out, state = [], [0, limit]
    try:
        for part in parts:
            if part is None:
                continue
            _walk(part, out, state)
            out.append("|")
            state[0] += 1
    except _Full:
        pass
    return "".join(out).encode("utf-8", "replace")[:limit]


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


class RequestInfo:
    """What the rules and the prefix hash read from a request; `obj` is None when the body is not a JSON object."""

    def __init__(self, path: str, headers: dict, body: bytes, encoded: bool = False):
        self.path, self.headers, self.raw = path, headers, body
        self.obj = None
        if not encoded:
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = None
            self.obj = parsed if isinstance(parsed, dict) else None
        o = self.obj or {}
        self.model = o.get("model") if isinstance(o.get("model"), str) else None
        self.stream = o.get("stream") is True
        self.has_tools = bool(o.get("tools"))
        self.max_tokens = None
        for field in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
            if isinstance(o.get(field), int) and not isinstance(o.get(field), bool):
                self.max_tokens = o[field]
                break

    @property
    def system_text(self) -> str:
        o = self.obj or {}
        text = _text_of(o.get("system")) or (o.get("instructions") if isinstance(o.get("instructions"), str) else "")
        if not text:
            msgs = o.get("messages") if isinstance(o.get("messages"), list) else o.get("input")
            if isinstance(msgs, list):
                text = "\n".join(_text_of(m.get("content")) for m in msgs
                                 if isinstance(m, dict) and m.get("role") in ("system", "developer"))
        return text[:4096]

    def fingerprint(self, limit: int, first: int = 1) -> tuple[str, str | None]:
        """(prefix, conversation) hashes over the start of the prompt only, so they stay the same from turn to turn.
        The prefix hash covers the first `limit` bytes of system, tools and the first `first` messages in prompt order,
        and so names the agent and its tool set. The conversation hash covers the first `limit` bytes of those
        messages alone, and so tells two sessions of one agent apart."""
        o = self.obj
        if o is None:
            return _digest(self.raw[:limit]), None
        system = o.get("system", o.get("instructions"))
        msgs = o.get("messages", o.get("input", o.get("prompt")))
        if isinstance(msgs, list):
            is_sys = lambda m: isinstance(m, dict) and m.get("role") in ("system", "developer")  # noqa: E731
            sys_msgs, head = [m for m in msgs if is_sys(m)], [m for m in msgs if not is_sys(m)][:first]
        else:
            sys_msgs, head = [], ([msgs] if msgs else [])
        prefix = _digest(_head_bytes([system, sys_msgs or None, o.get("tools"), head], limit))
        return prefix, (_digest(_head_bytes([head], limit)) if head else None)


def classify(rules: list, default: str, info: RequestInfo) -> tuple[str, str | None]:
    for rule in rules:
        if rule.matches(info):
            return rule.cls, rule.name
    return default, None


def _merge_dicts(a, b):
    return {**a, **b} if isinstance(a, dict) and isinstance(b, dict) else b


def cap_field(info: RequestInfo) -> str:
    """The output-limit field the request's API uses (the one the client already sent, when it sent one)."""
    o = info.obj or {}
    for field in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
        if field in o:
            return field
    return "max_output_tokens" if info.path.rstrip("/").endswith("/responses") else "max_tokens"


def apply_policy(rules: list, cls: str, info: RequestInfo, body: bytes) -> tuple[bytes, list]:
    """Splice the fields of every matching policy rule into `body`; returns (body, rule names applied)."""
    applied = []
    for rule in rules:
        if (rule.cls and rule.cls != cls) or not rule.matches(info):
            continue
        edits = {}
        for key, value in rule.set_default.items():
            edits[key] = (lambda v: lambda old: v if old is MISSING else (
                {**v, **old} if isinstance(v, dict) and isinstance(old, dict) else old))(value)
        for key, value in rule.force.items():
            edits[key] = (lambda v: lambda old: _merge_dicts(old, v) if old is not MISSING else v)(value)
        if rule.cap is not None:
            edits[cap_field(info)] = (lambda cap: lambda old: cap if (
                old is MISSING or isinstance(old, bool) or not isinstance(old, int) or old < 1) else min(old, cap))(rule.cap)
        if not edits:
            continue
        new = inject(body, edits)
        if new != body:
            body = new
            applied.append(rule.name)
    return body, applied


# --------------------------------------------------------------------------------------------------------- scheduler

class Ticket:
    __slots__ = ("id", "cls", "path", "model", "enq", "admitted", "prefix", "conv", "skips", "fut", "state",
                 "promoted", "timer", "fallback_ok", "rule", "stream", "outcome", "size", "match")

    def __init__(self, id: int, cls: str, path: str, model: str | None, prefix: str, conv: str | None,
                 fallback_ok: bool = False, rule: str | None = None, stream: bool = False, size: int = 0):
        self.id, self.cls, self.path, self.model, self.prefix, self.conv = id, cls, path, model, prefix, conv
        self.fallback_ok, self.rule, self.stream, self.size = fallback_ok, rule, stream, size
        self.enq = self.admitted = 0.0
        self.skips, self.promoted, self.timer, self.fut, self.outcome, self.match = 0, False, None, None, None, 0
        self.state = "new"                  # queued -> running -> done; or queued -> cancelled | fallback


class Scheduler:
    """Admission control. Order: background requests that waited `max_wait_s` (oldest first), then interactive, then
    background. Inside a tier the oldest goes first, unless one of the first `window` entries continues the last
    interactive request's conversation; the oldest can be passed `max_skips` times before it must go."""

    def __init__(self, max_inflight=1, max_wait_s=120.0, window=4, max_skips=4, fallback_after_s=None,
                 clock=time.monotonic):
        self.max_inflight, self.max_wait, self.window, self.max_skips = max_inflight, max_wait_s, window, max_skips
        self.fallback_after, self.clock = fallback_after_s, clock
        self.queue: list[Ticket] = []
        self.running: dict[int, Ticket] = {}
        self.last_key: tuple[str, str | None] | None = None
        self.counts = collections.Counter()
        self.waits = {INTERACTIVE: collections.deque(maxlen=256), BACKGROUND: collections.deque(maxlen=256)}
        self.started = time.time()

    # -- ordering
    def _overdue(self, t: Ticket, now: float) -> bool:
        return t.cls == BACKGROUND and now - t.enq >= self.max_wait

    def _score(self, t: Ticket) -> int:
        if self.last_key is None or t.prefix != self.last_key[0]:
            return 0
        return 2 if t.conv is not None and t.conv == self.last_key[1] else 1

    def _choose(self, tier: list) -> Ticket:
        oldest = tier[0]
        if self.window <= 1 or oldest.skips >= self.max_skips or self.last_key is None:
            return oldest
        best, top = oldest, self._score(oldest)
        for t in tier[1:self.window]:
            if self._score(t) > top:
                best, top = t, self._score(t)
        if best is not oldest:
            oldest.skips += 1
        return best

    def _pick(self, now: float) -> Ticket | None:
        overdue = [t for t in self.queue if self._overdue(t, now)]
        if overdue:
            return min(overdue, key=lambda t: (t.enq, t.id))
        for cls in (INTERACTIVE, BACKGROUND):
            tier = [t for t in self.queue if t.cls == cls]
            if tier:
                return self._choose(tier)
        return None

    def order(self) -> list:
        """The queue in the order it would be served if nothing else arrived (prefix reordering aside)."""
        now = self.clock()
        return sorted(self.queue, key=lambda t: (0 if self._overdue(t, now) else 1 if t.cls == INTERACTIVE else 2,
                                                 t.enq, t.id))

    # -- life cycle
    def submit(self, t: Ticket) -> int:
        """Queue `t`; returns 0 when it goes to the upstream at once, else its 1-based place in line. `t.fut`
        resolves to ("run" | "fallback", seconds waited)."""
        loop = asyncio.get_running_loop()
        t.fut, t.enq, t.state = loop.create_future(), self.clock(), "queued"
        self.queue.append(t)
        if t.fallback_ok and self.fallback_after is not None:
            t.timer = loop.call_later(self.fallback_after, self._fallback_due, t)
        self._pump()
        return 0 if t.state != "queued" else self.order().index(t) + 1

    def _pump(self) -> None:
        while self.queue and len(self.running) < self.max_inflight:
            now = self.clock()
            t = self._pick(now)
            self.queue.remove(t)
            self._admit(t, now)

    def _admit(self, t: Ticket, now: float) -> None:
        t.promoted, t.match = self._overdue(t, now), self._score(t)
        t.state, t.admitted = "running", now
        if t.timer:
            t.timer.cancel()
        self.running[t.id] = t
        wait = now - t.enq
        self.waits[t.cls].append(wait)
        self.counts[f"admitted_{t.cls}"] += 1
        if t.promoted:
            self.counts["promoted"] += 1
        if t.cls == INTERACTIVE:
            if self.last_key is not None:
                self.counts["prefix_match"] += t.match >= 1
                self.counts["conversation_match"] += t.match == 2
            self.last_key = (t.prefix, t.conv)
        if t.fut and not t.fut.done():
            t.fut.set_result(("run", wait))

    def _fallback_due(self, t: Ticket) -> None:
        if t.state != "queued":
            return
        self.queue.remove(t)
        t.state = "fallback"
        self.counts["fallback"] += 1
        if t.fut and not t.fut.done():
            t.fut.set_result(("fallback", self.clock() - t.enq))

    def finish(self, t: Ticket) -> str:
        """The request is over (done, failed or cancelled); returns the stage it was in. Frees its slot."""
        stage = t.state
        if t.timer:
            t.timer.cancel()
        if stage == "queued":
            self.queue.remove(t)
            self.counts["cancelled_queued"] += 1
        elif stage == "running":
            del self.running[t.id]
        if t.fut and not t.fut.done():
            t.fut.cancel()
        t.state = "done" if stage != "queued" else "cancelled"
        self._pump()
        return stage

    # -- reporting
    def status(self) -> dict:
        now = self.clock()

        def summary(t):
            return {"id": t.id, "class": t.cls, "path": t.path, "model": t.model, "prefix": t.prefix, "rule": t.rule,
                    "waited_s": round(now - t.enq, 3)}

        def stats(cls):
            data = sorted(self.waits[cls])
            pick = (lambda q: round(data[min(len(data) - 1, int(q * len(data)))], 3)) if data else (lambda q: None)
            return {"admitted": self.counts[f"admitted_{cls}"], "recent": len(data),
                    "wait_s": {"avg": round(sum(data) / len(data), 3) if data else None, "p50": pick(0.5),
                               "p95": pick(0.95), "max": round(data[-1], 3) if data else None}}
        order = self.order()
        n = self.counts["admitted_interactive"]
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "max_inflight": self.max_inflight,
            "queue": {"length": len(self.queue),
                      "by_class": {c: sum(t.cls == c for t in self.queue) for c in (INTERACTIVE, BACKGROUND)},
                      "items": [{**summary(t), "position": i + 1, "overdue": self._overdue(t, now)}
                                for i, t in enumerate(order)]},
            "current": ({**summary(r), "running_s": round(now - r.admitted, 3), "promoted": r.promoted}
                        if (r := next(iter(self.running.values()), None)) else None),
            "inflight": len(self.running),
            "wait": {INTERACTIVE: stats(INTERACTIVE), BACKGROUND: stats(BACKGROUND)},
            "prefix_match": {"interactive_admitted": n, "prefix": self.counts["prefix_match"],
                             "conversation": self.counts["conversation_match"],
                             "rate": round(self.counts["prefix_match"] / (n - 1), 3) if n > 1 else None,
                             "last_prefix": self.last_key[0] if self.last_key else None},
            "counters": {k: self.counts[k] for k in ("promoted", "fallback", "cancelled_queued", "cancelled_running",
                                                     "completed", "upstream_errors")},
        }


# ------------------------------------------------------------------------------------------------------------- HTTP

HOP = frozenset(("connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
                 "trailer", "transfer-encoding", "upgrade"))
REASONS = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 413: "Payload Too Large",
           431: "Request Header Fields Too Large", 500: "Internal Server Error", 502: "Bad Gateway"}
CHUNK = 65536


class BadRequest(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class ConnectionClosed(Exception):
    pass


class ClientGone(Exception):
    pass


class UpstreamError(Exception):
    pass


class Request:
    def __init__(self, method: str, target: str, version: str, headers: list, body: bytes):
        self.method, self.target, self.version, self.headers, self.body = method, target, version, headers, body
        self.path = target.split("?", 1)[0].rstrip("/") or "/"

    def get(self, name: str, default: str = "") -> str:
        name = name.lower()
        for key, value in self.headers:
            if key.lower() == name:
                return value
        return default

    def lower_headers(self) -> dict:
        out: dict = {}
        for key, value in self.headers:
            out.setdefault(key.lower(), value)
        return out


def parse_head(raw: bytes) -> tuple[str, list]:
    lines = raw.decode("latin-1").split("\r\n")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip():
            raise BadRequest(400, "malformed header line")
        headers.append((name, value.strip()))
    return lines[0], headers


async def read_head(reader: asyncio.StreamReader) -> bytes:
    try:
        return (await reader.readuntil(b"\r\n\r\n"))[:-4]
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            raise ConnectionClosed from None
        raise BadRequest(400, "truncated request") from None
    except asyncio.LimitOverrunError:
        raise BadRequest(431, "request headers too large") from None


async def iter_chunked(reader: asyncio.StreamReader, limit: int | None = None):
    """The payload pieces of a chunked body, as they arrive; the trailer is read and dropped."""
    total = 0
    while True:
        line = await reader.readline()
        if not line.endswith(b"\n"):
            raise asyncio.IncompleteReadError(line, None)
        try:
            size = int(line.split(b";", 1)[0].strip() or b"0", 16)
        except ValueError:
            raise BadRequest(400, "bad chunk size") from None
        if size == 0:
            while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                pass
            return
        total += size
        if limit is not None and total > limit:
            raise BadRequest(413, "request body too large")
        while size:
            piece = await reader.read(min(size, CHUNK))
            if not piece:
                raise asyncio.IncompleteReadError(b"", size)
            size -= len(piece)
            yield piece
        await reader.readexactly(2)


async def read_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, max_body: int) -> Request:
    start, headers = parse_head(await read_head(reader))
    parts = start.split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise BadRequest(400, "malformed request line")
    method, target, version = parts
    if not target.startswith("/"):
        raise BadRequest(400, "send origin-form request targets (a path)")
    req = Request(method, target, version, headers, b"")
    te = req.get("transfer-encoding").lower()
    length = req.get("content-length")
    if req.get("expect").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    try:
        if te:
            if te.split(",")[-1].strip() != "chunked":
                raise BadRequest(400, "unsupported transfer encoding")
            req.body = b"".join([p async for p in iter_chunked(reader, max_body)])
        elif length:
            if not length.isdigit():
                raise BadRequest(400, "bad Content-Length")
            if int(length) > max_body:
                raise BadRequest(413, "request body too large")
            req.body = await reader.readexactly(int(length))
    except asyncio.IncompleteReadError:
        raise BadRequest(400, "truncated request body") from None
    return req


class Responder:
    """Writes one response to the client, as chunked HTTP/1.1 (close-delimited for HTTP/1.0 clients)."""

    def __init__(self, writer: asyncio.StreamWriter):
        self.w, self.http11, self.chunked = writer, True, False
        self.started = self.finished = False

    async def _write(self, data: bytes) -> None:
        if self.w.is_closing():
            raise ClientGone
        self.w.write(data)
        try:
            await self.w.drain()
        except (ConnectionError, OSError):
            raise ClientGone from None

    async def head(self, status: int, reason: str, headers: list, length: int | None = None, body: bool = True) -> None:
        self.chunked = body and length is None and self.http11
        lines = [f"HTTP/1.1 {status} {reason or REASONS.get(status, '')}".rstrip()]
        lines += [f"{k}: {v}" for k, v in headers]
        if length is not None:
            lines.append(f"Content-Length: {length}")
        elif self.chunked:
            lines.append("Transfer-Encoding: chunked")
        lines.append("Connection: close")
        self.started = True
        await self._write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace"))

    async def data(self, piece: bytes) -> None:
        if piece:
            await self._write(b"%x\r\n%s\r\n" % (len(piece), piece) if self.chunked else piece)

    async def end(self) -> None:
        if self.chunked and not self.finished:
            await self._write(b"0\r\n\r\n")
        self.finished = True

    async def json(self, status: int, obj, headers: list | None = None) -> None:
        body = json.dumps(obj).encode()
        await self.head(status, "", [("Content-Type", "application/json")] + (headers or []), len(body))
        await self._write(body)
        self.finished = True


def error_body(path: str, kind: str, message: str) -> dict:
    inner = {"type": kind, "message": message}
    return {"type": "error", "error": inner} if path.startswith("/v1/messages") else {"error": inner}


def sse_error(path: str, message: str) -> bytes:
    """The error event for a stream that has already started (the status line is spent)."""
    path = path.rstrip("/")
    if path.endswith("/messages"):
        e = {"type": "error", "error": {"type": "api_error", "message": message}}
        return b"event: error\ndata: " + _dump(e) + b"\n\n"
    if path.endswith("/responses"):
        e = {"type": "response.failed", "response": {"status": "failed",
                                                      "error": {"code": "server_error", "message": message}}}
        return b"event: response.failed\ndata: " + _dump(e) + b"\n\n"
    return b"data: " + _dump({"error": {"type": "server_error", "message": message}}) + b"\n\ndata: [DONE]\n\n"


class Target:
    """An upstream base URL."""

    def __init__(self, url: str):
        u = urlsplit(url or "")
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ConfigError(f"{url!r}: expected http://HOST:PORT or https://HOST[/PREFIX]")
        self.url, self.tls, self.host = url, u.scheme == "https", u.hostname
        self.port = u.port or (443 if self.tls else 80)
        self.base = u.path.rstrip("/")
        self.authority = u.netloc.rsplit("@", 1)[-1]

    async def open(self):
        ctx = ssl.create_default_context() if self.tls else None
        return await asyncio.open_connection(self.host, self.port, ssl=ctx, limit=2 ** 18,
                                             server_hostname=self.host if self.tls else None)


def request_bytes(req: Request, target: Target, body: bytes, drop: tuple = (), add: list | None = None) -> bytes:
    """The request as sent upstream: the client's headers (hop-by-hop ones and `drop` removed), the target's Host, the
    body's own length, one-shot connection, then the headers in `add`."""
    skip = set(HOP) | {"host", "content-length", "expect"} | set(drop)
    for token in req.get("connection").split(","):
        skip.add(token.strip().lower())
    lines = [f"{req.method} {target.base}{req.target} HTTP/1.1", f"Host: {target.authority}"]
    lines += [f"{k}: {v}" for k, v in req.headers if k.lower() not in skip]
    lines += [f"{k}: {v}" for k, v in add or []]
    if body or req.method in ("POST", "PUT", "PATCH"):
        lines.append(f"Content-Length: {len(body)}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace") + body


async def read_response_head(reader: asyncio.StreamReader) -> tuple[int, str, list]:
    while True:
        try:
            raw = await read_head(reader)
        except (ConnectionClosed, BadRequest):
            raise UpstreamError("upstream closed the connection before answering") from None
        start, headers = parse_head(raw)
        parts = start.split(" ", 2)
        if len(parts) < 2 or not parts[0].startswith("HTTP/") or not parts[1].isdigit():
            raise UpstreamError("upstream sent a malformed response")
        status = int(parts[1])
        if status < 200 and status != 101:
            continue                                    # interim response
        return status, parts[2] if len(parts) > 2 else "", headers


async def response_body(reader: asyncio.StreamReader, method: str, status: int, headers: list):
    """Payload pieces of an upstream response as they arrive; UpstreamError when the framing says it was cut short."""
    h = {k.lower(): v for k, v in headers}
    if method == "HEAD" or status in (204, 304) or status < 200:
        return
    try:
        if h.get("transfer-encoding", "").lower().split(",")[-1].strip() == "chunked":
            async for piece in iter_chunked(reader):
                yield piece
        elif h.get("content-length", "").isdigit():
            left = int(h["content-length"])
            while left:
                piece = await reader.read(min(left, CHUNK))
                if not piece:
                    raise UpstreamError("upstream ended the body early")
                left -= len(piece)
                yield piece
        else:
            while True:
                piece = await reader.read(CHUNK)
                if not piece:
                    return
                yield piece
    except (asyncio.IncompleteReadError, BadRequest):
        raise UpstreamError("upstream ended the body early") from None


async def wait_eof(reader: asyncio.StreamReader) -> None:
    """Returns when the client closes its side. The request is already read, so anything else is dropped."""
    try:
        while await reader.read(4096):
            pass
    except (ConnectionError, OSError):
        pass


# --------------------------------------------------------------------------------------------------------------- proxy

class Proxy:
    def __init__(self, cfg: dict, log=None):
        self.cfg, self.log = cfg, log or (lambda line: print(line, file=sys.stderr, flush=True))
        q, fb = cfg["queue"], cfg["fallback"]
        self.upstream = Target(cfg["upstream"])
        self.fallback = Target(fb["base_url"]) if fb["enabled"] else None
        self.fb_classes = set(fb["classes"])
        self.fb_headers = [(k, expand_env(str(v))) for k, v in fb["headers"].items()] if fb["enabled"] else []
        self.sched = Scheduler(cfg["max_inflight"], q["max_wait_s"], q["reorder_window"], q["max_skips"],
                               fb["after_s"] if fb["enabled"] else None)
        self.rules = [Rule(r, "classify") for r in cfg["classify"]["rules"]]
        self.policy = [Rule(r, "policy") for r in cfg["policy"]["rules"]] if cfg["policy"]["enabled"] else []
        self.queued_paths = {p.rstrip("/") for p in cfg["queued_paths"]}
        self.allowed_hosts = {h.lower() for h in cfg["allowed_hosts"]}
        self.max_body = int(cfg["max_body_mib"] * 1048576)
        self.ids = itertools.count(1)
        self.server = None
        self.lines: collections.deque = collections.deque(maxlen=1000)

    def event(self, evt: str, **kv) -> None:
        line = logfmt(evt, **kv)
        self.lines.append(line)
        self.log(line)

    # -- connection
    async def start(self) -> int:
        host, _, port = self.cfg["listen"].rpartition(":")
        self.server = await asyncio.start_server(self.handle_client, host.strip("[]") or None, int(port),
                                                 limit=2 ** 18, backlog=256)
        bound = self.server.sockets[0].getsockname()[1]
        self.event("listen", listen=self.cfg["listen"], port=bound, upstream=self.upstream.url,
                   max_inflight=self.sched.max_inflight, policy="on" if self.policy else "off",
                   fallback="on" if self.fallback else "off")
        return bound

    async def stop(self) -> None:
        if self.server:
            self.server.close()

    def host_ok(self, host: str) -> bool:
        """DNS-rebinding check like the server's: IPs, localhost and listed names pass."""
        if "*" in self.allowed_hosts or not host.strip():
            return True
        name = host.strip().lower()
        name = name[1:].split("]")[0] if name.startswith("[") else (name.rsplit(":", 1)[0] if name.count(":") == 1 else name)
        try:
            ipaddress.ip_address(name)
            return True
        except ValueError:
            return name == "localhost" or name.endswith(".localhost") or name in self.allowed_hosts

    async def handle_client(self, reader, writer) -> None:
        resp = Responder(writer)
        try:
            try:
                req = await asyncio.wait_for(read_request(reader, writer, self.max_body), self.cfg["read_timeout_s"])
            except (ConnectionClosed, asyncio.TimeoutError, ConnectionError):
                return
            except BadRequest as e:
                await resp.json(e.status, error_body("", "invalid_request_error", e.message))
                return
            resp.http11 = req.version == "HTTP/1.1"
            if not self.host_ok(req.get("host")):
                await resp.json(403, error_body(req.path, "forbidden", f"Host {req.get('host')!r} is not allowed "
                                                "(allowed_hosts)"))
                return
            watcher = asyncio.ensure_future(wait_eof(reader))
            work = asyncio.ensure_future(self.dispatch(req, resp))
            try:
                await asyncio.wait({watcher, work}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                watcher.cancel()
                if not work.done():                     # the client left, or the proxy is stopping
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
            if work.cancelled():
                return
            exc = work.exception()
            if isinstance(exc, ClientGone):
                return
            if exc is not None:
                self.event("error", path=req.path, error=f"{type(exc).__name__}: {exc}")
                if not resp.started:
                    await resp.json(500, error_body(req.path, "server_error", "scheduler error"))
        except (ClientGone, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - one connection must not take the proxy down
            self.event("error", error=f"{type(e).__name__}: {e}")
        finally:
            writer.close()

    async def dispatch(self, req: Request, resp: Responder) -> None:
        if req.path == STATUS_PATH:
            await resp.json(200, self.sched.status())
        elif req.path.startswith("/_sched/"):
            await resp.json(404, error_body(req.path, "not_found_error", "not found"))
        elif req.method == "POST" and req.path in self.queued_paths:
            await self.inference(req, resp)
        else:
            await self.relay(req, resp, self.upstream, req.body)

    # -- inference requests
    async def inference(self, req: Request, resp: Responder) -> None:
        encoded = req.get("content-encoding").lower() not in ("", "identity")
        info = RequestInfo(req.path, req.lower_headers(), req.body, encoded)
        if info.obj is None and not encoded:            # not JSON: the upstream answers 400 without using the engine
            await self.relay(req, resp, self.upstream, req.body)
            return
        cls, rule = classify(self.rules, self.cfg["classify"]["default"], info)
        body, applied = req.body, []
        if self.policy and not encoded:
            body, applied = apply_policy(self.policy, cls, info, req.body)
        prefix, conv = info.fingerprint(self.cfg["prefix"]["bytes"], self.cfg["prefix"]["messages"])
        t = Ticket(next(self.ids), cls, req.path, info.model, prefix, conv,
                   fallback_ok=bool(self.fallback) and cls in self.fb_classes, rule=rule, stream=info.stream,
                   size=len(req.body))
        position = self.sched.submit(t)
        self.event("enqueue", id=t.id, cls=cls, rule=rule, path=req.path, model=info.model, position=position,
                   prefix=prefix, conv=conv, bytes=len(req.body), stream=int(info.stream),
                   policy=",".join(applied) or None, queue=len(self.sched.queue), running=len(self.sched.running))
        extra = [("X-Queue-Position", str(position)), ("X-Queue-Class", cls), ("X-Queue-Id", str(t.id))]
        try:
            early = await self.wait_turn(t, resp, extra)
            mode, wait = t.fut.result()
            self.event("fallback" if mode == "fallback" else "admit", id=t.id, cls=cls, wait_ms=round(wait * 1000),
                       promoted=int(t.promoted) or None, match=t.match if cls == INTERACTIVE else None)
            started = time.monotonic()
            extra.append(("X-Queue-Wait-Ms", str(round(wait * 1000))))
            if mode == "fallback":
                t.outcome = await self.fallback_relay(req, resp, early, extra + [("X-Queue-Fallback", "1")])
            else:
                t.outcome = await self.relay(req, resp, self.upstream, body, extra, early)
            self.event("done", id=t.id, cls=cls, outcome=t.outcome, run_ms=round((time.monotonic() - started) * 1000))
        except (asyncio.CancelledError, ClientGone):
            t.outcome = "client_gone"
            raise
        finally:
            stage = self.sched.finish(t)
            if t.outcome == "client_gone":
                self.sched.counts["cancelled_running"] += stage in ("running", "fallback")
                self.event("cancel", id=t.id, cls=cls, stage=stage)
            elif t.outcome == "ok":
                self.sched.counts["completed"] += 1

    async def wait_turn(self, t: Ticket, resp: Responder, extra: list) -> bool:
        """Wait for admission. A streaming client that waits long gets its response started early (200 plus SSE comments
        naming its place in line), so its idle timeout does not fire. Returns whether the response head was sent."""
        q, early = self.cfg["queue"], False
        ping_at = float(q["keepalive_after_s"]) if (t.stream and q["keepalive_after_s"]) else None
        while not t.fut.done():
            waited = self.sched.clock() - t.enq
            await asyncio.wait({t.fut}, timeout=None if ping_at is None else max(0.0, ping_at - waited))
            if t.fut.done():
                break
            waited = self.sched.clock() - t.enq
            if not early:
                await resp.head(200, "OK", [("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                                            ("X-Accel-Buffering", "no")] + extra)
                early = True
            place = self.sched.order().index(t) + 1 if t in self.sched.queue else 0
            await resp.data(f": queued position={place} waited={waited:.0f}s\n\n".encode())
            ping_at = waited + float(q["keepalive_every_s"])
        return early

    async def fallback_relay(self, req: Request, resp: Responder, early: bool, extra: list) -> str:
        fb, body = self.cfg["fallback"], req.body
        if fb["force"]:
            body = inject(body, {k: (lambda v: lambda old: _merge_dicts(old, v) if old is not MISSING else v)(v)
                                 for k, v in fb["force"].items()})
        drop = tuple(h for h in ("authorization", "x-api-key") if not fb["forward_client_auth"])
        drop += tuple(k.lower() for k, _ in self.fb_headers)
        outcome = await self.relay(req, resp, self.fallback, body, extra, early, drop, self.fb_headers)
        return "fallback" if outcome == "ok" else f"fallback_{outcome}"

    async def relay(self, req: Request, resp: Responder, target: Target, body: bytes, extra: list | None = None,
                    early: bool = False, drop: tuple = (), add: list | None = None) -> str:
        """Send the request to `target` and stream its answer back; returns the outcome. Closing the upstream connection
        on the way out, normal or not, is what frees the upstream's slot when the client has left."""
        up_writer, extra = None, extra or []
        try:
            try:
                up_reader, up_writer = await asyncio.wait_for(target.open(), self.cfg["connect_timeout_s"])
                up_writer.write(request_bytes(req, target, body, drop, add))
                await up_writer.drain()
                status, reason, uheaders = await read_response_head(up_reader)
            except (OSError, asyncio.TimeoutError, UpstreamError) as e:
                return await self.upstream_failed(req, resp, early, extra, f"upstream unavailable: {e or type(e).__name__}")
            ok = 200 <= status < 300
            if early and not ok:
                await resp.data(sse_error(req.path, await self.error_text(up_reader, req, status, uheaders)))
                await resp.end()
                self.sched.counts["upstream_errors"] += 1
                return f"upstream_{status}"
            is_sse = early or "event-stream" in dict((k.lower(), v) for k, v in uheaders).get("content-type", "")
            if not early:
                skip = set(HOP) | {"content-length"}
                for k, v in uheaders:
                    if k.lower() == "connection":
                        skip |= {tok.strip().lower() for tok in v.split(",")}
                lengths = [v for k, v in uheaders if k.lower() == "content-length" and v.isdigit()]
                chunked_up = any(k.lower() == "transfer-encoding" for k, _ in uheaders)
                nobody = req.method == "HEAD" or status in (204, 304)
                if nobody:
                    skip.discard("content-length")
                await resp.head(status, reason, [(k, v) for k, v in uheaders if k.lower() not in skip] + extra,
                                int(lengths[0]) if lengths and not chunked_up and not nobody else None, not nobody)
            try:
                async for piece in response_body(up_reader, req.method, status, uheaders):
                    await resp.data(piece)
            except UpstreamError as e:
                self.sched.counts["upstream_errors"] += 1
                if is_sse and resp.chunked:             # a stream says so; any other body is cut so the client sees it
                    await resp.data(sse_error(req.path, str(e)))
                    await resp.end()
                return "upstream_truncated"
            await resp.end()
            return "ok" if ok else f"upstream_{status}"
        finally:
            if up_writer is not None:
                up_writer.close()

    async def upstream_failed(self, req: Request, resp: Responder, early: bool, extra: list, message: str) -> str:
        self.sched.counts["upstream_errors"] += 1
        if early:
            await resp.data(sse_error(req.path, message))
            await resp.end()
        else:
            await resp.json(502, error_body(req.path, "upstream_unavailable", message), extra)
        return "upstream_unavailable"

    async def error_text(self, reader, req: Request, status: int, headers: list) -> str:
        raw = b""
        try:
            async for piece in response_body(reader, req.method, status, headers):
                raw += piece
                if len(raw) > 65536:
                    break
        except UpstreamError:
            pass
        try:
            err = json.loads(raw).get("error")
            message = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            message = None
        return f"upstream {status}: {message or raw[:300].decode('utf-8', 'replace')}"


# ------------------------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Priority and prefix-aware request scheduler for a single-slot LLM server")
    ap.add_argument("--config", help="JSON settings file (see scheduler.example.json)")
    ap.add_argument("--listen", help="HOST:PORT to listen on (default 127.0.0.1:8081)")
    ap.add_argument("--upstream", help="base URL of the server (default http://127.0.0.1:8080)")
    ap.add_argument("--max-inflight", type=int, help="requests admitted to the upstream at once (default 1)")
    ap.add_argument("--log-file", help="append log lines here as well as to stderr")
    return ap


async def run(proxy: Proxy) -> None:
    await proxy.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            pass                                        # Windows: Ctrl+C raises KeyboardInterrupt instead
    await stop.wait()
    await proxy.stop()


def main(argv: list | None = None) -> int:
    a = build_parser().parse_args(argv)
    overrides = {k: v for k, v in (("listen", a.listen), ("upstream", a.upstream),
                                   ("max_inflight", a.max_inflight)) if v is not None}
    try:
        cfg = load_config(a.config, overrides)
    except ConfigError as e:
        print(f"[scheduler] error: {e}", file=sys.stderr)
        return 2
    path = a.log_file or cfg["log"]["file"]
    sink = open(path, "a", encoding="utf-8", buffering=1) if path else None

    def log(line: str) -> None:
        print(line, file=sys.stderr, flush=True)
        if sink:
            sink.write(line + "\n")
    try:
        proxy = Proxy(cfg, log)
    except ConfigError as e:
        print(f"[scheduler] error: {e}", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(proxy))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"[scheduler] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
