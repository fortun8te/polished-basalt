"""Pure logic: queue ordering with a fake clock, classification rules, the byte-preserving injector, prefix hashes."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import scheduler_proxy as sp


def run(coro_fn):
    asyncio.run(coro_fn())


def sched(**kw):
    now = [0.0]
    return sp.Scheduler(clock=lambda: now[0], **kw), now


def tk(i, cls="interactive", prefix="p", conv=None):
    return sp.Ticket(i, cls, "/v1/chat/completions", "m", prefix, conv)


def drain(s):
    """Finish whatever runs until the queue is empty; returns the ids in the order they were admitted."""
    order = []
    while s.running:
        s.finish(next(iter(s.running.values())))
        order.extend(s.running)
    return order


# ------------------------------------------------------------------------------------------------------- scheduler

def test_strict_priority_and_fifo_within_a_class():
    async def go():
        s, _ = sched()
        assert s.submit(tk(0)) == 0
        for t in (tk(1, "background"), tk(2), tk(3, "background"), tk(4)):
            s.submit(t)
        assert drain(s) == [2, 4, 1, 3]
    run(go)


def test_position_counts_places_ahead_in_priority_order():
    async def go():
        s, _ = sched()
        assert s.submit(tk(0)) == 0
        assert s.submit(tk(1, "background")) == 1
        assert s.submit(tk(2)) == 1             # interactive passes the queued background request
        assert s.submit(tk(3, "background")) == 3
    run(go)


def test_overdue_background_goes_before_interactive():
    async def go():
        s, now = sched(max_wait_s=120)
        s.submit(tk(0))
        s.submit(tk(1, "background"))
        now[0] = 100
        s.submit(tk(2, "background"))
        now[0] = 125                             # 1 has waited 125 s, 2 only 25 s
        s.submit(tk(3))
        assert drain(s) == [1, 3, 2]
        assert s.counts["promoted"] == 1
    run(go)


def test_overdue_backgrounds_keep_arrival_order():
    async def go():
        s, now = sched(max_wait_s=10)
        s.submit(tk(0))
        s.submit(tk(1, "background"))
        now[0] = 1
        s.submit(tk(2, "background"))
        now[0] = 50
        s.submit(tk(3))
        assert drain(s) == [1, 2, 3]
    run(go)


def test_prefix_match_wins_inside_the_window():
    async def go():
        s, _ = sched(window=4)
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, prefix="B", conv="b"))
        s.submit(tk(2, prefix="A", conv="a"))
        assert drain(s) == [2, 1]
    run(go)


def test_same_conversation_beats_same_agent():
    async def go():
        s, _ = sched()
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, prefix="A", conv="other"))
        s.submit(tk(2, prefix="A", conv="a"))
        assert drain(s) == [2, 1]
    run(go)


def test_window_limits_how_far_a_match_may_jump():
    async def go():
        s, _ = sched(window=2)
        s.submit(tk(0, prefix="A", conv="a"))
        for i, p in ((1, "B"), (2, "C"), (3, "A")):
            s.submit(tk(i, prefix=p, conv=p.lower()))
        assert drain(s) == [1, 2, 3]
    run(go)


def test_oldest_cannot_be_passed_more_than_max_skips_times():
    async def go():
        s, _ = sched(max_skips=2)
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, prefix="B", conv="b"))
        for i in (2, 3, 4):
            s.submit(tk(i, prefix="A", conv="a"))
        assert drain(s) == [2, 3, 1, 4]
    run(go)


def test_window_of_one_is_plain_fifo():
    async def go():
        s, _ = sched(window=1)
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, prefix="B", conv="b"))
        s.submit(tk(2, prefix="A", conv="a"))
        assert drain(s) == [1, 2]
    run(go)


def test_background_work_does_not_replace_the_resident_conversation():
    async def go():
        s, _ = sched()
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, "background", prefix="Z", conv="z"))
        s.submit(tk(2, prefix="A", conv="a"))
        drain(s)
        assert s.last_key == ("A", "a")
    run(go)


def test_max_inflight_admits_several_and_queues_the_rest():
    async def go():
        s, _ = sched(max_inflight=2)
        assert [s.submit(tk(i)) for i in range(3)] == [0, 0, 1]
        s.finish(s.running[0])
        assert set(s.running) == {1, 2}
    run(go)


def test_finish_of_a_queued_ticket_removes_it_and_counts():
    async def go():
        s, _ = sched()
        s.submit(tk(0))
        t = tk(1)
        s.submit(t)
        assert s.finish(t) == "queued"
        assert not s.queue and s.counts["cancelled_queued"] == 1
        assert s.finish(s.running[0]) == "running"
        assert not s.running
    run(go)


def test_fallback_timer_takes_a_queued_background_ticket_out():
    async def go():
        s, _ = sched(fallback_after_s=0.05)
        s.submit(tk(0))
        t, u = tk(1, "background"), tk(2)
        t.fallback_ok = True
        s.submit(t)
        s.submit(u)
        mode, _ = await asyncio.wait_for(t.fut, 2)
        assert mode == "fallback" and t not in s.queue and u in s.queue and not u.fut.done()
    run(go)


def test_status_reports_waits_and_match_rate():
    async def go():
        s, now = sched()
        s.submit(tk(0, prefix="A", conv="a"))
        s.submit(tk(1, prefix="A", conv="a"))
        s.submit(tk(2, prefix="B", conv="b"))
        now[0] = 2
        drain(s)
        st = s.status()
        assert st["prefix_match"]["interactive_admitted"] == 3
        assert st["prefix_match"]["rate"] == 0.5 and st["prefix_match"]["conversation"] == 1
        assert st["wait"]["interactive"]["wait_s"]["max"] == 2
        assert st["queue"]["length"] == 0 and st["current"] is None
    run(go)


# ------------------------------------------------------------------------------------------------ classification

def info(body, path="/v1/chat/completions", **headers):
    return sp.RequestInfo(path, {k.lower().replace("_", "-"): v for k, v in headers.items()}, json.dumps(body).encode())


def rules(*raw):
    return [sp.Rule(r, "classify") for r in raw]


def klass(rs, i, default="interactive"):
    return sp.classify(rs, default, i)[0]


def test_default_rules_cover_header_model_and_tiny_untooled_calls():
    rs = rules(*sp.DEFAULT_RULES)
    assert klass(rs, info({"model": "q"}, x_priority="Background")) == "background"
    assert klass(rs, info({"model": "claude-3-5-haiku-latest"})) == "background"
    assert klass(rs, info({"model": "q", "max_tokens": 64})) == "background"
    assert klass(rs, info({"model": "q", "max_tokens": 64, "tools": [{"name": "x"}]})) == "interactive"
    assert klass(rs, info({"model": "q", "max_tokens": 32000})) == "interactive"
    assert klass(rs, info({"model": "q"})) == "interactive"
    assert klass(rs, info({"model": "q"}, x_priority="interactive")) == "interactive"


def test_rule_conditions_are_anded_and_the_first_match_decides():
    rs = rules({"name": "a", "model": "qwen", "path": "/messages$", "class": "background"},
               {"name": "b", "stream": False, "class": "background"})
    assert sp.classify(rs, "interactive", info({"model": "qwen", "stream": True}, "/v1/messages")) == ("background", "a")
    assert sp.classify(rs, "interactive", info({"model": "qwen", "stream": True})) == ("interactive", None)
    assert sp.classify(rs, "interactive", info({"model": "x", "stream": False})) == ("background", "b")


def test_system_regex_reads_the_system_prompt_of_each_api():
    rs = rules({"system_regex": "write a title", "class": "background"})
    assert klass(rs, info({"messages": [{"role": "system", "content": "Please WRITE A TITLE."}]})) == "background"
    assert klass(rs, info({"system": [{"type": "text", "text": "write a title"}]}, "/v1/messages")) == "background"
    assert klass(rs, info({"instructions": "write a title", "input": "x"}, "/v1/responses")) == "background"
    assert klass(rs, info({"messages": [{"role": "user", "content": "write a title"}]})) == "interactive"


def test_classify_default_can_be_background():
    assert klass([], info({}), default="background") == "background"


@pytest.mark.parametrize("raw", [{"class": "background"}, {"model": "(", "class": "background"},
                                 {"model": "x", "class": "urgent"}, {"model": "x", "colour": 1, "class": "background"}])
def test_bad_rules_are_refused(raw):
    with pytest.raises(sp.ConfigError):
        sp.Rule(raw, "classify")


def test_unknown_config_keys_are_refused(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"queue": {"max_wait": 5}}))
    with pytest.raises(sp.ConfigError, match="queue.max_wait"):
        sp.load_config(str(path))


# ---------------------------------------------------------------------------------------------------------- inject

AWKWARD = (b'{ "a" : "}{\\"," ,\n  "messages":[{"role":"user","content":"x, \\u00e9 ] }"}],\n'
           b'  "max_tokens" : 4096 , "n":null,"deep":{"max_tokens":1,"k":[1,[2,{"z":"}"}]]} }\n')


def test_inject_replaces_in_place_and_appends_without_touching_other_bytes():
    out = sp.inject(AWKWARD, {"max_tokens": lambda old: 100, "extra": lambda old: {"b": True}})
    assert json.loads(out) == {**json.loads(AWKWARD), "max_tokens": 100, "extra": {"b": True}}
    restored = out.replace(b'"max_tokens" : 100', b'"max_tokens" : 4096').replace(b',"extra":{"b":true}', b"")
    assert restored == AWKWARD


def test_inject_into_empty_objects_and_missing_keys():
    assert sp.inject(b"{}", {"a": lambda o: 1}) == b'{"a":1}'
    assert sp.inject(b"{ }", {"a": lambda o: 1}) == b'{"a":1 }'
    assert sp.inject(b'{"x":1}', {"a": lambda o: sp.MISSING}) == b'{"x":1}'
    assert sp.inject(b'{"a":1}', {"a": lambda o: o}) == b'{"a":1}'


def test_inject_changes_the_last_of_a_repeated_key():
    assert sp.inject(b'{"a":1,"a":2}', {"a": lambda o: o + 10}) == b'{"a":1,"a":12}'


def test_inject_sees_escaped_key_names():
    assert sp.inject(b'{"max\\u005ftokens":9}', {"max_tokens": lambda o: 5}) == b'{"max\\u005ftokens":5}'


@pytest.mark.parametrize("bad", [b"[1]", b'"x"', b"", b'{"a":', b'{"a" 1}', b'{"a":1 "b":2}'])
def test_inject_refuses_what_is_not_an_object(bad):
    with pytest.raises(ValueError):
        sp.inject(bad, {"a": lambda o: 1})


# ----------------------------------------------------------------------------------------------------------- policy

def policy(*raw):
    return [sp.Rule(r, "policy") for r in raw]


def apply(rs, body, path="/v1/chat/completions", cls="background"):
    raw = json.dumps(body).encode()
    return sp.apply_policy(rs, cls, sp.RequestInfo(path, {}, raw), raw), raw


def test_cap_lowers_but_never_raises_and_names_the_field_of_the_api():
    rs = policy({"name": "cap", "max_tokens_cap": 100})
    (out, applied), raw = apply(rs, {"model": "q", "max_tokens": 4096})
    assert json.loads(out)["max_tokens"] == 100 and applied == ["cap"]
    (out, applied), raw = apply(rs, {"model": "q", "max_tokens": 50})
    assert out == raw and applied == []
    (out, _), _ = apply(rs, {"model": "q", "max_completion_tokens": 900})
    assert json.loads(out)["max_completion_tokens"] == 100 and "max_tokens" not in json.loads(out)
    (out, _), _ = apply(rs, {"model": "q", "max_tokens": -1})
    assert json.loads(out)["max_tokens"] == 100
    (out, _), _ = apply(rs, {"model": "q"}, "/v1/responses")
    assert json.loads(out)["max_output_tokens"] == 100


def test_set_default_keeps_what_the_client_sent_and_merges_template_kwargs():
    rs = policy({"set_default": {"reasoning_budget_tokens": 256,
                                 "chat_template_kwargs": {"enable_thinking": False, "keep": 1}}})
    (out, _), _ = apply(rs, {"model": "q", "reasoning_budget_tokens": 8000, "chat_template_kwargs": {"keep": 2}})
    got = json.loads(out)
    assert got["reasoning_budget_tokens"] == 8000
    assert got["chat_template_kwargs"] == {"enable_thinking": False, "keep": 2}


def test_force_overrides_and_rule_class_and_conditions_filter():
    rs = policy({"class": "background", "model": "haiku", "force": {"reasoning_effort": "low"}})
    (out, _), _ = apply(rs, {"model": "haiku-x", "reasoning_effort": "high"})
    assert json.loads(out)["reasoning_effort"] == "low"
    (out, applied), raw = apply(rs, {"model": "qwen"})
    assert out == raw and not applied
    (out, applied), raw = apply(rs, {"model": "haiku-x"}, cls="interactive")
    assert out == raw and not applied


# ---------------------------------------------------------------------------------------------------- fingerprints

def fp(body, path="/v1/chat/completions", limit=8192, first=1):
    return sp.RequestInfo(path, {}, json.dumps(body).encode()).fingerprint(limit, first)


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "first"}]


def test_hashes_are_the_same_on_every_turn_of_a_conversation():
    turn2 = MSGS + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    assert fp({"messages": MSGS, "tools": [{"n": 1}]}) == fp({"messages": turn2, "tools": [{"n": 1}], "stream": True})


def test_same_agent_other_session_shares_the_prefix_but_not_the_conversation():
    tools = [{"n": 1}] * 2000                     # longer than the hashed head, as an agent's tool list is
    a = fp({"messages": MSGS, "tools": tools})
    b = fp({"messages": [MSGS[0], {"role": "user", "content": "second"}], "tools": tools})
    assert a[0] == b[0] and a[1] != b[1]


def test_other_system_prompt_or_tools_change_the_prefix():
    assert fp({"messages": MSGS})[0] != fp({"messages": [{"role": "system", "content": "T"}, MSGS[1]]})[0]
    assert fp({"messages": MSGS, "tools": [1]})[0] != fp({"messages": MSGS, "tools": [2]})[0]


def test_cache_control_markers_are_not_part_of_the_hash():
    plain = {"system": [{"type": "text", "text": "S"}], "messages": [{"role": "user", "content": [
        {"type": "text", "text": "first"}]}]}
    marked = json.loads(json.dumps(plain))
    marked["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
    marked["system"][0]["cache_control"] = {"type": "ephemeral"}
    assert fp(plain, "/v1/messages") == fp(marked, "/v1/messages")


def test_only_the_first_k_bytes_count():
    long_a, long_b = "x" * 5000 + "A", "x" * 5000 + "B"
    assert fp({"system": long_a, "messages": [{"role": "user", "content": "u"}]}, limit=1000)[0] == \
        fp({"system": long_b, "messages": [{"role": "user", "content": "u"}]}, limit=1000)[0]
    assert fp({"system": long_a, "messages": [{"role": "user", "content": "u"}]}, limit=8000)[0] != \
        fp({"system": long_b, "messages": [{"role": "user", "content": "u"}]}, limit=8000)[0]


def test_responses_api_and_plain_text_bodies_hash_too():
    a = fp({"instructions": "I", "input": [{"role": "user", "content": "q"}]}, "/v1/responses")
    b = fp({"instructions": "I", "input": [{"role": "user", "content": "q"}, {"type": "function_call_output"}]},
           "/v1/responses")
    assert a == b and a[1] is not None
    assert sp.RequestInfo("/x", {}, b"not json").fingerprint(100) == (sp._digest(b"not json"), None)
    assert fp({"input": "just a string"}, "/v1/responses")[1] is not None


# ----------------------------------------------------------------------------------------------- example config

EXAMPLE = Path(__file__).resolve().parents[1] / "scheduler.example.json"


def test_the_example_config_loads_with_every_optional_part_switched_on(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUD_API_KEY", "k")
    proxy = sp.Proxy(sp.load_config(str(EXAMPLE)), log=lambda line: None)
    assert not proxy.policy and proxy.fallback is None
    raw = json.loads(EXAMPLE.read_text())
    raw["policy"]["enabled"] = raw["fallback"]["enabled"] = True
    path = tmp_path / "on.json"
    path.write_text(json.dumps(raw))
    proxy = sp.Proxy(sp.load_config(str(path)), log=lambda line: None)
    assert proxy.policy and proxy.fallback and dict(proxy.fb_headers)["x-api-key"] == "k"


def test_the_example_rules_split_agent_turns_from_helper_calls():
    proxy = sp.Proxy(sp.load_config(str(EXAMPLE)), log=lambda line: None)

    def cls(body, **headers):
        i = info(body, "/v1/messages", **headers)
        return sp.classify(proxy.rules, "interactive", i)[0]
    tools = [{"name": "Bash"}]
    assert cls({"model": "qwen", "max_tokens": 32000, "tools": tools}) == "interactive"
    assert cls({"model": "qwen"}) == "interactive"
    assert cls({"model": "claude-haiku-4-5", "max_tokens": 4000, "tools": tools}) == "background"
    assert cls({"model": "flashnext-iq3s.basalt", "max_tokens": 4000, "tools": tools}) == "background"
    assert cls({"model": "qwen", "max_tokens": 1}) == "background"
    assert cls({"model": "claude-haiku-4-5"}, x_priority="interactive") == "interactive"
    assert cls({"model": "qwen", "tools": tools}, x_priority="background") == "background"
