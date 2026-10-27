"""The proxy end to end: a real listening proxy, a fake slow streaming upstream, raw-socket clients."""
from __future__ import annotations

import http.client
import json
import re
import socket
import time

import pytest

import scheduler_proxy as sp
from conftest import chat_body, enqueue, hold, wait_until
from mock_upstream import MockUpstream

BG = {"X-Priority": "background"}
TURN2 = [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]


def finish(*clients):
    for c in clients:
        c.wait()
        assert c.status == 200, c.head


# ------------------------------------------------------------------------------------------------ priority classes

def test_interactive_turns_go_before_background_calls(upstream, proxy):
    r0 = hold(upstream, proxy)
    b1 = enqueue(proxy, chat_body("B1", user="b1"), 1, headers=BG)
    i1 = enqueue(proxy, chat_body("I1", user="i1"), 2)
    b2 = enqueue(proxy, chat_body("B2", user="b2"), 3, headers=BG)
    i2 = enqueue(proxy, chat_body("I2", user="i2"), 4)
    upstream.release("g0")
    finish(r0, b1, i1, b2, i2)
    assert upstream.tags() == ["R0", "I1", "I2", "B1", "B2"]
    assert upstream.max_active == 1
    assert [c.headers["x-queue-position"] for c in (r0, b1, i1, b2, i2)] == ["0", "1", "1", "3", "2"]
    assert [c.headers["x-queue-class"] for c in (b1, i1)] == ["background", "interactive"]


def test_haiku_model_and_tiny_untooled_calls_count_as_background(upstream, proxy):
    r0 = hold(upstream, proxy)
    small = enqueue(proxy, chat_body("SMALL", model="claude-haiku-4-5"), 1)
    tiny = enqueue(proxy, chat_body("TINY", max_tokens=16), 2)
    main = enqueue(proxy, chat_body("MAIN", max_tokens=32000), 3)
    upstream.release("g0")
    finish(r0, small, tiny, main)
    assert upstream.tags() == ["R0", "MAIN", "SMALL", "TINY"]


def test_a_background_call_waiting_past_max_wait_is_promoted(upstream, make_proxy):
    p = make_proxy({"queue": {"max_wait_s": 0.3}})
    r0 = hold(upstream, p)
    b1 = enqueue(p, chat_body("B1"), 1, headers=BG)
    time.sleep(0.4)
    i1 = enqueue(p, chat_body("I1"), 2)
    items = p.status()["queue"]["items"]
    assert [(i["class"], i["overdue"]) for i in items] == [("background", True), ("interactive", False)]
    upstream.release("g0")
    finish(r0, b1, i1)
    assert upstream.tags() == ["R0", "B1", "I1"]
    assert p.status()["counters"]["promoted"] == 1


def test_background_is_served_when_nothing_interactive_waits(upstream, proxy):
    finish(proxy.post(chat_body("ONLY", user="x"), headers=BG))
    assert upstream.tags() == ["ONLY"]


# ----------------------------------------------------------------------------------------------- prefix ordering

def test_the_conversation_the_upstream_just_served_goes_first(upstream, proxy):
    r0 = hold(upstream, proxy, system="agent-1", user="conv-A")
    x = enqueue(proxy, chat_body("X", system="agent-2", user="conv-X"), 1)
    a2 = enqueue(proxy, chat_body("A2", system="agent-1", user="conv-A", extra_messages=TURN2), 2)
    upstream.release("g0")
    finish(r0, x, a2)
    assert upstream.tags() == ["R0", "A2", "X"]
    m = proxy.status()["prefix_match"]
    assert (m["interactive_admitted"], m["prefix"], m["conversation"], m["rate"]) == (3, 1, 1, 0.5)


def test_same_agent_other_session_loses_to_the_resident_conversation(upstream, proxy):
    tools = [{"type": "function", "function": {"name": f"t{i}", "description": "d" * 50}} for i in range(400)]
    r0 = hold(upstream, proxy, system="agent-1", user="conv-A", tools=tools)
    y = enqueue(proxy, chat_body("Y", system="agent-1", user="conv-B", tools=tools), 1)
    a2 = enqueue(proxy, chat_body("A2", system="agent-1", user="conv-A", extra_messages=TURN2, tools=tools), 2)
    upstream.release("g0")
    finish(r0, y, a2)
    assert upstream.tags() == ["R0", "A2", "Y"]


def test_the_reorder_window_bounds_how_far_a_match_jumps(upstream, make_proxy):
    p = make_proxy({"queue": {"reorder_window": 2}})
    r0 = hold(upstream, p, system="agent-1", user="conv-A")
    x1 = enqueue(p, chat_body("X1", system="agent-2", user="x1"), 1)
    x2 = enqueue(p, chat_body("X2", system="agent-3", user="x2"), 2)
    a2 = enqueue(p, chat_body("A2", system="agent-1", user="conv-A", extra_messages=TURN2), 3)
    upstream.release("g0")
    finish(r0, x1, x2, a2)
    assert upstream.tags() == ["R0", "X1", "X2", "A2"]


def test_an_overtaken_request_goes_next_after_max_skips(upstream, make_proxy):
    p = make_proxy({"queue": {"max_skips": 1}})
    r0 = hold(upstream, p, system="agent-1", user="conv-A")
    x = enqueue(p, chat_body("X", system="agent-2", user="x"), 1)
    a2 = enqueue(p, chat_body("A2", system="agent-1", user="conv-A", extra_messages=TURN2), 2)
    a3 = enqueue(p, chat_body("A3", system="agent-1", user="conv-A", extra_messages=TURN2 * 2), 3)
    upstream.release("g0")
    finish(r0, x, a2, a3)
    assert upstream.tags() == ["R0", "A2", "X", "A3"]


def test_a_matching_prefix_never_beats_priority(upstream, proxy):
    r0 = hold(upstream, proxy, system="agent-1", user="conv-A")
    bg = enqueue(proxy, chat_body("BG", system="agent-1", user="conv-A", extra_messages=TURN2), 1, headers=BG)
    other = enqueue(proxy, chat_body("OTHER", system="agent-2", user="conv-O"), 2)
    upstream.release("g0")
    finish(r0, bg, other)
    assert upstream.tags() == ["R0", "OTHER", "BG"]


# ------------------------------------------------------------------------------------------------- cancellation

def test_a_client_that_leaves_the_queue_is_dropped(upstream, proxy):
    r0 = hold(upstream, proxy)
    q1 = enqueue(proxy, chat_body("Q1"), 1)
    q2 = enqueue(proxy, chat_body("Q2"), 2)
    q1.abort()
    wait_until(lambda: proxy.queue_len() == 1, what="the queue to drop Q1")
    upstream.release("g0")
    finish(r0, q2)
    time.sleep(0.1)
    assert upstream.tags() == ["R0", "Q2"]
    assert proxy.status()["counters"]["cancelled_queued"] == 1


def test_a_client_that_leaves_a_running_request_frees_the_slot(upstream, proxy):
    slow = proxy.post(chat_body("SLOW", mock={"events": 400, "delay": 0.02}))     # 8 s if left alone
    slow.wait_for(b'"i": 1')
    nxt = proxy.post(chat_body("NEXT"))
    wait_until(lambda: proxy.queue_len() == 1, what="NEXT to queue")
    left = time.monotonic()
    slow.abort()
    finish(nxt)
    assert nxt.t_done - left < 2.0
    wait_until(lambda: upstream.disconnects, what="the upstream to see the hang-up")
    assert upstream.by_tag("SLOW").gone_at - left < 1.0
    wait_until(lambda: proxy.status()["counters"]["cancelled_running"] == 1, what="the cancel to be counted")
    assert proxy.status()["inflight"] == 0


def test_a_client_that_leaves_before_its_turn_with_an_early_response_is_dropped_too(upstream, make_proxy):
    p = make_proxy({"queue": {"keepalive_after_s": 0.1, "keepalive_every_s": 0.1}})
    r0 = hold(upstream, p)
    q = enqueue(p, chat_body("Q"), 1)
    q.wait_for(b": queued position=1")
    q.abort()
    wait_until(lambda: p.queue_len() == 0, what="the queue to drop Q")
    upstream.release("g0")
    finish(r0)
    assert upstream.tags() == ["R0"]


# ------------------------------------------------------------------------------------ streaming and passthrough

def test_sse_events_reach_the_client_before_the_upstream_is_done(upstream, proxy):
    c = proxy.post(chat_body("S", mock={"events": 5, "gate": "g", "gate_after": 2, "mid_gate": "mid"}))
    upstream.release("g")
    c.wait_for(b'"i": 1')
    assert not c.finished.is_set() and not c.has(b'"i": 2')
    upstream.release("mid")
    finish(c)
    assert [e["i"] for e in c.events() if e != "[DONE]"] == [0, 1, 2, 3, 4] and c.events()[-1] == "[DONE]"
    assert c.headers["content-type"] == "text/event-stream"
    assert c.headers["transfer-encoding"] == "chunked" and c.headers["x-upstream-tag"] == "S"
    assert c.headers["x-queue-position"] == "0"


def test_json_responses_keep_their_content_length(upstream, proxy):
    c = proxy.post(chat_body("J", stream=False))
    finish(c)
    assert json.loads(c.body) == {"tag": "J", "object": "chat.completion"}
    assert c.headers["content-length"] == str(len(c.body)) and "transfer-encoding" not in c.headers


def test_an_upstream_that_chunks_its_stream_is_passed_on_whole(upstream, proxy):
    c = proxy.post(chat_body("CH", mock={"chunked": True, "events": 4}))
    finish(c)
    assert len([e for e in c.events() if e != "[DONE]"]) == 4 and c.events()[-1] == "[DONE]"


def test_http_1_0_clients_get_a_close_delimited_stream(upstream, proxy):
    c = proxy.post(chat_body("OLD"), version="HTTP/1.0")
    finish(c)
    assert "transfer-encoding" not in c.headers and c.has(b"data: [DONE]")


def test_headers_pass_through_and_hop_by_hop_ones_do_not(upstream, proxy):
    sent = {"Authorization": "Bearer sk-local", "x-api-key": "key-1", "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024", "X-Custom": "abc", "Accept": "text/event-stream", "User-Agent": "tester/1",
            "Connection": "keep-alive", "Keep-Alive": "timeout=5", "Proxy-Authorization": "secret"}
    body = chat_body("H", stream=False)
    finish(proxy.post(body, path="/v1/messages?beta=true", headers=sent))
    seen = {k.lower(): v for k, v in upstream.by_tag("H").headers}
    for k, v in sent.items():
        if k not in ("Connection", "Keep-Alive", "Proxy-Authorization"):
            assert seen[k.lower()] == v, k
    assert seen["connection"] == "close" and "keep-alive" not in seen and "proxy-authorization" not in seen
    assert seen["host"] == f"127.0.0.1:{upstream.port}" and seen["content-length"] == str(len(body))
    assert upstream.by_tag("H").path == "/v1/messages?beta=true"


def test_an_upstream_error_comes_back_with_its_status_and_body(upstream, proxy):
    c = proxy.post(chat_body("E", mock={"status": 400, "error": "prompt too long"}))
    c.wait()
    assert c.status == 400 and json.loads(c.body)["error"]["message"] == "prompt too long"
    assert c.headers["x-queue-position"] == "0"


def test_chunked_request_bodies_reach_the_upstream_whole(upstream, proxy):
    body = chat_body("CK", stream=False)
    pieces = [body[:20], body[20:50], body[50:]]
    wire = b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in pieces) + b"0\r\n\r\n"
    finish(proxy.post(wire, headers={"Transfer-Encoding": "chunked"}))
    rec = upstream.by_tag("CK")
    assert rec.body == body and dict((k.lower(), v) for k, v in rec.headers)["content-length"] == str(len(body))


def test_expect_continue_is_answered_before_the_body_is_read(upstream, proxy):
    body = chat_body("EX", stream=False)
    s = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
    s.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {len(body)}\r\n"
              "Expect: 100-continue\r\n\r\n".encode())
    assert s.recv(100).startswith(b"HTTP/1.1 100 Continue")
    s.sendall(body)
    data = b""
    while chunk := s.recv(65536):
        data += chunk
    assert data.startswith(b"HTTP/1.1 200") and upstream.tags() == ["EX"]


def test_other_paths_pass_straight_through_even_while_the_slot_is_busy(upstream, proxy):
    r0 = hold(upstream, proxy)
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3)
    conn.request("GET", "/v1/models")
    resp = conn.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["data"][0]["id"] == "mock-model"
    assert proxy.queue_len() == 0
    upstream.release("g0")
    finish(r0)


def test_a_body_that_is_not_json_goes_to_the_upstream_unqueued(upstream, proxy):
    r0 = hold(upstream, proxy)
    c = proxy.post(b"not json at all")
    wait_until(lambda: len(upstream.requests) == 2, what="the bad request to be forwarded")
    assert proxy.queue_len() == 0
    upstream.release("g0")
    finish(r0)
    c.wait()


def test_a_dead_upstream_gives_502_and_does_not_hold_the_slot(make_proxy, tmp_path):
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = dead.getsockname()[1]
    dead.close()
    up = MockUpstream()
    up.port = port
    p = make_proxy(up=up)
    for _ in range(2):
        c = p.post(chat_body("D"))
        c.wait()
        assert c.status == 502 and json.loads(c.body)["error"]["type"] == "upstream_unavailable"
    assert p.status()["inflight"] == 0


def test_an_upstream_that_cuts_the_body_short_is_visible_to_the_client(upstream, proxy):
    c = proxy.post(chat_body("CUT", stream=False, mock={"truncate": True}))
    c.wait()
    assert c.headers["content-length"] == "100" and len(c.body) < 100
    assert proxy.status()["counters"]["upstream_errors"] == 1


def test_a_second_slot_admits_two_requests_at_once(upstream, make_proxy):
    p = make_proxy({"max_inflight": 2})
    r0 = hold(upstream, p, "R0", "g0")
    r1 = hold(upstream, p, "R1", "g1")
    r2 = enqueue(p, chat_body("R2"), 1)
    assert upstream.max_active == 2 and p.status()["inflight"] == 2
    upstream.release("g0")
    upstream.release("g1")
    finish(r0, r1, r2)


def test_hosts_are_checked_like_the_server_does(upstream, make_proxy):
    p = make_proxy({"allowed_hosts": ["llm.lan"]})
    for host, code in (("evil.example", 403), ("llm.lan:8081", 200), ("localhost:8081", 200), ("[::1]:8081", 200),
                       ("192.168.1.5", 200)):
        c = p.post(chat_body("HOST", stream=False), headers={"Host": host})
        c.wait()
        assert c.status == code, host


# ------------------------------------------------------------------------------------- waiting clients get a sign

def test_a_queued_stream_gets_its_response_started_and_kept_alive(upstream, make_proxy):
    p = make_proxy({"queue": {"keepalive_after_s": 0.2, "keepalive_every_s": 0.2}})
    r0 = hold(upstream, p)
    s = enqueue(p, chat_body("S"), 1)
    s.wait_for(b": queued position=1")
    assert s.status == 200 and s.headers["x-queue-position"] == "1" and s.headers["content-type"] == "text/event-stream"
    assert len(re.findall(rb": queued position=1", bytes(s.raw))) >= 1
    upstream.release("g0")
    finish(r0, s)
    assert s.events()[-1] == "[DONE]" and upstream.tags() == ["R0", "S"]


def test_a_queued_request_that_is_not_streaming_gets_nothing_until_its_turn(upstream, make_proxy):
    p = make_proxy({"queue": {"keepalive_after_s": 0.1, "keepalive_every_s": 0.1}})
    r0 = hold(upstream, p)
    n = enqueue(p, chat_body("N", stream=False), 1)
    time.sleep(0.4)
    assert bytes(n.raw) == b""
    upstream.release("g0")
    finish(r0, n)
    assert n.headers["x-queue-position"] == "1"


def test_an_error_after_the_early_response_arrives_as_a_stream_event(upstream, make_proxy):
    p = make_proxy({"queue": {"keepalive_after_s": 0.1, "keepalive_every_s": 0.1}})
    r0 = hold(upstream, p)
    s = enqueue(p, chat_body("S", mock={"status": 400, "error": "prompt too long"}), 1)
    s.wait_for(b": queued")
    upstream.release("g0")
    finish(r0, s)
    events = s.events()
    assert "prompt too long" in events[0]["error"]["message"] and events[-1] == "[DONE]"


# ------------------------------------------------------------------------------------------------------ policy

def spaced(tag: str, **fields) -> bytes:
    """A body in an awkward but valid layout: indents, escapes, the output cap in the middle."""
    obj = {"model": "qwen", "stream": False, "messages": [{"role": "system", "content": "S {\"}"},
                                                         {"role": "user", "content": "héllo 日本, ]"}],
           "max_tokens": 4096, "temperature": 0.2, **fields, "_mock": {"tag": tag}}
    return json.dumps(obj, indent=1).encode()


POLICY = {"policy": {"enabled": True, "rules": [{
    "name": "lean", "class": "background", "max_tokens_cap": 100,
    "set_default": {"reasoning_budget_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}}]}}
ADDED = b',"reasoning_budget_tokens":256,"chat_template_kwargs":{"enable_thinking":false}'


def test_policy_splices_fields_and_leaves_every_other_byte_alone(upstream, make_proxy):
    p = make_proxy(POLICY)
    orig = spaced("BG")
    finish(p.post(orig, headers=BG))
    sent = upstream.by_tag("BG").body
    assert sent != orig
    assert sent.replace(ADDED, b"", 1).replace(b'"max_tokens": 100', b'"max_tokens": 4096', 1) == orig
    got = json.loads(sent)
    assert got["max_tokens"] == 100 and got["reasoning_budget_tokens"] == 256
    assert got["messages"] == json.loads(orig)["messages"]
    before = sp.RequestInfo("/x", {}, orig).fingerprint(8192)
    assert sp.RequestInfo("/x", {}, sent).fingerprint(8192) == before


def test_policy_leaves_interactive_requests_untouched(upstream, make_proxy):
    p = make_proxy(POLICY)
    orig = spaced("IA")
    finish(p.post(orig))
    assert upstream.by_tag("IA").body == orig


def test_policy_is_off_by_default(upstream, proxy):
    orig = spaced("BG")
    finish(proxy.post(orig, headers=BG))
    assert upstream.by_tag("BG").body == orig


def test_policy_never_raises_a_cap_the_client_set_lower(upstream, make_proxy):
    p = make_proxy(POLICY)
    orig = spaced("LOW").replace(b'"max_tokens": 4096', b'"max_tokens": 50')
    finish(p.post(orig, headers=BG))
    assert upstream.by_tag("LOW").body.replace(ADDED, b"", 1) == orig


# ------------------------------------------------------------------------------------------------------ fallback

def test_background_calls_waiting_too_long_go_to_the_alternate_upstream(upstream, make_proxy, monkeypatch):
    monkeypatch.setenv("CLOUD_KEY", "cloud-secret")
    cloud = MockUpstream().start()
    try:
        p = make_proxy({"fallback": {"enabled": True, "base_url": cloud.url, "after_s": 0.2,
                                     "headers": {"x-api-key": "${CLOUD_KEY}"}, "force": {"model": "cloud-small"}}})
        r0 = hold(upstream, p)
        i1 = enqueue(p, chat_body("I1"), 1)
        b1 = p.post(chat_body("B1", stream=False), headers={**BG, "Authorization": "Bearer local-only"})
        finish(b1)
        assert b1.headers["x-queue-fallback"] == "1" and upstream.tags() == ["R0"]
        seen = {k.lower(): v for k, v in cloud.by_tag("B1").headers}
        assert seen["x-api-key"] == "cloud-secret" and "authorization" not in seen
        assert json.loads(cloud.by_tag("B1").body)["model"] == "cloud-small"
        assert p.queue_len() == 1 and p.status()["counters"]["fallback"] == 1     # I1 still waits for the local slot
        upstream.release("g0")
        finish(r0, i1)
        assert upstream.tags() == ["R0", "I1"]
    finally:
        cloud.stop()


def test_nothing_is_forwarded_elsewhere_unless_enabled(upstream, proxy):
    r0 = hold(upstream, proxy)
    b1 = enqueue(proxy, chat_body("B1"), 1, headers=BG)
    time.sleep(0.4)
    assert proxy.queue_len() == 1 and "x-queue-fallback" not in b1.headers
    upstream.release("g0")
    finish(r0, b1)


def test_fallback_needs_its_settings_at_startup(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"fallback": {"enabled": True, "base_url": "ftp://x"}}))
    with pytest.raises(sp.ConfigError):
        sp.load_config(str(path))
    path.write_text(json.dumps({"fallback": {"enabled": True, "base_url": "https://x.example",
                                             "headers": {"authorization": "Bearer ${NOPE_NOT_SET_XYZ}"}}}))
    with pytest.raises(sp.ConfigError, match="NOPE_NOT_SET_XYZ"):
        sp.load_config(str(path))


# ------------------------------------------------------------------------------------------- status and logging

def test_status_endpoint_shows_queue_current_request_and_waits(upstream, proxy):
    r0 = hold(upstream, proxy, system="agent-1", user="conv-A")
    enqueue(proxy, chat_body("B1"), 1, headers=BG)
    enqueue(proxy, chat_body("I1"), 2)
    st = proxy.status()
    assert st["max_inflight"] == 1 and st["inflight"] == 1
    assert st["queue"]["by_class"] == {"interactive": 1, "background": 1}
    assert [(i["class"], i["position"]) for i in st["queue"]["items"]] == [("interactive", 1), ("background", 2)]
    assert st["current"]["class"] == "interactive" and st["current"]["path"] == "/v1/chat/completions"
    assert st["current"]["running_s"] >= 0 and re.fullmatch(r"[0-9a-f]{16}", st["current"]["prefix"])
    upstream.release("g0")
    wait_until(lambda: proxy.status()["queue"]["length"] == 0 and proxy.status()["inflight"] == 0)
    wait_until(lambda: proxy.status()["counters"]["completed"] == 3, what="all three to complete")
    st = proxy.status()
    assert st["wait"]["interactive"]["admitted"] == 2 and st["wait"]["background"]["admitted"] == 1
    assert st["wait"]["background"]["wait_s"]["max"] > 0 and st["prefix_match"]["last_prefix"]
    assert upstream.tags() == ["R0", "I1", "B1"]
    finish(r0)


def test_status_and_unknown_scheduler_paths_are_answered_locally(upstream, proxy):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3)
    conn.request("GET", "/_sched/nope")
    assert conn.getresponse().status == 404
    proxy.status()
    assert upstream.requests == []


def test_every_request_leaves_structured_log_lines(upstream, proxy):
    finish(proxy.post(chat_body("L", model="claude-haiku-4-5")))
    wait_until(lambda: any(" evt=done " in ln for ln in proxy.proxy.lines), what="the done line")
    rows = [dict(re.findall(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)', ln)) for ln in proxy.proxy.lines]
    evts = {r["evt"]: r for r in rows}
    assert {"listen", "enqueue", "admit", "done"} <= set(evts)
    e, d = evts["enqueue"], evts["done"]
    assert e["id"] == d["id"] == evts["admit"]["id"] and e["cls"] == "background" and e["rule"] == "small-model"
    assert d["outcome"] == "ok" and int(d["run_ms"]) >= 0 and re.fullmatch(r"[0-9a-f]{16}", e["prefix"])
    assert all(ln.startswith("ts=") and "\n" not in ln for ln in proxy.proxy.lines)

@pytest.mark.parametrize("headers,body", [
    (b"Content-Length: 2\r\nContent-Length: 4\r\n", b"{}xx"),
    (b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n", b"0\r\n\r\n"),
    (b"Transfer-Encoding: gzip, chunked\r\n", b"0\r\n\r\n"),
    (b"Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n", b"0\r\n\r\n"),
    (b"Content-Length: \r\n", b""),
    (b"Content-Length: " + b"9" * 5000 + b"\r\n", b""),
    (b"Transfer-Encoding: chunked\r\n", b"-1\r\n"),
    (b"Transfer-Encoding: chunked\r\n", b"1\r\nxZZ0\r\n\r\n"),
    (b"Transfer-Encoding: chunked\r\n", b"\r\n"),
    (b"Bad Header: value\r\n", b""),
    (b"X-Test: value\rbroken\r\n", b""),
])
def test_ambiguous_or_malformed_requests_are_rejected_before_forwarding(upstream, proxy, headers, body):
    head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n"
    c = proxy.post(body, raw_head=head)
    c.wait()
    assert c.status == 400
    assert upstream.requests == []


def test_all_connection_header_tokens_are_removed_before_forwarding(upstream, proxy):
    body = chat_body("CONNECTION", stream=False)
    head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Connection: keep-alive\r\nConnection: x-secret\r\nX-Secret: private\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode())
    finish(proxy.post(body, raw_head=head))
    seen = {k.lower(): v for k, v in upstream.by_tag("CONNECTION").headers}
    assert "x-secret" not in seen
    assert seen["connection"] == "close"


@pytest.mark.parametrize("response", [
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 4\r\n\r\n{}xx",
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked\r\n\r\n0\r\n\r\n",
    "HTTP/1.1 200 OK\r\nInvalid Header: x\r\n\r\n",
])
def test_bad_upstream_framing_returns_502_and_releases_slot(upstream, proxy, response):
    c = proxy.post(chat_body("BAD", stream=False, mock={"raw_response": response}))
    c.wait()
    assert c.status == 502
    finish(proxy.post(chat_body("NEXT", stream=False)))
    assert upstream.tags() == ["BAD", "NEXT"]
    assert proxy.status()["inflight"] == 0
