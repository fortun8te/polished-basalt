"""Fixtures and helpers: a scheduler proxy and a mock upstream in background threads, blocking raw-socket clients."""
from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import scheduler_proxy as sp  # noqa: E402
from mock_upstream import LoopThread, MockUpstream  # noqa: E402


def wait_until(pred, timeout: float = 5.0, interval: float = 0.01, what: str = "condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")


def chat_body(tag: str, system: str = "You are an agent.", user: str = "hello", stream: bool = True,
              extra_messages: list | None = None, model: str = "qwen", mock: dict | None = None, **fields) -> bytes:
    """A chat-completions request. The first user message names the conversation, the system prompt names the agent."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}] + (extra_messages or [])
    obj = {"model": model, "messages": messages, "stream": stream, **fields,
           "_mock": {"tag": tag, **(mock or {})}}
    return json.dumps(obj, separators=(",", ":")).encode()


class Client:
    """One request on a raw socket, read by a background thread so a test can look at it while it is in flight."""

    def __init__(self, port: int, body: bytes = b"", path: str = "/v1/chat/completions", headers: dict | None = None,
                 method: str = "POST", version: str = "HTTP/1.1", raw_head: bytes | None = None):
        self.raw = bytearray()
        self.finished = threading.Event()
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=15)
        hdrs = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", **(headers or {})}
        if (body or method == "POST") and "Transfer-Encoding" not in hdrs:
            hdrs.setdefault("Content-Length", str(len(body)))
        head = raw_head or (f"{method} {path} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
                            + "\r\n").encode()
        try:
            self.sock.sendall(head + body)
        except OSError:
            self.sock.close()
            raise
        self.t0 = time.monotonic()
        self.t_done = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    break
                self.raw += data
        except OSError:
            pass
        finally:
            self.sock.close()
            self.t_done = time.monotonic()
            self.finished.set()

    def abort(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def wait(self, timeout: float = 10) -> "Client":
        assert self.finished.wait(timeout), f"no complete response; got {bytes(self.raw)[:300]!r}"
        return self

    def has(self, needle: bytes) -> bool:
        return needle in bytes(self.raw)

    def wait_for(self, needle: bytes, timeout: float = 5):
        wait_until(lambda: self.has(needle), timeout, what=f"{needle!r} in the response")

    @property
    def head(self) -> bytes:
        return bytes(self.raw).split(b"\r\n\r\n", 1)[0]

    @property
    def status(self) -> int:
        return int(self.head.split(b" ", 2)[1])

    @property
    def headers(self) -> dict:
        lines = self.head.decode("latin-1").split("\r\n")[1:]
        return {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines)}

    @property
    def body(self) -> bytes:
        """The payload, with chunked framing removed when the response used it."""
        raw = bytes(self.raw)
        if b"\r\n\r\n" not in raw:
            return b""
        payload = raw.split(b"\r\n\r\n", 1)[1]
        if "chunked" not in self.headers.get("transfer-encoding", ""):
            return payload
        out = b""
        while payload:
            size_line, _, rest = payload.partition(b"\r\n")
            size = int(size_line, 16)
            if size == 0:
                break
            out += rest[:size]
            payload = rest[size + 2:]
        return out

    def events(self) -> list:
        """The `data:` payloads of an SSE body, parsed (the [DONE] marker comes back as the string)."""
        out = []
        for block in self.body.decode().split("\n\n"):
            for line in block.split("\n"):
                if line.startswith("data: "):
                    text = line[6:]
                    out.append(text if text == "[DONE]" else json.loads(text))
        return out


class ProxyHarness:
    def __init__(self, tmp_path, upstream_url: str, config: dict | None = None):
        cfg = {"listen": "127.0.0.1:0", "upstream": upstream_url, **(config or {})}
        path = Path(tmp_path) / "sched.json"
        path.write_text(json.dumps(cfg))
        self.cfg = sp.load_config(str(path))
        self.lt = LoopThread()
        self.proxy = sp.Proxy(self.cfg, log=lambda line: None)

        async def boot():
            return await self.proxy.start()
        self.port = self.lt.call(boot())

    def status(self) -> dict:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", sp.STATUS_PATH)
        data = json.loads(conn.getresponse().read())
        conn.close()
        return data

    def queue_len(self) -> int:
        return self.status()["queue"]["length"]

    def post(self, body: bytes, **kw) -> Client:
        return Client(self.port, body, **kw)

    def stop(self):
        async def down():
            await self.proxy.stop()
        self.lt.call(down())
        self.lt.stop()


@pytest.fixture
def upstream():
    up = MockUpstream().start()
    yield up
    up.stop()


@pytest.fixture
def make_proxy(tmp_path, upstream):
    made = []

    def make(config: dict | None = None, up=None) -> ProxyHarness:
        folder = tmp_path / f"p{len(made)}"
        folder.mkdir()
        h = ProxyHarness(folder, (up or upstream).url, config)
        made.append(h)
        return h
    yield make
    for h in made:
        h.stop()


@pytest.fixture
def proxy(make_proxy):
    return make_proxy()


def hold(up: MockUpstream, proxy: ProxyHarness, tag: str = "R0", gate: str = "g0", **body_kw) -> Client:
    """Occupy the upstream's one slot with a request that waits for `up.release(gate)`."""
    c = proxy.post(chat_body(tag, mock={"gate": gate}, **body_kw))
    wait_until(lambda: tag in up.tags(), what=f"{tag} to reach the upstream")
    return c


def enqueue(proxy: ProxyHarness, body: bytes, depth: int, **kw) -> Client:
    """Send a request and wait until the proxy reports `depth` requests queued, so the arrival order is known."""
    c = proxy.post(body, **kw)
    wait_until(lambda: proxy.queue_len() == depth, what=f"queue length {depth}")
    return c
