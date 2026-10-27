"""A fake single-slot LLM server for the scheduler tests.

It speaks the way the real one does: HTTP/1.0 with a close-delimited SSE stream, and it notices a client that hangs up.
A request steers it through a `_mock` object in its JSON body:

    tag         name recorded for the request (tests assert on arrival order by tag)
    gate        hold the response until the test calls `release(gate)`
    gate_after  hold again after this many events, until `release(mid_gate)`
    events      number of SSE events (default 3), `delay` seconds apart (default 0.005)
    status      answer with this error status and a JSON error body
    chunked     answer with Transfer-Encoding: chunked instead of close-delimited
    truncate    promise a long Content-Length and cut the body short
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace


class LoopThread:
    """An asyncio loop running in its own thread, for blocking test code to call into."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout: float = 10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


class Gone(Exception):
    pass


class MockUpstream:
    def __init__(self):
        self.requests: list = []
        self.disconnects: list = []
        self.active = self.max_active = 0
        self.gates: dict = {}
        self.lt = LoopThread()
        self.port = 0

    def start(self) -> "MockUpstream":
        async def boot():
            self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
            return self.server.sockets[0].getsockname()[1]
        self.port = self.lt.call(boot())
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        async def down():
            self.server.close()
        self.lt.call(down())
        self.lt.stop()

    def _gate(self, name: str) -> asyncio.Event:
        if name not in self.gates:
            self.gates[name] = asyncio.Event()
        return self.gates[name]

    def release(self, name: str):
        self.lt.loop.call_soon_threadsafe(lambda: self._gate(name).set())

    def tags(self) -> list:
        return [r.tag for r in self.requests]

    def by_tag(self, tag: str):
        return next(r for r in self.requests if r.tag == tag)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        rec = None
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head[:-4].decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = [(k, v.strip()) for k, _, v in (line.partition(":") for line in lines[1:])]
            length = int(dict((k.lower(), v) for k, v in headers).get("content-length", 0))
            body = await reader.readexactly(length) if length else b""
            try:
                ctl = (json.loads(body) or {}).get("_mock", {}) if body else {}
            except ValueError:
                ctl = {}
            rec = SimpleNamespace(method=method, path=target, headers=headers, body=body, tag=ctl.get("tag"),
                                  t0=time.monotonic(), gone_at=None, done=False, ctl=ctl)
            self.requests.append(rec)
            if method == "GET":
                data = json.dumps({"object": "list", "data": [{"id": "mock-model"}]}).encode()
                writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
                             % (len(data), data))
                await writer.drain()
                return
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await self.answer(reader, writer, rec, json.loads(body) if body else {})
            finally:
                self.active -= 1
        except (asyncio.IncompleteReadError, ConnectionError, Gone):
            if rec is not None and not rec.done:
                rec.gone_at = time.monotonic()
                self.disconnects.append(rec)
        finally:
            writer.close()

    async def answer(self, reader, writer, rec, req):
        ctl = rec.ctl
        watch = asyncio.ensure_future(reader.read(1))      # EOF: the client left (the real server polls for this)

        async def pause(seconds=None, event=None):
            waits = {watch}
            if event is not None:
                waits.add(asyncio.ensure_future(event.wait()))
            done, pending = await asyncio.wait(waits, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
            for p in pending - {watch}:
                p.cancel()
            if watch.done() and not watch.result():
                raise Gone

        try:
            if ctl.get("gate"):
                await pause(event=self._gate(ctl["gate"]))
            tag = rec.tag
            if ctl.get("status"):
                data = json.dumps({"error": {"type": "invalid_request_error", "message": ctl.get("error", "bad")}}).encode()
                writer.write(b"HTTP/1.0 %d Error\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
                             % (ctl["status"], len(data), data))
                await writer.drain()
            elif ctl.get("truncate"):
                writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"cut\":")
                await writer.drain()
            elif not req.get("stream"):
                data = json.dumps({"tag": tag, "object": "chat.completion"}).encode()
                writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nX-Upstream-Tag: %s\r\n"
                             b"Content-Length: %d\r\n\r\n%s" % (str(tag).encode(), len(data), data))
                await writer.drain()
            else:
                chunked = ctl.get("chunked")
                head = b"HTTP/1.1 200 OK\r\n" if chunked else b"HTTP/1.0 200 OK\r\n"
                head += b"Content-Type: text/event-stream\r\nX-Upstream-Tag: %s\r\n" % str(tag).encode()
                head += b"Transfer-Encoding: chunked\r\n\r\n" if chunked else b"\r\n"
                writer.write(head)

                def frame(payload: bytes) -> bytes:
                    return b"%x\r\n%s\r\n" % (len(payload), payload) if chunked else payload
                for i in range(int(ctl.get("events", 3))):
                    writer.write(frame(b"data: " + json.dumps({"tag": tag, "i": i}).encode() + b"\n\n"))
                    await writer.drain()
                    if ctl.get("gate_after") == i + 1:
                        await pause(event=self._gate(ctl["mid_gate"]))
                    await pause(float(ctl.get("delay", 0.005)))
                writer.write(frame(b"data: [DONE]\n\n") + (b"0\r\n\r\n" if chunked else b""))
                await writer.drain()
            rec.done = True
        finally:
            watch.cancel()
