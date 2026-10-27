"""Bound request framing and stalled clients without timing out model work."""
import json
import socket
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from serve import handler
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.prompt import PromptText
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


class HttpLimits(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(handler, 'HANDLER_MAX', 2),
                        patch.object(handler, 'HEADER_DEADLINE', .35),
                        patch.object(handler, 'BODY_DEADLINE', .45),
                        patch.object(handler, 'CHUNK_MAX', 128)]
        for p in self.patches:
            p.start()
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, 'hello', max_context=4096), tok,
                      ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        self.svc = svc
        self.httpd = handler.serve(svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for p in reversed(self.patches):
            p.stop()

    def connect(self):
        return socket.create_connection(('127.0.0.1', self.port), timeout=2)

    def send(self, payload):
        with self.connect() as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
            result = b''
            while True:
                part = sock.recv(65536)
                if not part:
                    return result
                result += part

    @staticmethod
    def chunked(data, size):
        return b''.join(b'%x\r\n' % len(data[i:i+size]) + data[i:i+size] + b'\r\n'
                        for i in range(0, len(data), size)) + b'0\r\n\r\n'

    def post(self, obj, path='/v1/chat/completions', chunks=None):
        data = json.dumps(obj).encode()
        framing = (b'Transfer-Encoding: chunked\r\n' if chunks else
                   b'Content-Length: %d\r\n' % len(data))
        return self.send(b'POST ' + path.encode() + b' HTTP/1.0\r\nHost: localhost\r\n' + framing +
                         b'Content-Type: application/json\r\n\r\n' +
                         (self.chunked(data, chunks) if chunks else data))

    def test_healthy_chunked_and_tiny_chunks(self):
        req = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': 'hello'}]}
        self.assertIn(b' 200 ', self.post(req, chunks=7))
        self.assertIn(b' 200 ', self.post(req, chunks=1))
        req['padding'] = 'a' * 200
        self.assertIn(b' 413 ', self.post(req, chunks=1))

    def test_bad_nested_fields_are_client_errors(self):
        base = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': 'hi'}]}
        for extra in ({'chat_template_kwargs': ['x']}, {'max_tokens': {}}, {'max_tokens': []}):
            with self.subTest(extra=extra):
                self.assertIn(b' 400 ', self.post({**base, **extra}))
        for content in ([1], 1, [{'type': 'text', 'text': 'hi'}, None], [{'type': 'text', 'text': 1}]):
            with self.subTest(content=content):
                req = {'model': base['model'], 'messages': [{'role': 'user', 'content': content}]}
                self.assertIn(b' 400 ', self.post(req, '/v1/messages'))

    def test_stalled_headers_and_body_release_bounded_slots(self):
        with self.connect() as header, self.connect() as body:
            header.sendall(b'GET /health HTTP/1.0\r\nHost: local')
            body.sendall(b'POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Length: 9\r\n\r\n{')
            time.sleep(.1)
            with self.connect() as overflow:
                overflow.sendall(b'GET /health HTTP/1.0\r\n\r\n')
                try:
                    self.assertEqual(overflow.recv(1024), b'')
                except ConnectionResetError:
                    pass
            # Send progress during each deadline; a trickle must not extend it.
            time.sleep(.2)
            header.sendall(b'h')
            body.sendall(b' ')
            time.sleep(.3)
            for client in (header, body):
                try:
                    self.assertEqual(client.recv(1024), b'')
                except ConnectionResetError:
                    pass
        self.assertIn(b' 200 ', self.send(b'GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n'))

    def test_stream_priming_releases_heartbeat_and_preserves_errors(self):
        prompt = PromptText(ByteTokenizer(), 'hi')
        advanced = []
        def queued():
            yield 'start'
            yield None
            advanced.append(True)
            prompt.set_counted(2, 10)
            yield 'answer'
        events = handler._stream_primed(queued(), prompt)
        self.assertEqual(advanced, [])
        self.assertEqual(list(events), ['start', None, 'answer'])
        self.assertEqual(advanced, [True])

        def fail_early():
            yield 'start'
            raise ValueError('context exceeded')
        with self.assertRaisesRegex(ValueError, 'context'):
            handler._stream_primed(fail_early(), PromptText(ByteTokenizer(), 'hi'))

        def fail_late():
            yield None
            raise ValueError('context exceeded')
        events = handler._stream_primed(fail_late(), PromptText(ByteTokenizer(), 'hi'))
        self.assertIsNone(next(events))
        with self.assertRaisesRegex(ValueError, 'context'):
            next(events)

    def test_queued_stream_sends_heartbeat_before_prompt_count(self):
        for path in ('/v1/chat/completions', '/v1/messages', '/v1/responses'):
            with self.subTest(path=path):
                prompt = PromptText(ByteTokenizer(), 'hi')
                def run(*args, **kwargs):
                    yield 'ping', None
                    time.sleep(.6)
                    prompt.set_counted(2, 10)
                    yield 'done', {'finish': 'stop', 'completion_tokens': 0, 'prompt_tokens': 2}
                req = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': 'hi'}],
                       'input': 'hi', 'stream': True}
                data = json.dumps(req).encode()
                with patch.object(self.svc, 'prepare', return_value=(prompt, False, 10)), \
                     patch.object(self.svc, 'run', side_effect=run), self.connect() as sock:
                    sock.sendall(b'POST ' + path.encode() + b' HTTP/1.0\r\nHost: localhost\r\nContent-Length: %d\r\n\r\n' % len(data) + data)
                    sock.settimeout(.3)
                    received = b''
                    while b': keep-alive\n\n' not in received:
                        part = sock.recv(4096)
                        self.assertTrue(part, received)
                        received += part
                    self.assertIsNone(prompt.n)
                    sock.settimeout(2)
                    while True:
                        part = sock.recv(4096)
                        if not part:
                            break
                        received += part
                    self.assertEqual(prompt.n, 2)
                    self.assertIn(b' 200 ', received)
                    if path == '/v1/messages':
                        self.assertIn(b'"input_tokens": 2', received)
                    if path == '/v1/responses':
                        self.assertIn(b'response.completed', received)

    def test_deadline_ends_before_processing(self):
        # No request deadline remains while a valid request is being processed.
        original = handler.make_handler
        def delayed(svc):
            base = original(svc)
            class Slow(base):
                def _openai(self, req):
                    time.sleep(.6)
                    return super()._openai(req)
            return Slow
        self.httpd.shutdown()
        self.httpd.server_close()
        with patch.object(handler, 'make_handler', delayed):
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, 'hello', max_context=4096), tok,
                          ChatTemplate(ROOT / 'serve/chat_template.jinja'))
            self.httpd = handler.serve(svc, port=0)
            self.port = self.httpd.server_address[1]
        req = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': 'hi'}]}
        self.assertIn(b' 200 ', self.post(req))


if __name__ == '__main__':
    unittest.main()
