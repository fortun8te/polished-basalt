"""Synthetic checks for settled text caps, reasoning limits and queued cancellation."""
import threading
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.prompt import CTX_SLACK, PromptText
from serve.service import Service
from serve.util.text import ByteTokenizer, IM_END

ROOT = Path(__file__).resolve().parents[2]


class CountedEngine(MockEngine):
    takes_text = True

    def __init__(self, tok, script, ctx=140):
        super().__init__(tok, script, max_context=ctx)
        self.calls = []

    def generate(self, prompt, max_new, sampling, cancel, embeddings=None):
        if isinstance(prompt, PromptText):
            ids = prompt.ids()
            room = self.max_context - CTX_SLACK - len(ids)
            max_new = min(max_new, room) if prompt.clip else max_new
            prompt.set_counted(len(ids), max_new)
            yield None
        else:
            ids = prompt
        self.calls.append((len(ids), max_new))
        if max_new < 1 or len(ids) + max_new + CTX_SLACK > self.max_context:
            raise AssertionError("invalid continuation")
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class BudgetHardening(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def make(self, script, ctx=140):
        engine = CountedEngine(self.tok, script, ctx)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.repeat_stop_tokens = 0
        return svc, engine

    def run_text(self, svc, length, cap, sampling=None, force=None):
        prompt = PromptText(self.tok, "a" * length, clip=True)
        prompt.max_new = cap
        events = list(svc.run(prompt, True, None, cap, sampling or {}, threading.Event(), force))
        return events[-1][1]

    def test_clipped_cap_prevents_oversized_wrap(self):
        for requested in (132, 100):  # unset cap and fit-max-tokens
            with self.subTest(requested=requested):
                svc, engine = self.make("abcd")
                result = self.run_text(svc, 100, requested, {"reasoning_budget_tokens": 1})
                self.assertEqual(result["finish"], "length")
                self.assertEqual(result["completion_tokens"], 1)
                self.assertEqual(engine.calls, [(100, 32)])

    def test_clipped_cap_prevents_oversized_early_close_and_forced_call(self):
        cases = [("", 131, None, "stop"), ("</think>" , 100, "X" * 40, "length")]
        for script, length, force, finish in cases:
            with self.subTest(script=script):
                svc, engine = self.make(script)
                result = self.run_text(svc, length, 132, force=force)
                self.assertEqual(result["finish"], finish)
                self.assertEqual(len(engine.calls), 1)

    def test_continuation_when_wrapper_fits(self):
        svc, engine = self.make(["abcd", "</think>ok"], ctx=300)
        result = self.run_text(svc, 100, 192, {"reasoning_budget_tokens": 1})
        self.assertEqual(result["finish"], "stop")
        self.assertEqual(len(engine.calls), 2)
        self.assertTrue(all(cap > 0 and n + cap + CTX_SLACK <= 300 for n, cap in engine.calls))

    def test_explicit_reasoning_budget_and_zero_are_not_reduced(self):
        for budget in (900, 0):
            with self.subTest(budget=budget):
                svc, engine = self.make("a" * 600 + "</think>ok", ctx=2200)
                result = self.run_text(svc, 100, 1024, {"reasoning_budget_tokens": budget})
                self.assertEqual(result["finish"], "stop")
                self.assertGreaterEqual(result["reasoning_tokens"], 600)
                self.assertEqual(len(engine.calls), 1)

    def test_answer_room_uses_settled_text_cap(self):
        svc, engine = self.make(["a" * 1100, "</think>ok"], ctx=1708)
        result = self.run_text(svc, 300, 1700)
        self.assertEqual(engine.calls[0], (300, 1400))
        self.assertGreaterEqual(result["reasoning_tokens"], 888)  # 888 = 1400 - 512, then wrap
        self.assertEqual(len(engine.calls), 2)

    def test_continuation_rejects_a_mismatched_python_token_count(self):
        prompt = PromptText(self.tok, "abc", clip=True)
        prompt.set_counted(2, 10)
        with self.assertRaisesRegex(ValueError, "tokenizer counts differ"):
            prompt.ids()
        self.assertEqual(prompt.n, 2)

    def test_cancelled_before_admission_does_not_start_engine(self):
        svc, engine = self.make("ok")
        cancel = threading.Event()
        cancel.set()
        result = list(svc.run([1], True, None, 10, {}, cancel))
        self.assertEqual(result[-1][1]["finish"], "cancel")
        self.assertEqual(engine.calls, [])
        self.assertEqual(svc.status["queued"], 0)


if __name__ == "__main__":
    unittest.main()
