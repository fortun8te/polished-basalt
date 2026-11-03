"""The wrapped-call closing-tag precheck gives the same events as the structural walk on every feed."""
import itertools
import random
import unittest
from unittest.mock import patch

from serve import frontend


class ReferenceParser(frontend.OutputParser):
    def _wrapped_call_end(self, final=False):
        return frontend.call_end(self.buf, final=final)


def events_without_ids(events):
    return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in events]


def call(value):
    return "<tool_call>\n<function=write>\n<parameter=text>\n" + value + "\n</parameter>\n</function>\n</tool_call>"


TOOLS = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]


class CallScanTests(unittest.TestCase):
    def test_chunk_boundaries_literal_tags_multiple_calls_and_unfinished(self):
        bodies = [call("ordinary"), call("literal </tool_call> text </function> here"),
                  call("literal </parameter> prose"), call("a") + call("b"), "intro\n" + call("a") + "\ntail",
                  call("unfinished")[:-5], call("incomplete </tool_call> prose")[:-30],
                  "<tool_call>\n<function=write>\n</function>\n</tool_call>",
                  call("quoted <function=foo>\n<parameter=x>\nunfinished"),
                  "<tool_call> named in prose, then " + call("real")]
        rng = random.Random(1030)
        for text, stream in itertools.product(bodies, (False, True)):
            # every fixed chunk width, plus a random partition; compared after EACH feed
            for width in range(1, 16):
                a = frontend.OutputParser(thinking=False, tools=TOOLS, stream_tools=stream)
                b = ReferenceParser(thinking=False, tools=TOOLS, stream_tools=stream)
                pos = 0
                while pos < len(text):
                    end = pos + (rng.randint(1, 20) if width == 15 else width)
                    delta = text[pos:end]
                    self.assertEqual(events_without_ids(a.feed(delta)), events_without_ids(b.feed(delta)))
                    pos = end
                self.assertEqual(events_without_ids(a.finish()), events_without_ids(b.finish()))

    def test_no_structural_walk_until_a_closer_and_reset_for_the_next_call(self):
        parser = frontend.OutputParser(thinking=False, tools=TOOLS)
        with patch.object(frontend, "call_end", wraps=frontend.call_end) as walk:
            for _ in range(2):
                parser.feed("<tool_call>\n<function=write>\n<parameter=text>\n")
                before = walk.call_count
                for _ in range(100):
                    parser.feed("abcdefgh")
                self.assertEqual(walk.call_count, before)
                for char in "\n</parameter>\n</function>\n</tool_call>":
                    parser.feed(char)
                self.assertGreater(walk.call_count, before)
                self.assertEqual(parser._call_end_checked, 0)


if __name__ == "__main__":
    unittest.main()
