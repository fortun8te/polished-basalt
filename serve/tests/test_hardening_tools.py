"""Tool results retain their identity and incomplete calls never become executable."""
import json
import unittest

from serve.frontend import OutputParser, openai_to_messages, anthropic_to_messages
from serve.api.responses import Assembler, ResponsesError, input_messages, request_tools

TOOL = {"name": "read", "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}


def history():
    return ([{"id": "a", "function": {"name": "read", "arguments": '{"input":"A"}'}},
             {"id": "b", "function": {"name": "read", "arguments": '{"input":"B"}'}}],
            [{"role": "tool", "tool_call_id": "b", "content": "B result"},
             {"role": "tool", "tool_call_id": "a", "content": "A result"}])


class Identity(unittest.TestCase):
    def requests(self):
        calls, results = history()
        oa = {"messages": [{"role": "assistant", "content": "", "tool_calls": calls}, *results]}
        an = {"messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": c["id"], "name": "read", "input": json.loads(c["function"]["arguments"])}
            for c in calls]}, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["tool_call_id"], "content": r["content"]} for r in results]}]}
        rs = {"input": [{"type": "function_call", "call_id": c["id"], "name": "read",
                         "arguments": c["function"]["arguments"]} for c in calls] + [
                            {"type": "function_call_output", "call_id": r["tool_call_id"], "output": r["content"]}
                            for r in results]}
        return oa, an, rs

    def test_equivalent_histories(self):
        oa, an, rs = self.requests()
        a, b, c = openai_to_messages(oa)[0], anthropic_to_messages(an)[0], input_messages(rs)
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual([m["content"] for m in a[1:]], ["A result", "B result"])

    def test_bad_results_are_rejected(self):
        for case in ("unknown", "duplicate", "missing", "mixed", "duplicate_call"):
            oa, an, rs = self.requests()
            if case == "unknown":
                oa["messages"][1]["tool_call_id"] = "x"
                an["messages"][1]["content"][0]["tool_use_id"] = "x"
                rs["input"][2]["call_id"] = "x"
            elif case == "duplicate":
                oa["messages"][1]["tool_call_id"] = "a"
                an["messages"][1]["content"][0]["tool_use_id"] = "a"
                rs["input"][2]["call_id"] = "a"
            elif case == "missing":
                oa["messages"].pop()
                an["messages"][1]["content"].pop()
                rs["input"].pop()
            elif case == "mixed":
                oa["messages"][1].pop("tool_call_id")
                an["messages"][1]["content"][0].pop("tool_use_id")
                rs["input"][2].pop("call_id")
            else:
                oa["messages"][0]["tool_calls"][1]["id"] = "a"
                an["messages"][0]["content"][1]["id"] = "a"
                rs["input"][1]["call_id"] = "a"
            for fn, req in ((openai_to_messages, oa), (anthropic_to_messages, an), (input_messages, rs)):
                with self.subTest(case=case, api=fn.__name__), self.assertRaises(ValueError):
                    fn(req)

    def test_parallel_id_free_history_is_ambiguous(self):
        oa, an, rs = self.requests()
        for c in oa['messages'][0]['tool_calls']:
            c.pop('id')
        for r in oa['messages'][1:]:
            r.pop('tool_call_id')
        for c in an['messages'][0]['content']:
            c.pop('id')
        for r in an['messages'][1]['content']:
            r.pop('tool_use_id')
        for item in rs['input']:
            item.pop('call_id')
        for fn, req in ((openai_to_messages, oa), (anthropic_to_messages, an), (input_messages, rs)):
            with self.subTest(api=fn.__name__), self.assertRaises(ValueError):
                fn(req)

    def test_duplicate_flattened_tools_rejected(self):
        with self.assertRaises(ResponsesError):
            request_tools({"tools": [{"type": "function", "name": "x.y"},
                                    {"type": "namespace", "name": "x", "tools": [{"name": "y"}]}]})


class Calls(unittest.TestCase):
    def run_parser(self, text, step=1):
        p = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        out = []
        for i in range(0, len(text), step):
            out += p.feed(text[i:i + step])
        return out + p.finish()

    def test_complete_and_partial_calls(self):
        body = '<tool_call><function=read><parameter=input>replace A with B</parameter></function>'
        for step in (1, 9, 10000):
            events = self.run_parser(body + '</tool_call>', step)
            self.assertEqual([e.call.arguments for e in events if e.kind == 'tool_call'], [{"input": "replace A with B"}])
            self.assertEqual(json.loads(''.join(e.text for e in events if e.kind == 'tool_args')), {"input": "replace A with B"})
            partial = self.run_parser(body, step)
            self.assertFalse([e for e in partial if e.kind == 'tool_call'])
            with self.assertRaises(ValueError):
                json.loads(''.join(e.text for e in partial if e.kind == 'tool_args'))
            a = Assembler({}, 'model', 0, {'read': (None, 'read', 'custom')}, False)
            for event in partial:
                a.feed(event)
            finish_events = a.finish({'finish': 'stop'})
            self.assertFalse(any(e['type'] == 'response.custom_tool_call_input.done' for e in finish_events))
            self.assertEqual(a.response['status'], 'incomplete')
            self.assertEqual(a.response['output'][0]['status'], 'incomplete')

    def test_complete_custom_input_preserved(self):
        text = '<tool_call><function=read><parameter=input>replace A with B</parameter></function></tool_call>'
        a = Assembler({}, 'model', 0, {'read': (None, 'read', 'custom')}, False)
        for event in self.run_parser(text):
            a.feed(event)
        a.finish({'finish': 'stop'})
        self.assertEqual(a.response['status'], 'completed')
        self.assertEqual(a.response['output'][0]['input'], 'replace A with B')

    def test_unknown_and_malformed_never_finish(self):
        for text in ('<tool_call><function=other></function></tool_call>',
                     '<tool_call><function=read>ignored<parameter=input>x</parameter></function></tool_call>',
                     '<tool_call><function=read><parameter=input>x</parameter><parameter=input>y</parameter></function></tool_call>'):
            for step in (1, 10000):
                events = self.run_parser(text, step)
                self.assertFalse([e for e in events if e.kind == 'tool_call'], text)
                if any(e.kind == 'tool_start' for e in events):
                    with self.assertRaises(ValueError):
                        json.loads(''.join(e.text for e in events if e.kind == 'tool_args'))

    def test_incremental_stream_remains(self):
        p = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        events = p.feed('<tool_call><function=read><parameter=input>partial')
        self.assertTrue(any(e.kind == 'tool_start' for e in events))
        self.assertTrue(any(e.kind == 'tool_args' and 'partial' in e.text for e in events))
        self.assertFalse(any(e.kind == 'tool_call' for e in events))


if __name__ == '__main__':
    unittest.main()
