"""The engine's stdout is framed on complete lines: a cut line is never a reply, a reader that fails ends the engine and
every request waiting on it, and a malformed numeric line is a protocol fault.  Synthetic pipes and fake children only."""
from __future__ import annotations

import io
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine, frame_fault

FAKE = r'''
import os, sys, time
mode = sys.argv[sys.argv.index('--mode') + 1]
out = sys.stdout.buffer
def put(raw, pause=0.0):
    out.write(raw)
    out.flush()
    if pause:
        time.sleep(pause)
put(b'READY 4096 stop\n')
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if not line.startswith('GEN'):
        continue
    if mode == 'torn_done':        # the child dies inside the final line
        put(b'T 42\nDONE 1 1 1 1 stop')
        os._exit(0)
    elif mode == 'torn_token':     # "T 4" is a prefix of "T 42"
        put(b'T 7\nT 4')
        os._exit(0)
    elif mode == 'fragments':      # one line arrives in pieces over time
        put(b'T 4', 0.05)
        put(b'2\nDO', 0.05)
        put(b'NE 1 1 0.5 1.5 st', 0.05)
        put(b'op\n')
    elif mode == 'bad_utf8':       # the child stays alive but its output can no longer be read
        put(b'T 7\n')
        put(b'bad\xff\n')
        time.sleep(60)
    elif mode == 'malformed_done':
        put(b'T 7\nDONE 1 1 1 1\n')
        time.sleep(60)
'''


def bare_engine(stdout, slots: int = 2) -> ProcessEngine:
    """A ProcessEngine around a synthetic stdout: the pump's own state only."""
    e = object.__new__(ProcessEngine)
    e.proc = mock.Mock()
    e.proc.stdout = stdout
    e.proc.poll.return_value = None
    e.lines = queue.Queue()
    e.slot_q = [queue.Queue() for _ in range(slots)]
    e.slot_reports = [None] * slots
    e.slot_cv = threading.Condition()
    e.log = None
    e.ended = False
    e.last_err = None
    return e


class FrameFault(unittest.TestCase):
    def test_the_lines_the_native_engine_prints_pass(self):
        for line in ("T 42\n", "T -1\n", "BT 0\n", "BT 3 11 12 13\n", "PP 5 10 1 2\n", "READY 4096 stop\n", "ERR x\n",
                     "DONE 1 1 1 1 stop\n", "DONE 12 100 53.2 210.5 length\n",
                     "DONE 12 100 53.2 210.5 stop 4 9 0 3 7 11 2 5.5 100 6\n",
                     "BDONE 0 12 stop 33.1\n", "BDONE 0 12 stop 33.1 4 9\n", "BADM 0 1\n", "BADM 3 2\n"):
            self.assertIsNone(frame_fault(line), line)

    def test_a_numeric_line_that_is_not_one_is_a_fault(self):
        for line in ("T\n", "T x\n", "T 4 2\n", "BT x 1\n", "BT 0 1.5\n", "DONE\n", "DONE 1 1 1 1\n", "DONE x 1 1 1 stop\n",
                     "DONE 1 1 nan 1 stop\n", "DONE 1 1 1 1 stop 4 x\n", "DONE 1 1 1 1 stop 1.5 2 3\n", "DONE 1 1 1 1 stop 4\n", "BDONE 0 12 stop\n",
                     "BDONE 0 x stop 1\n", "BDONE 0 12 stop 1 4\n", "BADM 0\n", "BADM 0 x\n", "BADM 0 1 2\n"):
            self.assertIsNotNone(frame_fault(line), line)


class Pump(unittest.TestCase):
    def test_a_final_fragment_is_dropped_and_ends_the_engine(self):
        e = bare_engine(io.StringIO("T 42\nDONE 1 1 1 1 stop"))
        e._pump()
        self.assertEqual(e.lines.get_nowait(), "T 42\n")
        self.assertIsNone(e.lines.get_nowait())            # the end, never the cut DONE
        self.assertTrue(e.lines.empty())
        self.assertTrue(e.ended)
        self.assertIn("ended inside a line", e.death_note())
        e.proc.kill.assert_called_once()

    def test_a_complete_stream_is_not_a_failure(self):
        e = bare_engine(io.StringIO("T 42\nDONE 1 1 1 1 stop\n"))
        e._pump()
        self.assertEqual([e.lines.get_nowait() for _ in range(3)], ["T 42\n", "DONE 1 1 1 1 stop\n", None])
        self.assertTrue(e.ended)
        self.assertIsNone(e.pump_note)
        e.proc.kill.assert_not_called()

    def test_a_CRLF_line_is_complete(self):
        e = bare_engine(io.StringIO("T 42\r\n"))
        e._pump()
        self.assertEqual(e.lines.get_nowait(), "T 42\n")
        self.assertIsNone(e.pump_note)

    def test_a_malformed_line_is_not_forwarded_and_ends_the_engine(self):
        e = bare_engine(io.StringIO("T 7\nDONE 1 1\nT 8\n"))
        e._pump()
        self.assertEqual(e.lines.get_nowait(), "T 7\n")
        self.assertIsNone(e.lines.get_nowait())
        self.assertTrue(e.lines.empty())
        self.assertIn("malformed DONE", e.death_note())
        e.proc.kill.assert_called_once()

    def test_a_malformed_slot_line_never_reaches_its_slot(self):
        e = bare_engine(io.StringIO("BT 1 5\nBDONE 1 5 stop\nBT 0 6\n"))
        e._pump()
        self.assertEqual(e.slot_q[1].get_nowait(), "BT 1 5\n")
        self.assertIsNone(e.slot_q[1].get_nowait())
        self.assertTrue(e.slot_q[1].empty())
        self.assertIsNone(e.slot_q[0].get_nowait())

    def test_undecodable_bytes_end_every_waiting_request(self):
        e = bare_engine(io.TextIOWrapper(io.BytesIO(b"T 7\nbad\xff\n"), encoding="utf-8", newline="\n"), slots=3)
        got, woke = [], []

        def wait(q):
            got.append(q.get(timeout=10))

        def wait_cv():
            t0 = time.monotonic()
            with e.slot_cv:
                e.slot_cv.wait(timeout=10)
            woke.append(time.monotonic() - t0)

        waiters = [threading.Thread(target=wait, args=(q,)) for q in (e.lines, *e.slot_q)]
        waiters.append(threading.Thread(target=wait_cv))
        for t in waiters:
            t.start()
        time.sleep(0.1)
        e._pump()
        for t in waiters:
            t.join(5)
        self.assertFalse(any(t.is_alive() for t in waiters))
        self.assertEqual(got, [None] * 4)                  # the end reached the control reader and every slot
        self.assertLess(woke[0], 5)
        self.assertTrue(e.ended)
        self.assertIn("UnicodeDecodeError", e.death_note())
        e.proc.kill.assert_called_once()

    def test_any_exception_in_the_reader_is_published_as_the_end(self):
        class Boom:
            def __iter__(self):
                yield "T 1\n"
                raise RuntimeError("reader broke")
        e = bare_engine(Boom(), slots=1)
        e._pump()
        self.assertEqual(e.lines.get_nowait(), "T 1\n")
        self.assertIsNone(e.lines.get_nowait())
        self.assertIsNone(e.slot_q[0].get_nowait())
        self.assertTrue(e.ended)
        self.assertIn("reader broke", e.death_note())

    def test_a_replaced_engines_pump_does_not_mark_its_successor(self):
        e = bare_engine(None)
        old, successor = e.proc, mock.Mock()

        class Stream:
            def __iter__(self):
                yield "T 1\n"
                e.proc = successor                           # restart(): the successor is current before this pump ends
                yield "DONE 1"                               # cut

        old.stdout = Stream()
        e._pump()
        self.assertFalse(e.ended)
        self.assertIsNone(e.pump_note)
        old.kill.assert_called_once()                        # its own child is still ended
        successor.kill.assert_not_called()


class ReadyPump(unittest.TestCase):
    def test_a_cut_ready_line_is_not_ready(self):
        proc = mock.Mock()
        proc.stdout = iter(["INFO batch_slots=2\n", "READY 40"])
        out: queue.Queue = queue.Queue()
        ProcessEngine._ready_pump(proc, out)
        self.assertEqual(out.get_nowait(), "INFO batch_slots=2\n")
        self.assertIsNone(out.get_nowait())
        self.assertTrue(out.empty())

    def test_a_whole_ready_line_is(self):
        proc = mock.Mock()
        proc.stdout = iter(["READY 4096 stop\n", "T 1\n"])
        out: queue.Queue = queue.Queue()
        ProcessEngine._ready_pump(proc, out)
        self.assertEqual(out.get_nowait(), "READY 4096 stop\n")
        self.assertTrue(out.empty())


class FakeChild(unittest.TestCase):
    """The real pump, pipes and generate() against a child that cuts, fragments or corrupts its output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        py = root / "native.py"
        if os.name == "nt":                                  # no shebangs on Windows: a .cmd runs the same fake engine
            py.write_text(FAKE)
            self.script = root / "native.cmd"
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script = py
            py.write_text("#!" + sys.executable + "\n" + FAKE)
            py.chmod(0o700)

    def engine(self, mode: str) -> ProcessEngine:
        e = ProcessEngine(str(self.script), ["--mode", mode])
        self.addCleanup(e.close)
        return e

    def run_request(self, e: ProcessEngine) -> list:
        return [t for t in e.generate([1], 8, {}, threading.Event()) if t is not None]

    def test_a_child_that_dies_inside_DONE_fails_the_request(self):
        e = self.engine("torn_done")
        with self.assertRaises(EngineDied):
            self.run_request(e)
        self.assertFalse(e.alive())
        self.assertIn("ended inside a line", e.death_note())

    def test_a_cut_token_is_not_a_smaller_token(self):
        e = self.engine("torn_token")
        seen = []
        with self.assertRaises(EngineDied):
            for t in e.generate([1], 8, {}, threading.Event()):
                if t is not None:
                    seen.append(t)
        self.assertEqual(seen, [7])
        self.assertFalse(e.alive())

    def test_a_line_that_arrives_in_pieces_is_one_line(self):
        e = self.engine("fragments")
        self.assertEqual(self.run_request(e), [42])
        self.assertEqual(e.last["finish"], "stop")
        self.assertTrue(e.alive())

    def test_an_unreadable_stream_fails_the_request_and_ends_the_live_child(self):
        e = self.engine("bad_utf8")
        proc = e.proc
        with self.assertRaises(EngineDied):
            self.run_request(e)
        self.assertFalse(e.alive())
        self.assertIsNotNone(proc.wait(timeout=10))          # the child did not stay behind holding its memory
        self.assertIn("could not be read", e.death_note())

    def test_a_malformed_DONE_fails_the_request_instead_of_finishing_it(self):
        e = self.engine("malformed_done")
        proc = e.proc
        with self.assertRaises(EngineDied):
            self.run_request(e)
        self.assertIsNotNone(proc.wait(timeout=10))
        self.assertIn("malformed DONE", e.death_note())


if __name__ == "__main__":
    unittest.main()
