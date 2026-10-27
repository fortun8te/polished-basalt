"""The stderr pump owns its read pipe and closes it on every exit path."""
import io
import subprocess
import sys
import unittest

from serve.engine.process import pump_lines


class StderrPipeCleanup(unittest.TestCase):
    def test_eof_closes_the_owned_stream(self):
        stream = io.StringIO("basalt: ready\n")
        seen = []
        pump_lines(stream, seen.append)
        self.assertTrue(stream.closed)
        self.assertEqual(len(seen), 1)

    def test_sink_failure_still_closes_the_owned_stream(self):
        stream = io.StringIO("basalt: ready\n")

        def fail(_line):
            raise RuntimeError("sink failed")

        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            pump_lines(stream, fail)
        self.assertTrue(stream.closed)

    def test_a_real_child_stderr_pipe_is_closed_before_its_process_is_released(self):
        with subprocess.Popen([sys.executable, "-c", "import sys; print('basalt: ready', file=sys.stderr)"],
                              stderr=subprocess.PIPE, text=True) as child:
            seen = []
            pump_lines(child.stderr, seen.append)
            child.wait(timeout=10)
            self.assertTrue(child.stderr.closed)
            self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
