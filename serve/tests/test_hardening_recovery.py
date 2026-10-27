"""Synthetic checks for stalled engine requests and slot ownership."""
import queue
import threading
import unittest
from unittest import mock

from serve.engine.errors import EngineSilent
from serve.engine.process import ProcessEngine


class Recovery(unittest.TestCase):
    def engine(self):
        e = object.__new__(ProcessEngine)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.ended = False
        e.silence_s = 1.0
        e._activity_at = 0.0
        e.lines = queue.Queue()
        e.slot_q = [queue.Queue()]
        e.slot_busy, e.slot_held = [True], [[1, 2]]
        e.slot_reports, e.slot_checkpoints = [(2, [1])], [[1]]
        e.slot_drafts = {}
        e.slot_cv = threading.Condition()
        e.wlock = threading.Lock()
        e._ctl_mode = 'solo'
        e._parse_done = lambda line: None
        return e

    def test_quiet_batch_control_ends_its_engine(self):
        e = self.engine()
        now = [0.0]
        def get(timeout):
            now[0] += timeout
            raise queue.Empty
        e.lines.get = get
        with mock.patch('serve.engine.process.time.monotonic', lambda: now[0]):
            with self.assertRaises(EngineSilent):
                next(e._control(threading.Event(), lambda _: None))
        e.proc.kill.assert_called_once()
        self.assertTrue(e.ended)

    def test_control_prompt_progress_extends_deadline(self):
        e = self.engine()
        e.lines.put('PP 50 100 1 1')
        e.lines.put('T 7')
        e.lines.put('DONE 1')
        tokens = []
        self.assertEqual(list(e._control(threading.Event(), tokens.append, prompt_len=100)), [None, False])
        self.assertEqual(tokens, [7])
        e.proc.kill.assert_not_called()

    def test_timeout_keeps_slot_reserved_and_kills_only_own_engine(self):
        e = self.engine()
        now = [0.0]
        def get(timeout):
            now[0] += timeout
            raise queue.Empty
        e.slot_q[0].get = get
        class InlineThread:
            def __init__(self, target, **kwargs): self.target = target
            def start(self): self.target()
        with mock.patch('serve.engine.process.threading.Thread', InlineThread), \
                mock.patch('serve.engine.process.time.monotonic', lambda: now[0]):
            e._release_slot_when_done(0, [1, 2])
        self.assertTrue(e.slot_busy[0])
        e.proc.kill.assert_called_once()
        self.assertTrue(e.ended)

    def test_confirmed_slot_is_released(self):
        e = self.engine()
        e.slot_q[0].put('BDONE 0 0 cancel 1')
        e._release_slot_when_done(0, [1, 2], wait_s=1)
        self.assertFalse(e.slot_busy[0])
        e.proc.kill.assert_not_called()

    def test_old_cleanup_does_not_kill_or_command_replacement(self):
        e = self.engine()
        old, e.proc = e.proc, mock.Mock()
        e._silent('old request timed out', proc=old)
        e._release_slot_when_done(0, proc=old)
        old.kill.assert_not_called()
        e.proc.kill.assert_not_called()
        e.proc.stdin.write.assert_not_called()
        self.assertFalse(e.ended)
        self.assertTrue(e.slot_busy[0])


if __name__ == '__main__':
    unittest.main()
