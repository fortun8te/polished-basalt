"""Control admission cancellation and parallel-to-single restart regressions; no native engine."""
import threading
import unittest
from unittest import mock

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine


class ControlAdmission(unittest.TestCase):
    def engine(self):
        e = object.__new__(ProcessEngine)
        e.slot_cv = threading.Condition()
        e.waiting, e.wait_lens, e.ctl_epoch = 0, [], 0
        e.ctl = threading.Lock()
        e.batch = 2
        e.alive = lambda: True
        return e

    def take(self, e, cancel):
        gen = e._take_control(cancel, 10)
        with self.assertRaises(StopIteration) as done:
            next(gen)
        return done.exception.value

    def clean(self, e):
        self.assertEqual((e.waiting, e.wait_lens), (0, []))
        self.assertFalse(e.ctl.locked())

    def test_cancelled_before_admission_never_takes_control(self):
        e = self.engine()
        cancel = threading.Event()
        cancel.set()
        self.assertFalse(self.take(e, cancel))
        self.clean(e)
        self.assertEqual(e.ctl_epoch, 0)
        self.assertTrue(self.take(e, threading.Event()))
        e.ctl.release()
        self.clean(e)

    def test_cancelled_during_acquire_releases_control(self):
        e = self.engine()
        cancel = threading.Event()
        lock = e.ctl
        def acquire(**kwargs):
            got = lock.acquire(**kwargs)
            cancel.set()
            return got
        e.ctl = mock.Mock(wraps=lock)
        e.ctl.acquire.side_effect = acquire
        self.assertFalse(self.take(e, cancel))
        self.clean(e)
        self.assertEqual(e.ctl_epoch, 0)

    def test_restart_to_single_while_acquiring_rejects_old_waiter(self):
        e = self.engine()
        lock = e.ctl
        def acquire(**kwargs):
            got = lock.acquire(**kwargs)
            e.batch = 0
            return got
        e.ctl = mock.Mock(wraps=lock)
        e.ctl.acquire.side_effect = acquire
        with self.assertRaises(EngineDied):
            self.take(e, threading.Event())
        self.clean(e)

    def test_control_entry_without_parallel_slots_is_rejected(self):
        e = self.engine()
        e.batch = 0
        with self.assertRaises(EngineDied):
            self.take(e, threading.Event())
        self.clean(e)

    def test_dead_engine_still_reports_failure(self):
        e = self.engine()
        e.alive = lambda: False
        with self.assertRaises(EngineDied):
            self.take(e, threading.Event())
        self.clean(e)

    def test_restart_to_single_while_waiting_rejects_without_acquisition(self):
        e = self.engine()
        lock = e.ctl
        def acquire(**kwargs):
            e.batch = 0
            return False
        e.ctl = mock.Mock(wraps=lock)
        e.ctl.acquire.side_effect = acquire
        with self.assertRaises(EngineDied):
            self.take(e, threading.Event())
        self.clean(e)

    def test_parallel_restart_can_still_admit_waiter(self):
        e = self.engine()
        lock = e.ctl
        def acquire(**kwargs):
            e.batch = 4
            return lock.acquire(**kwargs)
        e.ctl = mock.Mock(wraps=lock)
        e.ctl.acquire.side_effect = acquire
        self.assertTrue(self.take(e, threading.Event()))
        e.ctl.release()
        self.clean(e)


class CancelledDispatch(unittest.TestCase):
    from serve.tests.test_pump_framing import FakeChild as _FakeChild
    setUp = _FakeChild.setUp
    engine = _FakeChild.engine

    def test_cancelled_batch_request_sends_no_generation_command(self):
        e = self.engine("fragments")
        e.batch = 2
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(e, "_send", wraps=e._send) as send:
            self.assertEqual(list(e.generate([1], 8, {}, cancel)), [])
        send.assert_not_called()
        self.assertTrue(e.alive())
        self.assertFalse(e.ctl.locked())
        self.assertEqual((e.waiting, e.wait_lens), (0, []))

    def test_stale_batch_entry_after_restart_sends_no_command(self):
        e = self.engine("fragments")
        self.assertEqual(e.batch, 0)
        with mock.patch.object(e, "_send", wraps=e._send) as send:
            with self.assertRaises(EngineDied):
                list(e.generate_batched([1], 8, {}, threading.Event()))
        send.assert_not_called()
        self.assertTrue(e.alive())
        self.assertFalse(e.ctl.locked())
