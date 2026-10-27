"""Quality-gate input and completion checks with synthetic rows and an engine fake."""
import contextlib
import io
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ppl_gate


def row(chunk=0, pos=2, target=3, nll='1.0'):
    return f'{chunk}\t{pos}\t{target}\t{nll}\t1.0\t0.0\t1\t3\t3\t0.0\n'


class FakeEngine:
    def __init__(self, command, *, stdin, stdout, stderr, text, bufsize, mode='ok'):
        self.rows = Path(command[command.index('--ppl-out') + 1])
        self.mode = mode
        self.stdin = self
        self.stdout = self
        self.pending = ['READY 4096 stop\n']
        self.returncode = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.pending:
            return self.pending.pop(0)
        raise StopIteration

    def write(self, message):
        if message.startswith('GEN '):
            if self.mode != 'missing_rows':
                self.rows.write_text(row(target=2 if self.mode == 'wrong_target' else 3))
            count = 0 if self.mode == 'short_done' else 1
            finish = 'cancel' if self.mode == 'cancel' else 'length'
            if self.mode == 'err':
                self.pending.append('ERR failed quality scoring\n')
            elif self.mode == 'bad_done':
                self.pending.append('DONE incomplete\n')
            elif self.mode != 'eof':
                self.pending.append(f'DONE {count} 3 1.0 2.0 {finish} 0 0 0 0 0\n')
        elif message == 'QUIT\n':
            self.returncode = 7 if self.mode == 'nonzero' else 0

    def flush(self):
        pass

    def close(self):
        pass

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, text):
        p = self.root / name
        p.write_text(text)
        return str(p)

    def args(self):
        base = self.root / 'base.kld'
        # ctx4/vocab4/chunks1; one row, 4 header uint16 + 4 log-probs.
        base.write_bytes(b'_logits_' + struct.pack('<Iii4i', 4, 4, 1, 0, 1, 2, 3) + bytes(16))
        return SimpleNamespace(base=str(base), model='synthetic.basalt', engine_bin='fake',
                               out=str(self.root / 'out'), chunks=1, t=3, progress=0, engine_args=[])

    def run_fake(self, args, mode='ok'):
        def launch(*a, **kw):
            return FakeEngine(*a, **kw, mode=mode)
        with patch.object(ppl_gate, 'gpu_count', return_value=0), patch.object(ppl_gate.subprocess, 'Popen', side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
            return ppl_gate.run(args)

    def test_valid_report_and_diff(self):
        a = self.put('a.tsv', row())
        self.assertIn('PPL 2.7183', ppl_gate.summary(a))
        self.assertIn('identical 1 (100.00%)', ppl_gate.diff(a, a))

    def test_report_rejects_corruption(self):
        for bad in ('', 'broken\n', row() + 'broken\n', row(nll='nan'), row(nll='inf'), row() + row()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ppl_gate.summary(self.put('bad.tsv', bad))

    def test_diff_rejects_different_rows_and_targets(self):
        a = self.put('a.tsv', row())
        for content in (row() + row(pos=3), row(target=2), row() + row()):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    ppl_gate.diff(a, self.put('b.tsv', content))

    def test_valid_run_writes_report(self):
        a = self.args()
        self.assertEqual(self.run_fake(a), 0)
        self.assertIn('rows 1', (Path(a.out) / 'report.txt').read_text())

    def test_run_rejects_incomplete_and_failed_engine(self):
        for mode in ('missing_rows', 'wrong_target', 'short_done', 'cancel', 'nonzero', 'err', 'bad_done', 'eof'):
            with self.subTest(mode=mode):
                a = self.args()
                out = Path(a.out)
                out.mkdir(exist_ok=True)
                (out / 'report.txt').write_text('old successful report')
                self.assertNotEqual(self.run_fake(a, mode), 0)
                self.assertFalse((out / 'report.txt').exists())

    def test_unavailable_chunks_rejected_before_launch(self):
        a = self.args()
        a.chunks = 2
        with patch.object(ppl_gate.subprocess, 'Popen') as launch, patch.object(ppl_gate, 'gpu_count', return_value=0):
            with self.assertRaises(ValueError):
                ppl_gate.run(a)
            launch.assert_not_called()

    def test_truncated_base_rejected_before_launch(self):
        a = self.args()
        Path(a.base).write_bytes(Path(a.base).read_bytes()[:-1])
        with patch.object(ppl_gate.subprocess, 'Popen') as launch, patch.object(ppl_gate, 'gpu_count', return_value=0):
            with self.assertRaises(ValueError):
                ppl_gate.run(a)
            launch.assert_not_called()

    def test_report_checks_every_metric_column(self):
        for column in (3, 4, 5, 9):
            fields = row().strip().split('\t')
            fields[column] = 'nan'
            with self.subTest(column=column), self.assertRaises(ValueError):
                ppl_gate.summary(self.put('bad.tsv', '\t'.join(fields) + '\n'))

    def test_cli_invalid_diff_exits_nonzero(self):
        a = self.put('a.tsv', row())
        b = self.put('b.tsv', row() + row(pos=3))
        result = subprocess.run([sys.executable, ppl_gate.__file__, '--diff', a, b],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('row sets differ', result.stderr)
        self.assertNotIn('100.00%', result.stdout)

    def test_cli_valid_report_exits_zero(self):
        result = subprocess.run([sys.executable, ppl_gate.__file__, '--report', self.put('good.tsv', row())],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PPL 2.7183', result.stdout)


if __name__ == '__main__':
    unittest.main()
