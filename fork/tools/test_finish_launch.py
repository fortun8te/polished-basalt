"""Offline regressions for candidate identity and truthful lifecycle receipts."""
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_identity as fi
import fork_launch as fl
from test_fork_identity import IdentityCase
from test_fork_launch import LauncherCase


class CompleteCandidateIdentity(IdentityCase):
    def test_companion_binary_change_gets_new_candidate_and_preserves_old_archive(self):
        first = self.record()
        companion = next(p for p in (self.root / 'bin').iterdir() if p.name != fi.ENGINE)
        original = companion.read_bytes()
        companion.write_bytes(b'changed companion')
        second = self.record()
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual((self.root / fi.CANDIDATES / first['id'] / 'bin' / companion.name).read_bytes(), original)

    def test_model_change_gets_new_candidate(self):
        first = self.record()
        (self.models / 'pack.basalt').write_bytes(b'PACK-BYTES')
        second = self.record()
        self.assertNotEqual(first['id'], second['id'])
        archived = fi.read_json(self.root / fi.CANDIDATES / first['id'] / fi.MANIFEST)
        self.assertEqual(archived['models'], first['models'])

    def test_repeated_record_does_not_rewrite_archive(self):
        first = self.record()
        archive = self.root / fi.CANDIDATES / first['id'] / 'source.tar.gz'
        os.utime(archive, ns=(1000000000,1000000000))
        self.record()
        self.assertEqual(archive.stat().st_mtime_ns,1000000000)


class LifecycleTruth(LauncherCase):
    def test_launch_rejects_weaker_model_checks_before_spawn(self):
        for mode in ('stat','skip'):
            self.assertEqual(self.start(models=mode),1)
        self.assertEqual(self.m.spawned,[])

    def test_missing_start_token_is_not_running(self):
        self.assertFalse(fl.running({'pid':123,'start_token':None},self.m.hooks()))

    def test_start_without_process_identity_is_not_success(self):
        hooks=self.m.hooks();hooks.start_token=lambda pid:None
        self.assertEqual(self.start(hooks=hooks),2)
        self.assertEqual(self.m.killed,[])

    def test_failed_stop_keeps_receipt_and_returns_failure(self):
        self.assertEqual(self.start(),0)
        hooks=self.m.hooks();hooks.terminate=lambda pid:None
        self.assertEqual(fl.stop(self.root,hooks=hooks,say=self.lines.append),1)
        self.assertTrue((self.root/fi.RUNS/fl.CURRENT).exists())

    def test_stop_without_recorded_process_identity_retains_receipt(self):
        record=self.root/fi.RUNS/fl.CURRENT
        fi.write_json(record,{'pid':123,'start_token':None,'arm':'a0','port':18892})
        self.assertEqual(fl.stop(self.root,hooks=self.m.hooks(),say=self.lines.append),1)
        self.assertTrue(record.exists())
        self.assertEqual(self.m.killed,[])

    def test_stop_exception_keeps_receipt(self):
        self.assertEqual(self.start(),0)
        hooks=self.m.hooks()
        def fail(pid):raise OSError('denied')
        hooks.terminate=fail
        self.assertEqual(fl.stop(self.root,hooks=hooks,say=self.lines.append),1)
        self.assertTrue((self.root/fi.RUNS/fl.CURRENT).exists())


class LocalReadiness(unittest.TestCase):
    def test_http_200_without_loaded_basalt_is_not_ready(self):
        class Response(io.BytesIO):status=200
        for payload in ({'loaded':False,'service':'basalt'},{'loaded':True,'service':'other'},{'status':'ok'},[]):
            with patch.object(fl.urllib.request,'urlopen',return_value=Response(json.dumps(payload).encode())):
                self.assertFalse(fl.http_ok(12345),payload)

    def test_loaded_basalt_is_ready(self):
        class Response(io.BytesIO):status=200
        with patch.object(fl.urllib.request,'urlopen',return_value=Response(b'{"loaded":true,"service":"basalt"}')):
            self.assertTrue(fl.http_ok(12345))

    def test_windows_wrappers_require_full_model_checks(self):
        windows=fi.default_root()/'fork/windows'
        self.assertNotIn('--reuse-model-hashes',(windows/'Build-Fork.cmd').read_text())
        start=(windows/'Start-ForkBasalt.ps1').read_text()
        self.assertNotIn('QuickModels',start)
        self.assertIn("'--models', 'full'",start)

    def test_fake_engine_flag_scan_is_not_real_engine_capability(self):
        # The historical smoke fake did not contain this flag. The native parser does.
        root=fi.default_root()
        sys.path.insert(0,str(root))
        from serve.config import engine_flags_of
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            fake=Path(temp)/'fake-engine';fake.write_text('print("fake")')
            self.assertNotIn('--conversation-cache-mib',engine_flags_of(str(fake)))
            fake.write_text(' --conversation-cache-mib N ')
            self.assertIn('--conversation-cache-mib',engine_flags_of(str(fake)))
        text=(root/'src/program/generate.cpp').read_text()
        self.assertIn('a == "--conversation-cache-mib"',text)
