"""The identity manifest: what is recorded at build time, what start-up verifies, and the rollback record.

    python -m pytest fork/tools -q
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_identity as fi  # noqa: E402

GIT = shutil.which("git")
QUIET = lambda *_: None  # noqa: E731


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], check=True, capture_output=True)


def make_tree(root: Path, models: Path, engine=b"engine-v1", arms=("a0", "a1")) -> None:
    """A miniature of the repository layout: sources, profiles, a built bin/ and model files, committed."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("/bin/\nbuild*/\n")
    (root / "CMakeLists.txt").write_text("project(x)\n")
    for rel, text in (("src/a.cpp", "int main() {}\n"), ("serve/server.py", "print(1)\n"), ("tools/t.py", "x = 1\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    models.mkdir(parents=True, exist_ok=True)
    (models / "pack.basalt").write_bytes(b"pack-bytes")
    (models / "ple.gguf").write_bytes(b"ple-bytes")
    (root / "fork/profiles").mkdir(parents=True, exist_ok=True)
    for i, arm in enumerate(arms):
        (root / f"fork/profiles/{arm}.json").write_text(
            '{"server": {"model": "%s", "port": %d}, "engine": {"ple-gguf": "%s", "spec": 7}}'
            % ((models / "pack.basalt").as_posix(), 18892 + i, (models / "ple.gguf").as_posix()))
    (root / "bin").mkdir(exist_ok=True)
    (root / "bin" / fi.ENGINE).write_bytes(engine)
    (root / "bin" / ("basalt-bench" + (".exe" if os.name == "nt" else ""))).write_bytes(b"bench-v1")
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "tree")


def commit_all(root, message="change"):
    git(root, "add", "-A")
    git(root, "commit", "-qm", message)


@unittest.skipUnless(GIT, "git is needed")
class IdentityCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        self.root, self.models = base / "tree", base / "models"
        make_tree(self.root, self.models)

    def record(self, **kw):
        return fi.record(self.root, say=QUIET, **kw)

    def verify(self, **kw):
        return fi.verify(self.root, **kw)

    def assertMismatch(self, text, **kw):
        problems = self.verify(**kw)
        self.assertTrue(any(text in p for p in problems), f"{text!r} not in {problems}")


class Record(IdentityCase):
    def test_a_recorded_candidate_verifies(self):
        m = self.record()
        self.assertEqual(self.verify(), [])
        self.assertEqual(m["source"]["commit"], subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
        self.assertFalse(m["source"]["dirty"])
        self.assertEqual(m["engine"], f"bin/{fi.ENGINE}")
        self.assertEqual(set(m["profiles"]), {"a0", "a1"})
        self.assertEqual(len(m["models"]), 2)                       # both profiles read the same two files
        self.assertEqual(m["binaries"][f"bin/{fi.ENGINE}"]["sha256"], fi.sha256_file(self.root / "bin" / fi.ENGINE))

    def test_recording_leaves_the_tree_clean(self):
        self.record()
        self.assertFalse(fi.git_state(self.root)["dirty"])          # manifest and archive are not source changes
        self.record()                                               # and recording again is allowed
        self.assertEqual(self.verify(), [])

    def test_no_engine_binary_is_refused_with_the_right_directory_named(self):
        (self.root / "bin" / fi.ENGINE).unlink()
        with self.assertRaisesRegex(fi.IdentityError, "source tree's bin/, not to the build directory"):
            self.record()

    def test_a_dirty_tree_is_refused_unless_allowed(self):
        (self.root / "serve/server.py").write_text("print(2)\n")
        with self.assertRaisesRegex(fi.IdentityError, "uncommitted"):
            self.record()
        m = self.record(allow_dirty=True)
        self.assertTrue(m["source"]["dirty"])
        self.assertEqual(self.verify(), [])

    def test_expected_model_hashes_are_enforced(self):
        good = fi.sha256_file(self.models / "pack.basalt")
        self.record(expect={str(self.models / "pack.basalt"): good})
        with self.assertRaisesRegex(fi.IdentityError, "is not the expected"):
            self.record(expect={str(self.models / "pack.basalt"): "0" * 64})
        with self.assertRaisesRegex(fi.IdentityError, "no profile reads"):
            self.record(expect={str(self.models / "other.bin"): good})

    def test_a_missing_model_is_refused(self):
        (self.models / "ple.gguf").unlink()
        with self.assertRaisesRegex(fi.IdentityError, "model file not found"):
            self.record()

    def test_model_hashes_are_reused_only_when_asked_and_unchanged(self):
        self.record()
        self.assertTrue(all(v["hash"] == "full" for v in self.record()["models"].values()))
        again = self.record(reuse_model_hashes=True)
        self.assertTrue(all(v["hash"].startswith("reused") for v in again["models"].values()))
        (self.models / "pack.basalt").write_bytes(b"other-bytes")   # changed: hashed again
        third = self.record(reuse_model_hashes=True)
        pack = third["models"][fi.norm_key(self.models / "pack.basalt")]
        self.assertEqual(pack["hash"], "full")
        self.assertEqual(pack["sha256"], fi.sha256_file(self.models / "pack.basalt"))

    def test_relative_model_paths_follow_the_profile_directory(self):
        (self.root / "fork/profiles/rel.json").write_text('{"server": {"model": "../../../models/pack.basalt"}}')
        commit_all(self.root)
        m = self.record()
        self.assertIn(fi.norm_key(self.models / "pack.basalt"), m["models"])


class Verify(IdentityCase):
    def setUp(self):
        super().setUp()
        self.record()

    def test_a_changed_engine_binary_is_a_mismatch(self):
        (self.root / "bin" / fi.ENGINE).write_bytes(b"engine-v2")
        self.assertMismatch(f"binary bin/{fi.ENGINE} is")

    def test_a_missing_engine_binary_is_a_mismatch(self):
        (self.root / "bin" / fi.ENGINE).unlink()
        self.assertMismatch(f"binary bin/{fi.ENGINE} is missing")

    def test_a_file_in_bin_that_was_not_recorded_is_a_mismatch(self):
        (self.root / "bin" / "extra.dll").write_bytes(b"x")
        self.assertMismatch("bin/extra.dll is not in the manifest")

    def test_edited_source_is_a_mismatch(self):
        (self.root / "serve/server.py").write_text("print('changed')\n")
        self.assertMismatch("source tree is")
        self.assertMismatch("dirty flag")

    def test_a_new_commit_is_a_mismatch(self):
        (self.root / "src/a.cpp").write_text("int main() { return 1; }\n")
        commit_all(self.root)
        self.assertMismatch("source commit is")
        self.assertMismatch("source tree is")

    def test_an_edited_profile_is_a_mismatch(self):
        p = self.root / "fork/profiles/a0.json"
        p.write_text(p.read_text().replace('"spec": 7', '"spec": 5'))
        self.assertMismatch("profile fork/profiles/a0.json changed", arm="a0")

    def test_an_unknown_arm_is_a_mismatch(self):
        self.assertMismatch("profile nope is not in the manifest", arm="nope")

    def test_a_substituted_model_of_the_same_size_fails_the_full_check_only(self):
        pack = self.models / "pack.basalt"
        pack.write_bytes(b"PACK-BYTES")                              # same length, other content
        os.utime(pack, ns=(1, 1))
        self.assertMismatch("does not match the recorded sha256", arm="a0")
        self.assertMismatch("was modified after it was hashed", arm="a0", models="stat")
        self.assertEqual(self.verify(arm="a0", models="skip"), [])

    def test_a_model_of_another_size_fails_every_mode(self):
        (self.models / "ple.gguf").write_bytes(b"short")
        self.assertMismatch("bytes, the manifest records", arm="a1", models="stat")
        self.assertMismatch("bytes, the manifest records", arm="a1", models="full")

    def test_a_missing_model_is_a_mismatch(self):
        (self.models / "pack.basalt").unlink()
        self.assertMismatch("model", arm="a0", models="stat")

    def test_no_manifest_is_a_mismatch(self):
        (self.root / fi.MANIFEST).unlink()
        self.assertMismatch("fork-identity.json is missing")

    def test_a_damaged_manifest_is_a_mismatch(self):
        (self.root / fi.MANIFEST).write_text("{not json")
        self.assertMismatch("unreadable")


class SourceCopies(IdentityCase):
    """The PC gets a copy of the tree without .git; the commit travels in fork-source.json."""

    def copy_without_git(self, name="pc"):
        dst = self.root.parent / name
        shutil.copytree(self.root, dst, ignore=shutil.ignore_patterns(".git"))
        return dst

    def test_a_stamped_copy_records_the_stamped_commit_and_verifies(self):
        stamp = fi.stamp(self.root)
        pc = self.copy_without_git()
        m = fi.record(pc, say=QUIET)
        self.assertEqual(m["source"]["from"], "stamp")
        self.assertEqual(m["source"]["commit"], stamp["commit"])
        self.assertEqual(fi.verify(pc), [])

    def test_a_copy_that_differs_from_the_stamp_is_refused(self):
        fi.stamp(self.root)
        pc = self.copy_without_git()
        (pc / "serve/server.py").write_text("print('edited on the PC')\n")
        with self.assertRaisesRegex(fi.IdentityError, "not the stamped checkout"):
            fi.record(pc, say=QUIET)

    def test_a_copy_without_a_stamp_is_refused(self):
        pc = self.copy_without_git()
        with self.assertRaisesRegex(fi.IdentityError, "no fork-source.json"):
            fi.record(pc, say=QUIET)

    def test_a_copy_inside_an_unrelated_repository_does_not_borrow_its_commit(self):
        nested = self.root / "nested-copy"                           # lives inside self.root's Git checkout
        shutil.copytree(self.root, nested, ignore=shutil.ignore_patterns(".git", "nested-copy"))
        self.assertIsNone(fi.git_state(nested))
        with self.assertRaisesRegex(fi.IdentityError, "not a Git checkout"):
            fi.source_identity(nested)

    def test_the_stamp_is_written_only_in_a_checkout(self):
        pc = self.copy_without_git()
        with self.assertRaisesRegex(fi.IdentityError, "not the top of a Git checkout"):
            fi.stamp(pc)

    def test_a_forged_stamp_commit_with_the_right_tree_still_verifies_as_that_commit(self):
        # The stamp is the only commit evidence on a copy, so the tree digest is what ties it to the files.
        stamp = fi.stamp(self.root)
        pc = self.copy_without_git()
        data = fi.read_json(pc / fi.SOURCE_STAMP)
        data["commit"] = "f" * 40
        fi.write_json(pc / fi.SOURCE_STAMP, data)
        m = fi.record(pc, say=QUIET)
        self.assertEqual(m["source"]["commit"], "f" * 40)
        self.assertEqual(m["source"]["tree_sha256"], stamp["tree_sha256"])


class Rollback(IdentityCase):
    def next_candidate(self, engine=b"engine-v2"):
        """Source and binaries move on (committed), as after the next build."""
        (self.root / "src/a.cpp").write_text("int main() { return 2; }\n")
        (self.root / "src/new.cpp").write_text("// added\n")
        (self.root / "bin" / fi.ENGINE).write_bytes(engine)
        (self.root / "bin" / "extra.dll").write_bytes(b"dll")
        commit_all(self.root)

    def test_in_a_checkout_the_binaries_return_and_the_commit_is_named(self):
        a = self.record()
        a_engine = (self.root / "bin" / fi.ENGINE).read_bytes()
        self.next_candidate()
        b = self.record()
        self.assertNotEqual(a["id"], b["id"])
        rec = fi.read_json(self.root / fi.ROLLBACK)
        self.assertEqual(rec["current"]["id"], b["id"])
        self.assertEqual(rec["previous"]["id"], a["id"])
        self.assertTrue(rec["previous"]["archived"])
        self.assertIn(f"restore --to {a['id']}", rec["restore"])
        self.assertEqual(fi.verify(self.root), [])

        problems = fi.restore(self.root, a["id"], say=QUIET)
        self.assertEqual((self.root / "bin" / fi.ENGINE).read_bytes(), a_engine)
        self.assertFalse((self.root / "bin" / "extra.dll").exists())
        self.assertTrue(any(f"git checkout --detach {a['source']['commit'][:12]}" in p for p in problems), problems)
        self.assertTrue((self.root / "src/new.cpp").exists())          # a checkout's files are Git's to move
        git(self.root, "checkout", "-q", "--detach", a["source"]["commit"])
        self.assertEqual(fi.verify(self.root), [])
        self.assertEqual(fi.read_json(self.root / fi.ROLLBACK)["previous"]["id"], b["id"])   # and forward again

    def test_in_a_copy_the_source_and_the_stamp_return_too(self):
        fi.stamp(self.root)
        pc = self.root.parent / "pc"
        shutil.copytree(self.root, pc, ignore=shutil.ignore_patterns(".git"))
        a = fi.record(pc, say=QUIET)
        a_tree = fi.digest_of(fi.tree_hashes(pc))
        a_engine = (pc / "bin" / fi.ENGINE).read_bytes()
        self.next_candidate()                                           # the next deploy: new files and a new stamp
        fi.stamp(self.root)
        shutil.copytree(self.root, pc, ignore=shutil.ignore_patterns(".git", "bin"), dirs_exist_ok=True)
        (pc / "bin" / fi.ENGINE).write_bytes(b"engine-v2")
        (pc / "bin" / "extra.dll").write_bytes(b"dll")
        b = fi.record(pc, say=QUIET)
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(fi.verify(pc), [])

        self.assertEqual(fi.restore(pc, a["id"], say=QUIET), [])
        self.assertEqual((pc / "bin" / fi.ENGINE).read_bytes(), a_engine)
        self.assertFalse((pc / "bin" / "extra.dll").exists())
        self.assertFalse((pc / "src/new.cpp").exists())
        self.assertEqual(fi.digest_of(fi.tree_hashes(pc)), a_tree)
        self.assertEqual(fi.read_json(pc / fi.SOURCE_STAMP)["commit"], a["source"]["commit"])
        self.assertEqual(fi.read_json(pc / fi.MANIFEST)["id"], a["id"])

    def test_re_recording_the_same_candidate_keeps_the_earlier_rollback_target(self):
        a = self.record()
        (self.root / "bin" / fi.ENGINE).write_bytes(b"engine-v2")
        b = self.record(allow_dirty=False)
        self.record()
        self.assertEqual(fi.read_json(self.root / fi.ROLLBACK)["previous"]["id"], a["id"])
        self.assertEqual(fi.read_json(self.root / fi.ROLLBACK)["current"]["id"], b["id"])

    def test_the_first_candidate_has_no_previous(self):
        self.record()
        self.assertIsNone(fi.read_json(self.root / fi.ROLLBACK)["previous"])

    def test_a_damaged_archive_is_not_restored(self):
        a = self.record()
        (self.root / "bin" / fi.ENGINE).write_bytes(b"engine-v2")
        self.record()
        (self.root / fi.CANDIDATES / a["id"] / "bin" / fi.ENGINE).write_bytes(b"corrupt")
        with self.assertRaisesRegex(fi.IdentityError, "damaged"):
            fi.restore(self.root, a["id"], say=QUIET)
        self.assertEqual((self.root / "bin" / fi.ENGINE).read_bytes(), b"engine-v2")     # nothing was touched

    def test_an_unknown_candidate_is_refused(self):
        self.record()
        with self.assertRaisesRegex(fi.IdentityError, "no archived candidate"):
            fi.restore(self.root, "c-nope", say=QUIET)


class CommandLine(IdentityCase):
    def run_main(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = fi.main([*argv, "--root", str(self.root)] if argv[0] != "restore" else [*argv, "--root", str(self.root)])
        return code, err.getvalue()

    def test_record_then_verify_exit_codes(self):
        self.assertEqual(self.run_main("record")[0], 0)
        self.assertEqual(self.run_main("verify", "--arm", "a0")[0], 0)
        (self.root / "bin" / fi.ENGINE).write_bytes(b"wrong binary")
        code, err = self.run_main("verify", "--arm", "a0")
        self.assertEqual(code, 1)
        self.assertIn("MISMATCH binary", err)

    def test_record_without_a_binary_exits_1(self):
        (self.root / "bin" / fi.ENGINE).unlink()
        code, err = self.run_main("record")
        self.assertEqual(code, 1)
        self.assertIn("build first", err)


class RepositoryScope(unittest.TestCase):
    def test_the_real_tree_is_inside_the_identity_scope(self):
        root = fi.default_root()
        files = fi.tree_files(root)
        for needed in ("CMakeLists.txt", "serve/server.py", "fork/tools/fork_identity.py", "fork/profiles/a0-daily-control.json"):
            self.assertIn(needed, files)
        self.assertFalse([f for f in files if f.startswith(("bin/", "build/", fi.RUNS, fi.CANDIDATES))])

    def test_cmake_writes_the_engine_where_the_identity_tool_looks(self):
        cmake = (fi.default_root() / "CMakeLists.txt").read_text()
        self.assertIn("set_target_properties(basalt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/" + fi.BIN_DIR + ")",
                      cmake)


if __name__ == "__main__":
    unittest.main()
