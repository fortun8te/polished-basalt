"""Wrong-binary sentinel: a stale engine at the old, wrong path (build/bin) or a swapped one at the right path never
starts, and the Windows scripts agree with where CMake writes the engine.

    python -m pytest fork/tools -q
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_identity as fi  # noqa: E402
import fork_launch as fl  # noqa: E402
from test_fork_launch import LauncherCase  # noqa: E402

REPO = fi.default_root()
WINDOWS = REPO / "fork" / "windows"
STALE = b"STALE ENGINE FROM AN EARLIER BUILD"


class WrongBinarySentinel(LauncherCase):
    def setUp(self):
        super().setUp()
        stale = self.root / "build" / "bin"                      # where the scripts used to look
        stale.mkdir(parents=True)
        (stale / fi.ENGINE).write_bytes(STALE)

    def engine_started(self):
        cmd = self.m.spawned[-1]["cmd"]
        return Path(cmd[cmd.index("--engine-bin") + 1])

    def test_the_engine_that_starts_is_the_recorded_one_in_bin_never_build_bin(self):
        self.assertEqual(self.start("a0"), 0, self.said())
        started = self.engine_started()
        self.assertEqual(started, self.root / "bin" / fi.ENGINE)
        self.assertNotIn("build", started.parts)
        self.assertEqual(fi.sha256_file(started), fi.read_json(self.root / fi.MANIFEST)["binaries"][f"bin/{fi.ENGINE}"]["sha256"])
        receipt = fi.read_json(Path(self.m.spawned[0]["out"]).parent / "receipt.json")
        self.assertEqual(receipt["engine_sha256"], fi.sha256_file(started))
        self.assertNotEqual(receipt["engine_sha256"], fi.sha256_file(self.root / "build" / "bin" / fi.ENGINE))

    def test_a_rebuild_that_was_not_recorded_is_refused(self):
        (self.root / "bin" / fi.ENGINE).write_bytes(b"engine-v2 from a new build")
        self.assertEqual(self.start("a0"), 1)
        self.assertEqual(self.m.spawned, [])
        self.assertIn("the manifest records", self.said())

    def test_the_sentinel_at_the_right_path_is_refused(self):
        (self.root / "bin" / fi.ENGINE).write_bytes(STALE)
        self.assertEqual(self.start("a0"), 1)
        self.assertEqual(self.m.spawned, [])

    def test_with_the_engine_only_at_the_wrong_path_nothing_starts(self):
        (self.root / "bin" / fi.ENGINE).unlink()
        self.assertEqual(self.start("a0"), 1)
        self.assertEqual(self.m.spawned, [])
        self.assertIn(f"binary bin/{fi.ENGINE} is missing", self.said())

    def test_recording_without_an_engine_in_bin_does_not_accept_the_build_directory_one(self):
        (self.root / "bin" / fi.ENGINE).unlink()
        with self.assertRaisesRegex(fi.IdentityError, "build first"):
            fi.record(self.root, build_dir=self.root / "build", say=lambda *_: None)

    def test_the_command_line_refuses_the_sentinel_without_touching_the_machine(self):
        (self.root / "bin" / fi.ENGINE).write_bytes(STALE)
        runs = tempfile.mkdtemp()
        self.assertEqual(fl.main(["start", "--arm", "a0", "--root", str(self.root), "--runs-dir", runs]), 1)
        self.assertEqual(list(Path(runs).glob("a0*")), [])

    def test_a_rebuild_is_started_after_it_is_recorded_and_the_old_one_after_a_restore(self):
        old_sha = fi.sha256_file(self.root / "bin" / fi.ENGINE)
        (self.root / "bin" / fi.ENGINE).write_bytes(b"engine-v2 from a new build")
        new = fi.record(self.root, say=lambda *_: None)
        self.assertEqual(self.start("a0"), 0, self.said())
        self.assertEqual(fi.sha256_file(self.engine_started()), new["binaries"][new["engine"]]["sha256"])
        self.m.alive.clear()
        previous = fi.read_json(self.root / fi.ROLLBACK)["previous"]["id"]
        self.assertEqual(fi.restore(self.root, previous, say=lambda *_: None), [])
        self.assertEqual(self.start("a0"), 0, self.said())
        self.assertEqual(fi.sha256_file(self.engine_started()), old_sha)


class WindowsScripts(unittest.TestCase):
    def text(self, name):
        return (WINDOWS / name).read_text()

    def test_no_script_looks_in_the_build_directory_for_the_engine(self):
        for name in ("Build-Fork.cmd", "Start-ForkBasalt.ps1", "Stop-ForkBasalt.ps1"):
            self.assertIsNone(re.search(r"build[\\/]+bin|FORK_BUILD%[\\/]+bin", self.text(name), re.I), name)

    def test_cmake_writes_every_fork_executable_to_the_source_trees_bin(self):
        cmake = (REPO / "CMakeLists.txt").read_text()
        for target in ("basalt", "basalt-bench", "basalt-probe"):
            self.assertIn(f"set_target_properties({target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_SOURCE_DIR}}/{fi.BIN_DIR})", cmake, target)

    def test_the_build_script_records_the_candidate_after_a_successful_build(self):
        text = self.text("Build-Fork.cmd")
        self.assertLess(text.index("--build \"%FORK_BUILD%\" --target"), text.index("fork_identity.py\" record"))
        self.assertIn('record --root "%FORK_SRC%"', text)
        self.assertEqual(text.count("if errorlevel 1 exit /b %errorlevel%"), 3)     # vcvars, configure, build

    def test_the_powershell_scripts_only_call_the_python_tools(self):
        for name, tool in (("Start-ForkBasalt.ps1", "fork_launch.py"), ("Stop-ForkBasalt.ps1", "fork_launch.py")):
            text = self.text(name)
            code = [ln for ln in re.sub(r"<#.*?#>", "", text, flags=re.S).splitlines() if ln.strip()]
            self.assertIn(tool, text)
            self.assertLessEqual(len(code), 20, name)
            for logic in ("Start-Process", "Stop-Process", "Set-Item", "Env:", "Invoke-RestMethod", "nvidia-smi", "ValidateSet"):
                self.assertNotIn(logic, "\n".join(code), (name, logic))
            self.assertIn("exit $LASTEXITCODE", text)
            self.assertIn("$PSScriptRoot", text)                                   # the tree is the one the script is in

    def test_the_start_script_passes_every_selector_to_the_launcher(self):
        text = self.text("Start-ForkBasalt.ps1")
        for flag in ("'--arm'", "'--disable'", "'--enable'", "'--set'", "'--wait'", "'--models'", "'--cuda-path'"):
            self.assertIn(flag, text)

    def test_every_test_the_plan_runs_is_built_by_the_build_script_and_exists_in_cmake(self):
        plan = (REPO / "docs" / "polished-basalt" / "TEST_PLAN.md").read_text()
        wanted = set(re.search(r'ctest [^\n]*-R "([^"]+)"', plan).group(1).split("|"))
        build = self.text("Build-Fork.cmd")
        targets = set(re.search(r"--target(.*?)--parallel", build, re.S).group(1).replace("^", " ").split())
        cmake = (REPO / "CMakeLists.txt").read_text()
        self.assertTrue(wanted)
        self.assertEqual(wanted - targets, set(), "built by Build-Fork.cmd")
        loops = {f"platform_{t}" for m in re.findall(r"foreach\(_t ([^)]*)\)", cmake) for t in m.split()}
        for name in wanted | targets:
            defined = (re.search(rf"(add_executable|basalt_cpu_test|basalt_gpu_test)\(\s*(NAME\s+)?{re.escape(name)}\b", cmake)
                       or name in loops or name in {"basalt", "basalt-probe", "basalt-bench"})
            self.assertTrue(defined, f"{name} is not defined in CMakeLists.txt")

    def test_the_profiles_and_the_scripts_agree_on_the_ports(self):
        text = self.text("Start-ForkBasalt.ps1")
        self.assertIn(str(fl.PRODUCTION_PORT), text)                               # documented; the launcher enforces it
        for arm in fl.arm_names(REPO):
            self.assertNotEqual(fl.arm_port(REPO, arm), fl.PRODUCTION_PORT)


if __name__ == "__main__":
    unittest.main()
