"""The launcher: refusals, the child environment, isolation per arm, and stop.

    python -m pytest fork/tools -q
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_identity as fi  # noqa: E402
import fork_launch as fl  # noqa: E402
from test_fork_identity import GIT, QUIET, make_tree  # noqa: E402

REPO = fi.default_root()


class FakeMachine:
    """Ports, GPU memory, processes: everything the launcher asks the machine, scripted."""
    def __init__(self):
        self.listening, self.free, self.spawned, self.killed, self.alive = set(), 16000, [], [], {}
        self.next_pid, self.env = 4242, {"PATH": "/usr/bin", "HOME": "/home/x"}

    def spawn(self, cmd, cwd, env, out, err):
        self.next_pid += 1
        self.spawned.append({"cmd": cmd, "cwd": cwd, "env": env, "out": out, "err": err, "pid": self.next_pid})
        self.alive[self.next_pid] = f"start-{self.next_pid}"
        return self.next_pid

    def terminate(self, pid):
        self.killed.append(pid)
        self.alive.pop(pid, None)

    def hooks(self):
        return fl.Hooks(port_listening=lambda p: p in self.listening, port_bindable=lambda p: p not in self.listening,
                        gpu_free_mib=lambda: self.free, http_ok=lambda p: True, spawn=self.spawn,
                        start_token=lambda pid: self.alive.get(pid), terminate=self.terminate, sleep=lambda s: None,
                        env=self.env, python="/venv/python")


@unittest.skipUnless(GIT, "git is needed")
class LauncherCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        self.root, self.models = base / "tree", base / "models"
        make_tree(self.root, self.models)
        fi.record(self.root, say=QUIET)
        self.m = FakeMachine()
        self.lines = []

    def start(self, arm="a0", **kw):
        kw.setdefault("hooks", self.m.hooks())
        return fl.start(self.root, arm, say=self.lines.append, **kw)

    def said(self):
        return "\n".join(self.lines)


class Start(LauncherCase):
    def test_a_recorded_candidate_starts_from_the_source_trees_bin_with_an_isolated_run_directory(self):
        self.assertEqual(self.start("a1"), 0, self.said())
        (spawn,) = self.m.spawned
        cmd = spawn["cmd"]
        self.assertEqual(cmd[:4], ["/venv/python", "-u", "-m", "serve.server"])
        self.assertEqual(cmd[cmd.index("--engine-bin") + 1], str(self.root / "bin" / fi.ENGINE))
        self.assertEqual(spawn["cwd"], str(self.root))
        run_dir = Path(cmd[cmd.index("--config") + 1]).parent
        self.assertEqual(run_dir.parent, self.root / fi.RUNS / "a1")
        for name in ("profile.resolved.json", "receipt.json"):
            self.assertTrue((run_dir / name).is_file())
        self.assertEqual(Path(spawn["out"]).parent, run_dir)
        self.assertEqual(Path(spawn["err"]).parent, run_dir)
        cfg = fi.read_json(run_dir / "profile.resolved.json")
        self.assertEqual(cfg["server"]["port"], 18893)                      # a1 is the second arm
        self.assertEqual(cfg["server"]["model"], str(self.models / "pack.basalt"))
        receipt = fi.read_json(run_dir / "receipt.json")
        self.assertEqual(receipt["engine_sha256"], fi.sha256_file(self.root / "bin" / fi.ENGINE))
        self.assertEqual(receipt["pid"], spawn["pid"])
        self.assertEqual(receipt["model_check"], "full")
        self.assertEqual(fi.read_json(self.root / fi.RUNS / fl.CURRENT)["pid"], spawn["pid"])
        self.assertEqual((self.root / fi.RUNS / ".gitignore").read_text(), "*\n")

    def test_every_run_has_its_own_directory(self):
        self.assertEqual(self.start("a0"), 0)
        self.m.alive.clear()                                                # the first one ended
        self.assertEqual(self.start("a0"), 0)
        self.assertEqual(self.start("a1"), 1)                               # the second still runs
        dirs = {Path(s["out"]).parent for s in self.m.spawned}
        self.assertEqual(len(dirs), 2)

    def test_inherited_switches_do_not_reach_the_child(self):
        self.m.env["BASALT_FORK_STAGING"] = "0"
        self.m.env["basalt_fork_ecoqos"] = "0"
        self.assertEqual(self.start("a0"), 0)
        env = self.m.spawned[0]["env"]
        self.assertFalse([k for k in env if k.upper().startswith("BASALT_FORK_")])
        self.assertEqual(self.m.env["BASALT_FORK_STAGING"], "0")             # the caller's environment is untouched
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        receipt = fi.read_json(Path(self.m.spawned[0]["out"]).parent / "receipt.json")
        self.assertEqual(receipt["env_fork"], {})

    def test_named_switches_are_the_only_ones_set_and_are_recorded(self):
        self.assertEqual(self.start("a0", disable=["staging", "BASALT_FORK_PDEP_SIGNS"], enable=["GDN_REC2"],
                                    numbers=["DRAFT_CUM_MULT=1.2"]), 0, self.said())
        want = {"BASALT_FORK_STAGING": "0", "BASALT_FORK_PDEP_SIGNS": "0", "BASALT_FORK_GDN_REC2": "1",
                "BASALT_FORK_DRAFT_CUM_MULT": "1.2"}
        env = self.m.spawned[0]["env"]
        self.assertEqual({k: v for k, v in env.items() if k.startswith("BASALT_FORK_")}, want)
        receipt = fi.read_json(Path(self.m.spawned[0]["out"]).parent / "receipt.json")
        self.assertEqual(receipt["env_fork"], want)

    def test_cuda_path_leads_the_search_path(self):
        self.assertEqual(self.start("a0", cuda_path="/cuda"), 0)
        self.assertTrue(self.m.spawned[0]["env"]["PATH"].startswith(os.path.join("/cuda", "bin") + os.pathsep))

    def test_wait_reports_readiness_or_an_early_exit(self):
        self.assertEqual(self.start("a0", wait_s=5), 0)
        self.assertIn("ready", self.said())
        self.m.alive.clear()
        self.lines.clear()
        h = self.m.hooks()
        h.http_ok = lambda p: False
        h.spawn = lambda *a: (self.m.spawn(*a), self.m.alive.clear())[0]    # dies at once
        self.assertEqual(self.start("a0", wait_s=5, hooks=h), 2)
        self.assertIn("exited during start-up", self.said())


class Switches(unittest.TestCase):
    def test_the_documented_example_name_pdep_is_rejected_with_the_real_names(self):
        with self.assertRaisesRegex(fl.LaunchError, "unknown switch 'pdep'.*PDEP_SIGNS"):
            fl.switch_values(disable=["pdep"])

    def test_a_switch_must_be_used_the_way_its_kind_reads(self):
        with self.assertRaisesRegex(fl.LaunchError, "off unless enabled"):
            fl.switch_values(disable=["GDN_REC2"])
        with self.assertRaisesRegex(fl.LaunchError, "out of range"):
            fl.switch_values(disable=["DRAFT_CUM_MULT"])
        with self.assertRaisesRegex(fl.LaunchError, "not an opt-in"):
            fl.switch_values(enable=["STAGING"])

    def test_numbers_outside_the_range_or_not_numbers_are_rejected(self):
        for bad in ("DRAFT_CUM_MULT=0", "DRAFT_CUM_MULT=9", "DRAFT_CUM_MULT=fast", "DRAFT_CUM_MULT", "STAGING=1"):
            with self.assertRaises(fl.LaunchError, msg=bad):
                fl.switch_values(numbers=[bad])
        self.assertEqual(fl.switch_values(numbers=["draft_cum_mult=0.25"]), {"DRAFT_CUM_MULT": "0.25"})

    def test_the_table_names_every_switch_the_sources_read(self):
        found = set()
        for sub in ("src", "include"):
            for path in (REPO / sub).rglob("*"):
                if path.suffix in {".cpp", ".hpp", ".cu", ".cuh", ".h"}:
                    text = path.read_text(errors="replace")
                    found |= set(re.findall(r'fork_(?:switch|opt_in|number)\(\s*"([A-Z0-9_]+)"', text))
                    found |= set(re.findall(r'BASALT_FORK_([A-Z0-9_]+)', text))
        self.assertEqual(found, set(fl.SWITCHES))

    def test_the_kinds_match_the_functions_that_read_them(self):
        kinds = {"fork_switch": fl.KILL, "fork_opt_in": fl.OPT_IN, "fork_number": fl.NUMBER}
        for sub in ("src", "include"):
            for path in (REPO / sub).rglob("*"):
                if path.suffix in {".cpp", ".hpp", ".cu", ".cuh", ".h"}:
                    for fn, name in re.findall(r'(fork_(?:switch|opt_in|number))\(\s*"([A-Z0-9_]+)"', path.read_text(errors="replace")):
                        self.assertEqual(fl.SWITCHES[name][0], kinds[fn], f"{name} in {path}")


class Refusals(LauncherCase):
    def refused(self, arm="a0", **kw):
        self.assertEqual(self.start(arm, **kw), 1, self.said())
        self.assertEqual(self.m.spawned, [])
        self.assertFalse((self.root / fi.RUNS / arm).exists(), "a refused start leaves no run directory")

    def test_a_tampered_engine_binary(self):
        (self.root / "bin" / fi.ENGINE).write_bytes(b"not the recorded engine")
        self.refused()
        self.assertIn("binary bin/", self.said())

    def test_a_changed_model_file(self):
        (self.models / "pack.basalt").write_bytes(b"PACK-BYTES")
        self.refused()
        self.assertIn("does not match the recorded sha256", self.said())

    def test_a_changed_source_tree(self):
        (self.root / "serve/server.py").write_text("print('edited')\n")
        self.refused()

    def test_no_manifest(self):
        (self.root / fi.MANIFEST).unlink()
        self.refused()
        self.assertIn("fork-identity.json is missing", self.said())

    def test_unknown_arm_and_unknown_switch(self):
        self.refused("nope")
        self.assertIn("unknown arm 'nope'; known: a0, a1", self.said())
        self.lines.clear()
        self.refused(disable=["pdep"])

    def test_a_listener_on_the_production_port_or_on_the_arms_own_port(self):
        for port in (fl.PRODUCTION_PORT, 18892):
            self.m.listening = {port}
            self.lines.clear()
            self.refused("a0")
            self.assertIn(str(port), self.said())

    def test_a_listener_on_another_arms_port_does_not_block_this_arm(self):
        self.m.listening = {18893}
        self.assertEqual(self.start("a0"), 0, self.said())

    def test_too_little_or_unknown_free_gpu_memory(self):
        self.m.free = 4000
        self.refused()
        self.assertIn("only 4000 MiB", self.said())
        self.m.free = None
        self.lines.clear()
        self.refused()
        self.assertIn("cannot read the free GPU memory", self.said())
        self.assertEqual(self.start("a0", skip_gpu_check=True), 0)

    def test_a_running_fork_server(self):
        self.assertEqual(self.start("a0"), 0)
        self.assertEqual(self.start("a1"), 1)
        self.assertIn("already running", self.said())
        self.assertEqual(len(self.m.spawned), 1)

    def test_a_stale_record_is_ignored(self):
        self.assertEqual(self.start("a0"), 0)
        self.m.alive.clear()
        self.assertEqual(self.start("a1"), 0)

    def test_two_arms_may_not_share_a_port(self):
        p = self.root / "fork/profiles/a1.json"
        p.write_text(p.read_text().replace("18893", "18892"))
        fi.record(self.root, allow_dirty=True, say=QUIET)
        self.refused("a1")
        self.assertIn("also used by a0", self.said())

    def test_an_arm_may_not_use_the_production_port(self):
        p = self.root / "fork/profiles/a0.json"
        p.write_text(p.read_text().replace("18892", str(fl.PRODUCTION_PORT)))
        fi.record(self.root, allow_dirty=True, say=QUIET)
        self.refused("a0")
        self.assertIn("production port", self.said())


class Isolation(unittest.TestCase):
    def profiles(self):
        return {p.stem: fi.read_json(p) for p in (REPO / "fork" / "profiles").glob("*.json")}

    def test_every_arm_has_its_own_port_and_none_is_the_production_port(self):
        ports = {a: c["server"]["port"] for a, c in self.profiles().items()}
        self.assertEqual(len(set(ports.values())), len(ports), ports)
        self.assertNotIn(fl.PRODUCTION_PORT, ports.values())
        for arm in ports:
            self.assertEqual(fl.arm_port(REPO, arm), ports[arm])

    def test_no_arm_writes_outside_its_run_directory_even_when_the_profile_names_another_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for arm, cfg in self.profiles().items():
                resolved, redirected = fl.materialize(REPO, REPO / "fork" / "profiles" / f"{arm}.json", run_dir)
                for key, val in resolved["engine"].items():
                    if fl.flag_of(key) in fl.OUTPUT_FLAGS:
                        self.assertEqual(Path(val).parent, run_dir, (arm, key))
                        self.assertIn(key, redirected)

    def test_the_profiles_themselves_never_name_a_production_path_to_write(self):
        for arm, cfg in self.profiles().items():
            for key, val in cfg["engine"].items():
                if fl.flag_of(key) in fl.OUTPUT_FLAGS:
                    self.assertNotIn("local-ai-20261026", str(val), (arm, key))

    def test_inputs_are_made_absolute_so_the_copy_in_the_run_directory_reads_the_same_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "p"
            prof.mkdir()
            (prof / "x.json").write_text('{"server": {"model": "m/pack.basalt"}, "engine": {"ple-gguf": "ple.gguf", "spec": 7}}')
            cfg, redirected = fl.materialize(Path("/work"), prof / "x.json", Path(tmp) / "run")
            self.assertEqual(cfg["server"]["model"], os.path.abspath(prof / "m" / "pack.basalt"))
            self.assertEqual(cfg["engine"]["ple-gguf"], os.path.abspath("/work/ple.gguf"))
            self.assertEqual(cfg["engine"]["spec"], 7)
            self.assertEqual(redirected, {})


class Promotion(unittest.TestCase):
    def test_the_plan_never_touches_the_endpoint_or_the_outputs(self):
        daily = {"server": {"model": "m.basalt", "host": "127.0.0.1", "port": 18891, "alias": "x"},
                 "engine": {"spec": 7, "expert-profile-save": "/prod/profile.bin", "ple-io": "ssd"}}
        arm = {"server": {"model": "m.basalt", "host": "0.0.0.0", "port": 18893, "alias": "x"},
               "engine": {"spec": 7, "expert-profile-save": "runs/x.bin", "ple-io": "ram", "adapt-every": 1}}
        plan = fl.profile_diff(arm, daily)
        self.assertEqual(plan, ["+ engine.adapt-every: 1", '~ engine.ple-io: "ssd" -> "ram"'])
        text = "\n".join(plan)
        for forbidden in ("port", "host", "expert-profile-save"):
            self.assertNotIn(forbidden, text)

    def test_the_real_arms_differ_from_a0_only_in_engine_settings(self):
        a0 = fi.read_json(REPO / "fork/profiles/a0-daily-control.json")
        for p in (REPO / "fork/profiles").glob("a[123]-*.json"):
            for line in fl.profile_diff(fi.read_json(p), a0):
                self.assertIn(" engine.", line, p.name)


@unittest.skipUnless(os.name == "posix", "uses ps and process groups")
class RealProcess(unittest.TestCase):
    def test_spawn_identify_and_stop_a_detached_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = fl.spawn_detached([sys.executable, "-c", "import time; time.sleep(60)"], tmp, dict(os.environ),
                                    os.path.join(tmp, "o.log"), os.path.join(tmp, "e.log"))
            self.addCleanup(lambda: os.path.exists(f"/proc/{pid}") and fl.terminate(pid, 1))
            token = fl.process_start_token(pid)
            self.assertTrue(token)
            self.assertEqual(fl.process_start_token(pid), token)
            hooks = fl.Hooks()
            runs = Path(tmp) / "runs"
            fi.write_json(runs / fl.CURRENT, {"pid": pid, "start_token": "some other start time", "arm": "a0", "port": 1})
            self.assertEqual(fl.stop(tmp, runs, hooks, say=lambda s: None), 1)       # a recycled pid is left alone
            self.assertEqual(fl.process_start_token(pid), token)
            fi.write_json(runs / fl.CURRENT, {"pid": pid, "start_token": token, "arm": "a0", "port": 1})
            self.assertEqual(fl.stop(tmp, runs, hooks, say=lambda s: None), 0)
            deadline = time.time() + 10
            while fl.process_start_token(pid) is not None and time.time() < deadline:
                time.sleep(0.1)
            self.assertIsNone(fl.process_start_token(pid))
            self.assertFalse((runs / fl.CURRENT).exists())

    def test_no_such_process_has_no_token(self):
        self.assertIsNone(fl.process_start_token(2 ** 22 + 12345))


class Stop(LauncherCase):
    def test_stop_ends_only_the_recorded_server(self):
        self.assertEqual(self.start("a0"), 0)
        pid = self.m.spawned[0]["pid"]
        self.m.alive[pid] = "a different start time"
        self.assertEqual(fl.stop(self.root, hooks=self.m.hooks(), say=self.lines.append), 1)
        self.assertEqual(self.m.killed, [])
        self.m.alive[pid] = f"start-{pid}"
        self.assertEqual(fl.stop(self.root, hooks=self.m.hooks(), say=self.lines.append), 0)
        self.assertEqual(self.m.killed, [pid])
        self.assertFalse((self.root / fi.RUNS / fl.CURRENT).exists())

    def test_stop_without_a_server_is_quiet_success(self):
        self.assertEqual(fl.stop(self.root, hooks=self.m.hooks(), say=self.lines.append), 0)
        self.assertEqual(self.m.killed, [])

    def test_status_reports_identity_and_readiness(self):
        self.assertEqual(self.start("a0"), 0)
        st = fl.status(self.root, hooks=self.m.hooks())
        self.assertTrue(st["running"] and st["ready"])
        self.assertEqual(st["identity_problems"], [])
        (self.root / "bin" / fi.ENGINE).write_bytes(b"swapped after start")
        self.assertTrue(fl.status(self.root, hooks=self.m.hooks())["identity_problems"])


class CommandLine(unittest.TestCase):
    def test_an_unknown_arm_exits_1_before_touching_the_machine(self):
        self.assertEqual(fl.main(["start", "--arm", "nope", "--root", str(REPO), "--runs-dir",
                                  tempfile.mkdtemp()]), 1)

    def test_split_list(self):
        self.assertEqual(fl.split_list(["a,b", "c", ""]), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
