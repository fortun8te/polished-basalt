#!/usr/bin/env python3
"""Start, stop and inspect one fork server arm, only when the files on disk are the recorded candidate.

  fork_launch.py start --arm a1-ple-ram [--disable STAGING,PDEP_SIGNS] [--enable GDN_REC2] [--set DRAFT_CUM_MULT=1.2]
  fork_launch.py stop
  fork_launch.py status
  fork_launch.py promote-plan --arm a1-ple-ram --daily DAILY.json     # what an arm changes, endpoint left alone

`start` refuses, and starts nothing, unless: fork_identity.py verify passes for the arm (the engine and every file in
bin/, the source commit and tree, the profile and the model files by sha256); nothing listens on the production port
or on any arm's port; and enough GPU memory is free. The server then runs from a resolved copy of the profile in its
own run directory (logs, receipt, and every file the engine would write), on the arm's own port, with a child
environment that holds exactly the BASALT_FORK_* switches named here and no inherited ones.
Only the standard library; Python 3.9 or newer.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fork_identity as fi  # noqa: E402

PRODUCTION_PORT = 18891
MIN_FREE_MIB = 14336                 # one 16 GB model at a time
CURRENT = "current-fork-server.json"
# Engine flags that write a file. A profile's value is replaced by a file in the run directory, so an arm never
# overwrites a file another server or an earlier run owns.
OUTPUT_FLAGS = ("expert-profile-save", "ppl-out", "draft-log")
INPUT_ENGINE_FLAGS = ("ple-gguf",)

# Every BASALT_FORK_<NAME> the sources read (include/ and src/): kind and, for numbers, the accepted range. A test
# compares this table with the sources.
KILL, OPT_IN, NUMBER = "kill", "opt-in", "number"
SWITCHES = {
    "PIN_HEADROOM": (KILL,), "STAGING": (KILL,), "PDEP_SIGNS": (KILL,), "ECOQOS": (KILL,),
    "SPLIT_COST": (KILL,), "THRESH_RAMP": (KILL,),
    "GDN_REC2": (OPT_IN,), "LOOKUP_OVERRIDE": (OPT_IN,), "MISS_THRESH": (OPT_IN,), "COUPLED_DRAFT": (OPT_IN,),
    "DRAFT_CUM_MULT": (NUMBER, 0.25, 4.0), "COUPLED_MIN_TEMP": (NUMBER, 0.0, 4.0),
}


class LaunchError(Exception):
    """A refusal: nothing was started."""


# ------------------------------------------------------------------------------------------------ switches, profiles

def switch_name(text: str) -> str:
    name = text.strip().upper()
    name = name[len("BASALT_FORK_"):] if name.startswith("BASALT_FORK_") else name
    if name not in SWITCHES:
        raise LaunchError(f"unknown switch {text!r}; known: {', '.join(sorted(SWITCHES))}")
    return name


def switch_values(disable=(), enable=(), numbers=()) -> dict[str, str]:
    """{name: value} for the child environment; LaunchError for an unknown name or a value the engine would ignore."""
    out: dict[str, str] = {}
    for item in disable:
        name = switch_name(item)
        if SWITCHES[name][0] != KILL:
            raise LaunchError(f"{name} cannot be disabled: " + ("it is off unless enabled (--enable)" if SWITCHES[name][0] == OPT_IN
                                                                else "it is a number (--set), and 0 is out of range"))
        out[name] = "0"
    for item in enable:
        name = switch_name(item)
        if SWITCHES[name][0] != OPT_IN:
            raise LaunchError(f"{name} is not an opt-in switch")
        out[name] = "1"
    for item in numbers:
        raw_name, sep, raw = item.partition("=")
        name = switch_name(raw_name)
        if not sep or SWITCHES[name][0] != NUMBER:
            raise LaunchError(f"{item!r}: expected NAME=NUMBER for a number switch")
        try:
            value = float(raw)
        except ValueError:
            raise LaunchError(f"{item!r}: {raw!r} is not a number") from None
        lo, hi = SWITCHES[name][1:]
        if not lo <= value <= hi:
            raise LaunchError(f"{name}={raw} is outside [{lo}, {hi}]; the engine would ignore it")
        out[name] = raw
    return out


def child_env(base, switches: dict[str, str], cuda_path: str | None = None) -> dict[str, str]:
    """The environment of the server: the caller's, without any BASALT_FORK_* it holds, plus the named switches."""
    env = {k: v for k, v in base.items() if not k.upper().startswith("BASALT_FORK_")}
    for name, value in switches.items():
        env[f"BASALT_FORK_{name}"] = value
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    if cuda_path:
        env["PATH"] = os.pathsep.join([os.path.join(cuda_path, "bin"), os.path.join(cuda_path, "bin", "x64"),
                                       env.get("PATH", "")])
    return env


def flag_of(key) -> str:
    return str(key).strip().lower().replace("_", "-").removeprefix("--")


def arm_names(root: Path) -> list[str]:
    return [p.stem for p in fi.profile_files(root)]


def arm_ports(root: Path) -> dict[str, object]:
    out = {}
    for p in fi.profile_files(root):
        out[p.stem] = (fi.read_json(p).get("server") or {}).get("port")
    return out


def arm_port(root: Path, arm: str, production_port: int = PRODUCTION_PORT) -> int:
    ports = arm_ports(root)
    port = ports.get(arm)
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        raise LaunchError(f"profile {arm}: server.port must be a port number (each arm has its own)")
    if port == production_port:
        raise LaunchError(f"profile {arm}: port {port} is the production port")
    same = sorted(a for a, p in ports.items() if p == port and a != arm)
    if same:
        raise LaunchError(f"profile {arm}: port {port} is also used by {', '.join(same)}; every arm has its own")
    return port


def materialize(root: Path, profile: Path, run_dir: Path) -> tuple[dict, dict]:
    """(config, redirected): the profile with its input files made absolute and every output file moved into
    run_dir. `redirected` maps the engine key to {from, to}."""
    cfg = copy.deepcopy(fi.read_json(profile))
    server, engine = cfg.setdefault("server", {}), cfg.setdefault("engine", {})
    for key in ("model", "tokenizer"):
        if server.get(key):
            server[key] = fi.resolve_input(server[key], profile.parent)
    redirected = {}
    for key in list(engine):
        flag = flag_of(key)
        if flag in INPUT_ENGINE_FLAGS and isinstance(engine[key], str):
            engine[key] = fi.resolve_input(engine[key], root)
        elif flag in OUTPUT_FLAGS:
            old = engine[key]
            new = str(run_dir / (Path(str(old)).name or flag))
            redirected[key] = {"from": old, "to": new}
            engine[key] = new
    return cfg, redirected


# ------------------------------------------------------------------------------------------------ the machine

def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def gpu_free_mib(run=subprocess.run):
    """Free memory of the first GPU in MiB; None when nvidia-smi cannot say."""
    try:
        r = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError, AttributeError):
        return None


def http_ok(port: int, path: str = "/health") -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


def process_start_token(pid: int, run=subprocess.run):
    """A string that names this process's start time (None: no such process). A recycled pid has another token."""
    if os.name == "nt":
        script = ("$p = Get-Process -Id %d -ErrorAction SilentlyContinue; "
                  "if ($p) { $p.StartTime.ToUniversalTime().ToString('o') }" % pid)
        shell = shutil.which("powershell") or shutil.which("pwsh") or "powershell"
        try:
            r = run([shell, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                    timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() or None
    try:
        r = run(["ps", "-o", "stat=", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    if r.returncode != 0 or not out or out.split()[0].startswith("Z"):       # gone, or a zombie that only awaits wait()
        return None
    return " ".join(out.split()[1:])


def spawn_detached(cmd, cwd, env, out_path, err_path) -> int:
    """Run cmd detached from this process (it keeps running after we exit), output to the two files; its pid."""
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    else:
        kw["start_new_session"] = True
    with open(out_path, "ab") as out, open(err_path, "ab") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kw)
    pid = proc.pid
    with warnings.catch_warnings():             # the child outlives us on purpose
        warnings.simplefilter("ignore", ResourceWarning)
        del proc
    return pid


def terminate(pid: int, grace_s: float = 30.0, run=subprocess.run) -> None:
    """Stop the process and its children: SIGTERM to the group (the server forwards a QUIT to the engine), then
    SIGKILL after grace_s; taskkill /T /F on Windows."""
    if os.name == "nt":
        run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)
        return
    try:
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace_s
    while time.time() < end:
        if process_start_token(pid) is None:
            return
        time.sleep(0.2)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Hooks:
    """The machine-facing calls; tests replace them."""
    def __init__(self, **kw):
        self.port_listening, self.port_bindable, self.gpu_free_mib = port_listening, port_bindable, gpu_free_mib
        self.http_ok, self.start_token, self.spawn = http_ok, process_start_token, spawn_detached
        self.terminate, self.sleep, self.env = terminate, time.sleep, os.environ
        self.python = sys.executable
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(k)
            setattr(self, k, v)


# ------------------------------------------------------------------------------------------------ start / stop / status

def runs_root(root: Path, runs_dir=None) -> Path:
    d = Path(runs_dir) if runs_dir else root / fi.RUNS
    if not (d / ".gitignore").exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitignore").write_text("*\n")
    return d


def read_current(runs: Path):
    try:
        return fi.read_json(runs / CURRENT)
    except (OSError, ValueError):
        return None


def running(rec, hooks) -> bool:
    return bool(rec) and hooks.start_token(rec["pid"]) == rec.get("start_token")


def start(root, arm: str, *, disable=(), enable=(), numbers=(), wait_s: float = 0.0, models: str = "full",
          min_free_mib: int = MIN_FREE_MIB, cuda_path=None, runs_dir=None, production_port: int = PRODUCTION_PORT,
          skip_gpu_check: bool = False, hooks: Hooks | None = None, say=print) -> int:
    """0: started (and ready when wait_s > 0). 1: refused, nothing started. 2: started but not ready."""
    hooks = hooks or Hooks()
    root = Path(root).resolve()
    try:
        names = arm_names(root)
        if arm not in names:
            raise LaunchError(f"unknown arm {arm!r}; known: {', '.join(names)}")
        switches = switch_values(disable, enable, numbers)
        problems = fi.verify(root, arm, models, say=lambda s: say(s))
        if problems:
            raise LaunchError("the files on disk are not the recorded candidate; nothing was started:\n  "
                              + "\n  ".join(problems))
        manifest = fi.read_json(root / fi.MANIFEST)
        port = arm_port(root, arm, production_port)
        runs = runs_root(root, runs_dir)
        rec = read_current(runs)
        if running(rec, hooks):
            raise LaunchError(f"a fork server is already running (pid {rec['pid']}, arm {rec['arm']}, port "
                              f"{rec['port']}); stop it first")
        busy = [p for p in {production_port, *(p for p in arm_ports(root).values() if isinstance(p, int))}
                if hooks.port_listening(p)]
        if busy:
            raise LaunchError(f"something already listens on port(s) {', '.join(map(str, sorted(busy)))}; "
                              "nothing was stopped or started")
        if not hooks.port_bindable(port):
            raise LaunchError(f"port {port} cannot be bound")
        free = None
        if not skip_gpu_check:
            free = hooks.gpu_free_mib()
            if free is None:
                raise LaunchError("cannot read the free GPU memory (nvidia-smi); pass --skip-gpu-check to go on")
            if free < min_free_mib:
                raise LaunchError(f"only {free} MiB of GPU memory is free (need {min_free_mib}); another model is "
                                  "loaded. Nothing was started")
    except (LaunchError, fi.IdentityError) as e:
        say(f"fork_launch: {e}")
        return 1

    run_dir = runs / arm / fi.now_stamp()
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = runs / arm / f"{fi.now_stamp()}-{n}"
    run_dir.mkdir(parents=True)
    profile = root / manifest["profiles"][arm]["profile"]
    cfg, redirected = materialize(root, profile, run_dir)
    resolved = run_dir / "profile.resolved.json"
    fi.write_json(resolved, cfg)
    engine = root / manifest["engine"]
    cmd = [hooks.python, "-u", "-m", "serve.server", "--engine-bin", str(engine), "--config", str(resolved)]
    env = child_env(hooks.env, switches, cuda_path)
    out_log, err_log = run_dir / "server.out.log", run_dir / "server.err.log"
    pid = hooks.spawn(cmd, str(root), env, str(out_log), str(err_log))
    token = hooks.start_token(pid)
    started = fi.utc_now()
    receipt = {
        "arm": arm, "port": port, "pid": pid, "start_token": token, "started_utc": started, "run_dir": str(run_dir),
        "command": cmd, "cwd": str(root), "switches": switches,
        "env_fork": {k: v for k, v in env.items() if k.startswith("BASALT_FORK_")},
        "candidate": manifest["id"], "engine": manifest["engine"],
        "engine_sha256": manifest["binaries"][manifest["engine"]]["sha256"], "commit": manifest["source"]["commit"],
        "dirty": manifest["source"]["dirty"], "tree_sha256": manifest["source"]["tree_sha256"],
        "profile_sha256": manifest["profiles"][arm]["sha256"],
        "models": {m: manifest["models"][fi.norm_key(m)]["sha256"] for m in manifest["profiles"][arm]["models"]},
        "model_check": models, "gpu_free_mib": free, "redirected_outputs": redirected,
    }
    fi.write_json(run_dir / "receipt.json", receipt)
    fi.write_json(runs / CURRENT, {"pid": pid, "start_token": token, "arm": arm, "port": port, "run_dir": str(run_dir),
                                   "started_utc": started, "candidate": manifest["id"]})
    say(f"fork server starting: arm {arm}, port {port}, pid {pid}, candidate {manifest['id']}")
    say(f"run directory {run_dir}")
    if wait_s <= 0:
        return 0
    end = time.time() + wait_s
    while time.time() < end:
        now = hooks.start_token(pid)
        if now is None or now != token:
            say("fork_launch: the server exited during start-up; last stderr lines:")
            say(_tail(err_log))
            return 2
        if hooks.http_ok(port):
            say(f"ready: http://127.0.0.1:{port}/health answers")
            return 0
        hooks.sleep(2.0)
    say(f"fork_launch: not ready after {wait_s:.0f} s (the server is still running)")
    return 2


def _tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def stop(root, runs_dir=None, hooks: Hooks | None = None, say=print) -> int:
    hooks = hooks or Hooks()
    runs = runs_root(Path(root).resolve(), runs_dir)
    rec = read_current(runs)
    if rec is None:
        say("no fork server was recorded; nothing was stopped")
        return 0
    token = hooks.start_token(rec["pid"])
    if token is None:
        (runs / CURRENT).unlink(missing_ok=True)
        say("the recorded fork server is not running")
        return 0
    if token != rec.get("start_token"):
        say(f"fork_launch: pid {rec['pid']} now belongs to another process (start {token}, recorded "
            f"{rec.get('start_token')}); nothing was stopped")
        return 1
    hooks.terminate(rec["pid"])
    (runs / CURRENT).unlink(missing_ok=True)
    say(f"fork server stopped (pid {rec['pid']}, arm {rec['arm']})")
    return 0


def status(root, runs_dir=None, hooks: Hooks | None = None) -> dict:
    hooks = hooks or Hooks()
    root = Path(root).resolve()
    rec = read_current(runs_root(root, runs_dir))
    out: dict = {"record": rec, "running": running(rec, hooks)}
    if rec:
        out["ready"] = out["running"] and hooks.http_ok(rec["port"])
        out["identity_problems"] = fi.verify(root, rec["arm"], "skip")
    return out


# ------------------------------------------------------------------------------------------------ promotion plan

ENDPOINT_KEYS = {"server": ("host", "port", "api-key"), "engine": OUTPUT_FLAGS}


def profile_diff(arm_cfg: dict, daily_cfg: dict) -> list[str]:
    """Lines (+ added, - removed, ~ changed) that turn the daily config's settings into the arm's, never touching
    the endpoint (host, port, key) or the files an arm writes. The daily config is not modified."""
    lines = []
    for section in ("server", "engine"):
        a, d = arm_cfg.get(section) or {}, daily_cfg.get(section) or {}
        skip = ENDPOINT_KEYS[section]
        an = {flag_of(k): (k, v) for k, v in a.items() if flag_of(k) not in skip}
        dn = {flag_of(k): (k, v) for k, v in d.items() if flag_of(k) not in skip}
        for flag in sorted(set(an) | set(dn)):
            if flag not in dn:
                lines.append(f"+ {section}.{an[flag][0]}: {json.dumps(an[flag][1])}")
            elif flag not in an:
                lines.append(f"- {section}.{dn[flag][0]}: {json.dumps(dn[flag][1])}")
            elif an[flag][1] != dn[flag][1]:
                lines.append(f"~ {section}.{dn[flag][0]}: {json.dumps(dn[flag][1])} -> {json.dumps(an[flag][1])}")
    return lines


# ------------------------------------------------------------------------------------------------ command line

def split_list(items) -> list[str]:
    return [part for item in items for part in item.split(",") if part.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "stop", "status", "promote-plan"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(fi.default_root()))
        p.add_argument("--runs-dir", help="where run directories go (default: <root>/fork-runs)")
        if name in ("start", "promote-plan"):
            p.add_argument("--arm", required=True, help="a profile in fork/profiles, without .json")
        if name == "start":
            p.add_argument("--disable", action="append", default=[], metavar="NAME,NAME", help="BASALT_FORK_<NAME>=0")
            p.add_argument("--enable", action="append", default=[], metavar="NAME,NAME", help="BASALT_FORK_<NAME>=1")
            p.add_argument("--set", action="append", default=[], metavar="NAME=NUMBER", dest="numbers")
            p.add_argument("--wait", type=float, default=0.0, metavar="SECONDS", help="wait for /health")
            p.add_argument("--models", choices=("full", "stat", "skip"), default="full",
                           help="how the model files are checked: sha256 (default), size and mtime, or not at all")
            p.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
            p.add_argument("--cuda-path")
            p.add_argument("--production-port", type=int, default=PRODUCTION_PORT)
            p.add_argument("--skip-gpu-check", action="store_true")
        if name == "promote-plan":
            p.add_argument("--daily", required=True, help="the daily config the settings would be applied to")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.cmd == "start":
        return start(root, a.arm, disable=split_list(a.disable), enable=split_list(a.enable), numbers=a.numbers,
                     wait_s=a.wait, models=a.models, min_free_mib=a.min_free_mib, cuda_path=a.cuda_path,
                     runs_dir=a.runs_dir, production_port=a.production_port, skip_gpu_check=a.skip_gpu_check)
    if a.cmd == "stop":
        return stop(root, a.runs_dir)
    if a.cmd == "status":
        print(json.dumps(status(root, a.runs_dir), indent=2, sort_keys=True))
        return 0
    path = root / fi.PROFILES_DIR / f"{a.arm}.json"
    if not path.is_file():
        print(f"fork_launch: no profile {path}", file=sys.stderr)
        return 1
    print("\n".join(profile_diff(fi.read_json(path), fi.read_json(Path(a.daily)))) or "no differences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
