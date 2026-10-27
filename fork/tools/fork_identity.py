#!/usr/bin/env python3
"""Identity manifest of a fork candidate: which bytes a launch will run, and how to get the previous ones back.

  fork_identity.py stamp   [--root R]      write R/fork-source.json from a Git checkout (run before copying the tree
                                           to a machine that has no .git)
  fork_identity.py record  [--root R]      after a build: sha256 of every file in R/bin, the source commit and dirty
                                           flag, a digest of the source tree, each profile, and the model files the
                                           profiles name -> R/fork-identity.json; the candidate is also archived under
                                           R/fork-candidates/ and R/fork-rollback.json names the previous one
  fork_identity.py verify  [--root R]      exit 1 unless the files on disk still match the manifest
  fork_identity.py restore --to ID         put an archived candidate's binaries and source back, then verify
  fork_identity.py show    [--root R]      the manifest and the rollback record

The engine is R/bin/basalt (the CMake runtime output directory is the source tree's bin/, not the build directory).
Only the standard library; Python 3.9 or newer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

SCHEMA = 1
MANIFEST = "fork-identity.json"
SOURCE_STAMP = "fork-source.json"
ROLLBACK = "fork-rollback.json"
CANDIDATES = "fork-candidates"
RUNS = "fork-runs"
BIN_DIR = "bin"                       # CMake: RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
ENGINE = "basalt.exe" if os.name == "nt" else "basalt"
PROFILES_DIR = "fork/profiles"
# What a launch executes or reads from the tree: the engine sources, the Python server, the profiles and the tools.
SOURCE_ROOTS = ("CMakeLists.txt", "cmake", "include", "src", "serve", "fork", "tools", "tests", "data")
SKIP_DIRS = {"__pycache__", ".pytest_cache", RUNS, CANDIDATES}
GENERATED = {MANIFEST, SOURCE_STAMP, ROLLBACK}
HASH_BLOCK = 8 << 20


class IdentityError(Exception):
    """The identity cannot be established or recorded."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                return h.hexdigest()
            h.update(block)


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm_key(path) -> str:
    """The comparison form of a file path: absolute, normalised, case-folded where the OS folds case."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


# ------------------------------------------------------------------------------------------------ source tree

def tree_files(root: Path) -> list[str]:
    """The files of the identity scope, as sorted relative POSIX paths."""
    out = []
    for name in SOURCE_ROOTS:
        p = root / name
        if p.is_file():
            out.append(name)
        elif p.is_dir():
            for dirpath, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for f in files:
                    if not f.endswith(".pyc"):
                        out.append(Path(dirpath, f).relative_to(root).as_posix())
    return sorted(out)


def tree_hashes(root: Path) -> dict[str, str]:
    return {rel: sha256_file(root / rel) for rel in tree_files(root)}


def digest_of(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(hashes):
        h.update(f"{rel}\0{hashes[rel]}\n".encode())
    return h.hexdigest()


def _git(root: Path, *args: str):
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def git_state(root: Path):
    """{commit, dirty, dirty_files} when root is itself the top of a Git checkout; None otherwise (a copy nested in
    some other repository does not borrow that repository's commit)."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or not top.strip():
        return None
    try:
        if not os.path.samefile(top.strip(), root):
            return None
    except OSError:
        return None
    commit = (_git(root, "rev-parse", "HEAD") or "").strip()
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if len(commit) != 40 or status is None:
        return None
    changed = []
    for line in status.splitlines():
        name = line[3:].strip().strip('"').split(" -> ")[-1]
        if name in GENERATED or name.split("/")[0] in (RUNS, CANDIDATES):
            continue
        changed.append(name)
    return {"commit": commit, "dirty": bool(changed), "dirty_files": changed[:20]}


def exclude_generated(root: Path) -> None:
    """Keep the files this tool writes out of `git add -A` in a checkout (the repository's local exclude list)."""
    out = _git(root, "rev-parse", "--git-path", "info/exclude")
    if out is None or git_state(root) is None:
        return
    path = Path(root, out.strip())
    have = path.read_text().splitlines() if path.is_file() else []
    want = [f"/{name}" for name in sorted(GENERATED)] + [f"/{RUNS}/", f"/{CANDIDATES}/"]
    missing = [w for w in want if w not in have]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write("".join(w + "\n" for w in missing))


def source_identity(root: Path) -> dict:
    """The source half of the identity: commit, dirty flag and the digest of the identity scope. From Git when the
    tree is a checkout, else from fork-source.json, whose tree digest must match the files present."""
    hashes = tree_hashes(root)
    tree = digest_of(hashes)
    git = git_state(root)
    if git is not None:
        return {"from": "git", "commit": git["commit"], "dirty": git["dirty"], "dirty_files": git["dirty_files"],
                "tree_sha256": tree, "files": len(hashes)}
    stamp_path = root / SOURCE_STAMP
    if not stamp_path.is_file():
        raise IdentityError(f"{root} is not a Git checkout and has no {SOURCE_STAMP}: run `fork_identity.py stamp` "
                            "in the checkout and copy the file into the tree")
    try:
        stamp = read_json(stamp_path)
        ok = (isinstance(stamp.get("commit"), str) and len(stamp["commit"]) == 40
              and isinstance(stamp.get("tree_sha256"), str) and isinstance(stamp.get("dirty"), bool))
    except (OSError, ValueError, AttributeError):
        ok = False
    if not ok:
        raise IdentityError(f"{SOURCE_STAMP} is not a valid stamp")
    if stamp["tree_sha256"] != tree:
        raise IdentityError(f"the source files differ from {SOURCE_STAMP} (tree {tree[:12]}, stamped "
                            f"{stamp['tree_sha256'][:12]}): the copy is not the stamped checkout")
    return {"from": "stamp", "commit": stamp["commit"], "dirty": stamp["dirty"],
            "dirty_files": list(stamp.get("dirty_files", []))[:20], "tree_sha256": tree, "files": len(hashes)}


def stamp(root: Path, out: Path | None = None) -> dict:
    root = Path(root).resolve()
    git = git_state(root)
    if git is None:
        raise IdentityError(f"{root} is not the top of a Git checkout")
    hashes = tree_hashes(root)
    data = {"schema": SCHEMA, "commit": git["commit"], "dirty": git["dirty"], "dirty_files": git["dirty_files"],
            "tree_sha256": digest_of(hashes), "files": len(hashes), "created_utc": utc_now()}
    exclude_generated(root)
    write_json(out or root / SOURCE_STAMP, data)
    return data


# ------------------------------------------------------------------------------------------------ binaries, profiles

def binaries(root: Path) -> dict[str, dict]:
    out = {}
    base = root / BIN_DIR
    if base.is_dir():
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                p = Path(dirpath, f)
                out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    return out


def resolve_input(value: str, base: Path) -> str:
    p = Path(os.path.expanduser(str(value)))
    return os.path.abspath(p if p.is_absolute() else base / p)


def profile_files(root: Path) -> list[Path]:
    return sorted((root / PROFILES_DIR).glob("*.json"))


def profile_models(profile: Path, cfg: dict, root: Path) -> list[str]:
    """The model files a profile makes the launch read: server.model (relative to the profile, as the server reads
    it) and engine.ple-gguf (relative to the working directory, the tree root)."""
    out = []
    server, engine = cfg.get("server") or {}, cfg.get("engine") or {}
    if server.get("model"):
        out.append(resolve_input(server["model"], profile.parent))
    for key, val in engine.items():
        if str(key).strip().lower().replace("_", "-").removeprefix("--") == "ple-gguf" and isinstance(val, str):
            out.append(resolve_input(val, root))
    return out


def load_arms(root: Path) -> dict[str, dict]:
    arms = {}
    for p in profile_files(root):
        try:
            cfg = read_json(p)
        except (OSError, ValueError) as e:
            raise IdentityError(f"{p.name}: {e}") from None
        arms[p.stem] = {"profile": p.relative_to(root).as_posix(), "sha256": sha256_file(p),
                        "port": (cfg.get("server") or {}).get("port"), "models": profile_models(p, cfg, root)}
    return arms


def hash_models(paths, previous: dict, reuse: bool, say) -> dict[str, dict]:
    out = {}
    for path in sorted(set(paths), key=norm_key):
        if not os.path.isfile(path):
            raise IdentityError(f"model file not found: {path}")
        st = os.stat(path)
        old = previous.get(norm_key(path))
        if reuse and old and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns:
            digest, how = old["sha256"], "reused: size and mtime unchanged since the earlier full hash"
        else:
            say(f"hashing {path} ({st.st_size / 2**30:.1f} GiB)")
            digest, how = sha256_file(path), "full"
        out[norm_key(path)] = {"path": path, "sha256": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                               "hash": how}
    return out


def build_info(build_dir: Path | None) -> dict:
    if build_dir is None:
        return {"dir": None}
    info = {"dir": str(build_dir)}
    cache = Path(build_dir) / "CMakeCache.txt"
    if cache.is_file():
        info["cmake_cache_sha256"] = sha256_file(cache)
        want = ("CMAKE_BUILD_TYPE", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
                "CMAKE_GENERATOR", "CUDAToolkit_ROOT")
        vars_ = {}
        for line in cache.read_text(errors="replace").splitlines():
            name, sep, rest = line.partition(":")
            if sep and name in want:
                vars_[name] = rest.partition("=")[2]
        info["cmake"] = vars_
    return info


def candidate_id(src: dict, bins: dict) -> str:
    eng = bins[f"{BIN_DIR}/{ENGINE}"]["sha256"]
    return f"c-{src['commit'][:10]}-{src['tree_sha256'][:8]}-{eng[:8]}"


# ------------------------------------------------------------------------------------------------ record / archive

def _archive(root: Path, manifest: dict, hashes: dict[str, str]) -> Path:
    cdir = root / CANDIDATES / manifest["id"]
    if not (root / CANDIDATES / ".gitignore").exists():
        (root / CANDIDATES).mkdir(parents=True, exist_ok=True)
        (root / CANDIDATES / ".gitignore").write_text("*\n")
    for rel in manifest["binaries"]:
        dst = cdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
    with tarfile.open(cdir / "source.tar.gz", "w:gz") as tar:
        for rel in sorted(hashes):
            tar.add(root / rel, arcname=rel, recursive=False)
    write_json(cdir / "tree.json", hashes)
    write_json(cdir / MANIFEST, manifest)
    return cdir


def _rollback_record(root: Path, manifest: dict, previous: dict | None) -> dict:
    py = "python"
    rec = {"schema": SCHEMA, "updated_utc": utc_now(),
           "current": {"id": manifest["id"], "manifest": f"{CANDIDATES}/{manifest['id']}/{MANIFEST}"},
           "previous": None, "restore": None}
    if previous and previous.get("id") != manifest["id"]:
        archived = (root / CANDIDATES / previous["id"] / "source.tar.gz").is_file()
        rec["previous"] = {"id": previous["id"], "commit": previous["source"]["commit"],
                           "recorded_utc": previous.get("recorded_utc"), "archived": archived,
                           "manifest": f"{CANDIDATES}/{previous['id']}/{MANIFEST}"}
        rec["restore"] = (f"{py} fork/tools/fork_identity.py restore --to {previous['id']}" if archived else
                          "not restorable: that candidate was recorded before it was archived; rebuild its commit")
    elif (root / ROLLBACK).is_file():
        try:
            old = read_json(root / ROLLBACK)
            rec["previous"], rec["restore"] = old.get("previous"), old.get("restore")
        except (OSError, ValueError):
            pass
    return rec


def record(root, build_dir=None, extra_models=(), allow_dirty=False, reuse_model_hashes=False, expect=None,
           say=print) -> dict:
    root = Path(root).resolve()
    src = source_identity(root)
    exclude_generated(root)
    if src["dirty"] and not allow_dirty:
        raise IdentityError("the source tree has uncommitted changes (" + ", ".join(src["dirty_files"][:5])
                            + "): commit them, or pass --allow-dirty to record a candidate that no commit names")
    bins = binaries(root)
    if f"{BIN_DIR}/{ENGINE}" not in bins:
        raise IdentityError(f"{root / BIN_DIR / ENGINE} does not exist: build first (CMake writes it to the source "
                            "tree's bin/, not to the build directory)")
    arms = load_arms(root)
    previous_manifest, previous_models = None, {}
    if (root / MANIFEST).is_file():
        try:
            previous_manifest = read_json(root / MANIFEST)
            previous_models = previous_manifest.get("models", {})
        except (OSError, ValueError):
            pass
    paths = [m for a in arms.values() for m in a["models"]] + [resolve_input(m, root) for m in extra_models]
    models = hash_models(paths, previous_models, reuse_model_hashes, say)
    for want_path, want_sha in (expect or {}).items():
        got = models.get(norm_key(want_path))
        if got is None:
            raise IdentityError(f"--expect names {want_path}, which no profile reads (add it with --model)")
        if got["sha256"] != want_sha.lower():
            raise IdentityError(f"{want_path}: sha256 {got['sha256']} is not the expected {want_sha.lower()}")
    hashes = tree_hashes(root)
    manifest = {"schema": SCHEMA, "recorded_utc": utc_now(), "engine": f"{BIN_DIR}/{ENGINE}", "source": src,
                "binaries": bins, "profiles": arms, "models": models, "build": build_info(build_dir),
                "python": sys.version.split()[0], "platform": platform.platform()}
    manifest["id"] = candidate_id(src, bins)
    _archive(root, manifest, hashes)
    write_json(root / MANIFEST, manifest)
    write_json(root / ROLLBACK, _rollback_record(root, manifest, previous_manifest))
    return manifest


# ------------------------------------------------------------------------------------------------ verify

def verify(root, arm: str | None = None, models: str = "full", say=None) -> list[str]:
    """The ways the files on disk differ from the manifest (empty: they match). `models`: full (hash every file the
    arm reads), stat (size and modification time only) or skip."""
    root = Path(root).resolve()
    mp = root / MANIFEST
    if not mp.is_file():
        return [f"{MANIFEST} is missing: run fork_identity.py record after a build"]
    try:
        m = read_json(mp)
        m["id"], m["engine"], m["source"], m["binaries"], m["profiles"], m["models"]  # noqa: B018 - shape check
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"{MANIFEST} is unreadable: {e}"]
    problems = []
    live_bins = binaries(root)
    for rel, rec in m["binaries"].items():
        got = live_bins.get(rel)
        if got is None:
            problems.append(f"binary {rel} is missing (recorded {rec['sha256'][:12]})")
        elif got["sha256"] != rec["sha256"]:
            problems.append(f"binary {rel} is {got['sha256'][:12]}, the manifest records {rec['sha256'][:12]}")
    for rel in sorted(set(live_bins) - set(m["binaries"])):
        problems.append(f"binary {rel} is not in the manifest")
    if m["engine"] not in m["binaries"]:
        problems.append(f"the manifest names no engine binary {m['engine']}")
    try:
        src = source_identity(root)
    except IdentityError as e:
        problems.append(f"source: {e}")
    else:
        rec = m["source"]
        if src["commit"] != rec["commit"]:
            problems.append(f"source commit is {src['commit'][:12]}, the manifest records {rec['commit'][:12]}")
        if src["tree_sha256"] != rec["tree_sha256"]:
            problems.append(f"source tree is {src['tree_sha256'][:12]}, the manifest records {rec['tree_sha256'][:12]}")
        if src["dirty"] != rec["dirty"]:
            problems.append(f"source dirty flag is {src['dirty']}, the manifest records {rec['dirty']}")
    arms = [arm] if arm else sorted(m["profiles"])
    seen: dict[str, str] = {}
    for name in arms:
        prof = m["profiles"].get(name)
        if prof is None:
            problems.append(f"profile {name} is not in the manifest (known: {', '.join(sorted(m['profiles']))})")
            continue
        p = root / prof["profile"]
        if not p.is_file():
            problems.append(f"profile {prof['profile']} is missing")
        elif sha256_file(p) != prof["sha256"]:
            problems.append(f"profile {prof['profile']} changed since the manifest was recorded")
        if models == "skip":
            continue
        for path in prof["models"]:
            rec = m["models"].get(norm_key(path))
            if rec is None:
                problems.append(f"model {path} is not in the manifest")
            elif not os.path.isfile(path):
                problems.append(f"model {path} is missing")
            elif os.path.getsize(path) != rec["size"]:
                problems.append(f"model {path} has {os.path.getsize(path)} bytes, the manifest records {rec['size']}")
            elif models == "stat":
                if os.stat(path).st_mtime_ns != rec["mtime_ns"]:
                    problems.append(f"model {path} was modified after it was hashed")
            else:
                if path not in seen:
                    if say:
                        say(f"hashing {path}")
                    seen[path] = sha256_file(path)
                if seen[path] != rec["sha256"]:
                    problems.append(f"model {path} does not match the recorded sha256 {rec['sha256'][:12]}")
    return problems


# ------------------------------------------------------------------------------------------------ restore

def _restore_source(root: Path, cdir: Path, to_id: str, m: dict, want: dict[str, str]) -> None:
    """Put the archived source files back and write the fork-source.json that names their commit (a copy of the
    tree; a Git checkout is moved with `git checkout` instead, never overwritten)."""
    staging = Path(tempfile.mkdtemp(prefix="restore-", dir=root))
    try:
        with tarfile.open(cdir / "source.tar.gz") as tar:
            if sorted(tar.getnames()) != sorted(want):
                raise IdentityError(f"the archive of {to_id} is damaged: its source file list changed")
            for member in tar.getmembers():
                if member.name.startswith(("/", "..")) or ".." in Path(member.name).parts or not member.isreg():
                    raise IdentityError(f"the archive of {to_id} holds an unsafe entry {member.name!r}")
            try:
                tar.extractall(staging, filter="data")
            except TypeError:                       # Python before the extraction filters: the names are checked above
                tar.extractall(staging)
        for rel, digest in want.items():
            if sha256_file(staging / rel) != digest:
                raise IdentityError(f"the archive of {to_id} is damaged: {rel}")
        for rel in set(tree_files(root)) - set(want):
            (root / rel).unlink()
        for rel in want:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / rel, dst)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    src = m["source"]
    write_json(root / SOURCE_STAMP, {"schema": SCHEMA, "commit": src["commit"], "dirty": src["dirty"],
                                     "dirty_files": src.get("dirty_files", []), "tree_sha256": src["tree_sha256"],
                                     "files": src["files"], "created_utc": utc_now(), "restored_from": to_id})


def restore(root, to_id: str, say=print) -> list[str]:
    """Put an archived candidate back: its binaries (bin/ is replaced), its manifest and, in a copy of the tree, its
    source files. A Git checkout keeps its working tree: the result then names the commit to check out. Returns what
    verify (models by size and time) finds afterwards."""
    root = Path(root).resolve()
    cdir = root / CANDIDATES / to_id
    try:
        m = read_json(cdir / MANIFEST)
        want = read_json(cdir / "tree.json")
    except (OSError, ValueError):
        raise IdentityError(f"no archived candidate {to_id} under {root / CANDIDATES}") from None
    for rel, rec in m["binaries"].items():
        if not (cdir / rel).is_file() or sha256_file(cdir / rel) != rec["sha256"]:
            raise IdentityError(f"the archive of {to_id} is damaged: {rel}")
    checkout = git_state(root) is not None
    if not checkout:
        _restore_source(root, cdir, to_id, m, want)
    for rel in set(binaries(root)) - set(m["binaries"]):
        (root / rel).unlink()
    for rel in m["binaries"]:
        tmp = root / (rel + ".restoring")
        shutil.copy2(cdir / rel, tmp)
        os.replace(tmp, root / rel)
    prior = read_json(root / MANIFEST) if (root / MANIFEST).is_file() else None
    write_json(root / MANIFEST, m)
    write_json(root / ROLLBACK, _rollback_record(root, m, prior))
    say(f"restored {to_id}")
    problems = verify(root, models="stat")
    if checkout and any(p.startswith("source") for p in problems):
        problems.append(f"the source of {to_id} is commit {m['source']['commit']}: git checkout --detach "
                        f"{m['source']['commit'][:12]}")
    return problems


# ------------------------------------------------------------------------------------------------ command line

def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stamp", "record", "verify", "restore", "show"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(default_root()), help="the source tree (default: this checkout)")
        if name == "stamp":
            p.add_argument("--out")
        if name == "record":
            p.add_argument("--build", help="the CMake build directory, for its cache values")
            p.add_argument("--allow-dirty", action="store_true")
            p.add_argument("--reuse-model-hashes", action="store_true",
                           help="keep an earlier full hash of a model whose size and mtime are unchanged")
            p.add_argument("--model", action="append", default=[], help="hash this file too")
            p.add_argument("--expect", action="append", default=[], metavar="PATH=SHA256",
                           help="refuse unless this model file has this sha256")
        if name == "verify":
            p.add_argument("--arm", help="check only this profile's files (default: all)")
            p.add_argument("--models", choices=("full", "stat", "skip"), default="full")
        if name == "restore":
            p.add_argument("--to", required=True, metavar="ID")
    a = ap.parse_args(argv)
    root = Path(a.root)
    try:
        if a.cmd == "stamp":
            d = stamp(root, Path(a.out) if a.out else None)
            print(f"stamped {d['commit'][:12]}{' (dirty)' if d['dirty'] else ''}, tree {d['tree_sha256'][:12]}, "
                  f"{d['files']} files")
            return 0
        if a.cmd == "record":
            expect = {}
            for item in a.expect:
                path, sep, digest = item.rpartition("=")
                if not sep or len(digest) != 64:
                    ap.error("--expect wants PATH=SHA256")
                expect[path] = digest
            m = record(root, Path(a.build) if a.build else None, a.model, a.allow_dirty, a.reuse_model_hashes, expect,
                       say=lambda s: print(s, file=sys.stderr, flush=True))
            print(f"recorded {m['id']}: engine {m['binaries'][m['engine']]['sha256'][:12]}, commit "
                  f"{m['source']['commit'][:12]}{' (dirty)' if m['source']['dirty'] else ''}, "
                  f"{len(m['models'])} model file(s)")
            return 0
        if a.cmd == "verify":
            problems = verify(root, a.arm, a.models, say=lambda s: print(s, file=sys.stderr, flush=True))
        elif a.cmd == "restore":
            problems = restore(root, a.to)
        else:
            for name in (MANIFEST, ROLLBACK):
                if (root / name).is_file():
                    print(f"== {name}\n{(root / name).read_text()}")
            return 0
    except IdentityError as e:
        print(f"fork_identity: {e}", file=sys.stderr)
        return 1
    for line in problems:
        print(f"MISMATCH {line}", file=sys.stderr)
    if problems:
        return 1
    print("identity matches the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
