#!/usr/bin/env python3
"""Run the existing local regression checks, without a model or PC connection."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SHARED = ["fork/tools", "tools/test_ppl_gate.py", "tests/program/adaptive_transfer_failure_test.py"]
QUICK = [
    "serve/tests/test_control_admission.py",
    "serve/tests/test_prompt_boundary.py",
    "serve/tests/test_engine_pipe_cleanup.py",
    "serve/tests/test_hardening_vision.py",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quick", "full"), nargs="?", default="quick")
    args = parser.parse_args(argv)
    targets = (QUICK if args.mode == "quick" else ["serve/tests", "fork/scheduler/tests"]) + SHARED
    # -n auto spreads the tests over local cores (pytest-xdist, from requirements-test.txt); without it, run serially.
    workers = ["-n", "auto"] if _has_xdist() else []
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "-W", "error", *workers, *targets], cwd=ROOT)


def _has_xdist():
    return importlib.util.find_spec("xdist") is not None


if __name__ == "__main__":
    raise SystemExit(main())
