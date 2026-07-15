# -*- coding: utf-8 -*-
"""
make_all.py -- Reproduce every figure and calculation in one command.

Code and data for:
    "Statistical Distributions for Relativistic Particles: A Unified Framework
     from Maxwell-Juttner to Non-Extensive Generalizations with Bayesian
     Parameter Inference"  (Annals of Physics, Elsevier).

Discovers and runs, in a deterministic order:
    * figures/fig*.py        (one script per figure -> figures/<name>.pdf + .png)
    * calculations/*.py      (analytic / numeric verification scripts)

Each script is executed in a fresh subprocess (isolation), timed, and its
success/failure recorded.  A summary table is printed at the end.  The process
exits non-zero if any script fails, so it can gate a CI / release run.

Usage:
    python make_all.py                 # run everything
    python make_all.py --list          # just list what would run
    python make_all.py figures         # restrict to a subdirectory (fig*/calc)
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, directory, glob-pattern) in execution order.
SCRIPT_GROUPS = [
    ("figures", os.path.join(HERE, "figures"), "fig*.py"),
    ("calculations", os.path.join(HERE, "calculations"), "*.py"),
]


def discover(only=None):
    """Return an ordered list of (group, absolute_path) script tuples."""
    scripts = []
    for label, directory, pattern in SCRIPT_GROUPS:
        if only is not None and label != only:
            continue
        if not os.path.isdir(directory):
            continue
        found = sorted(glob.glob(os.path.join(directory, pattern)))
        for path in found:
            base = os.path.basename(path)
            # Skip private/helper modules (e.g. __init__.py, _shared.py).
            if base.startswith("_") or base.startswith("."):
                continue
            scripts.append((label, path))
    return scripts


def run_one(path):
    """Run a single script in a subprocess. Return (ok, seconds, tail_output)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, path],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start
        ok = proc.returncode == 0
        tail = ""
        if not ok:
            combined = (proc.stdout or "") + (proc.stderr or "")
            tail = "\n".join(combined.strip().splitlines()[-15:])
        return ok, elapsed, tail
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - start, "TIMEOUT (>3600 s)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def main(argv):
    only = None
    list_only = False
    for arg in argv[1:]:
        if arg in ("--list", "-l"):
            list_only = True
        elif arg in ("figures", "calculations"):
            only = arg
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print(f"Unknown argument: {arg!r}\n{__doc__}")
            return 2

    scripts = discover(only)

    print("=" * 68)
    print(" Reproducing figures & calculations  (make_all.py)")
    print("=" * 68)

    if not scripts:
        print(" 0 scripts found -- nothing to run yet.")
        print(" (Add figures/fig*.py or calculations/*.py, then re-run.)")
        print("=" * 68)
        return 0

    if list_only:
        print(f" {len(scripts)} script(s) would run:")
        for group, path in scripts:
            print(f"   [{group:12s}] {os.path.relpath(path, HERE)}")
        print("=" * 68)
        return 0

    results = []
    for i, (group, path) in enumerate(scripts, 1):
        rel = os.path.relpath(path, HERE)
        print(f" [{i}/{len(scripts)}] running {rel} ...", flush=True)
        ok, elapsed, tail = run_one(path)
        status = "OK  " if ok else "FAIL"
        print(f"        -> {status}  ({elapsed:6.2f} s)")
        if not ok and tail:
            for line in tail.splitlines():
                print(f"           | {line}")
        results.append((rel, ok, elapsed))

    # ---- Summary table -----------------------------------------------------
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    total = sum(t for _, _, t in results)
    width = max((len(r) for r, _, _ in results), default=6)

    print()
    print("=" * 68)
    print(" SUMMARY")
    print("-" * 68)
    print(f" {'script'.ljust(width)}   status   seconds")
    print(f" {'-' * width}   ------   -------")
    for rel, ok, elapsed in results:
        print(f" {rel.ljust(width)}   {'OK' if ok else 'FAIL':6s}   {elapsed:7.2f}")
    print("-" * 68)
    print(f" {len(results)} script(s): {n_ok} OK, {n_fail} FAIL, "
          f"{total:.2f} s total")
    print("=" * 68)

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
