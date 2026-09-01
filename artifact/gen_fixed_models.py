#!/usr/bin/env python3
"""Regenerate artifact/fixed_models/ from the compiler, row by row.

    python artifact/gen_fixed_models.py                 # every fixable row
    python artifact/gen_fixed_models.py t5-small bart   # a subset

The sources under artifact/fixed_models/ are GraphMend's own output, and this
is what produced them. It exists so that claim is checkable rather than
asserted: regenerate, then diff against what is committed.

Each row runs through the ordinary claim path with GM_DUMP_DIR set. The dump
block at the end of paper_eval/entry.py reads the module hub after the
measurement, so what is written is exactly the code that produced that row's
break counts, not what compiling a file in isolation would have produced.

Rows whose breaks are not fixable have nothing to dump and are skipped.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from paper_eval.registry import MODELS  # noqa: E402

# The three rows GraphMend does not repair: no transformed source exists, so a
# dump would write nothing and the empty directory would read as a defect.
UNFIXED = {"clap-htsat-fused", "moe-minicpm-x4-base", "stella-en-400M-v5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="rows to regenerate (default: all)")
    ap.add_argument("--out", default=os.path.join(HERE, "fixed_models"))
    o = ap.parse_args()

    rows = o.models or [k for k in MODELS if k not in UNFIXED]
    unknown = [r for r in rows if r not in MODELS]
    if unknown:
        sys.exit(f"not known rows: {', '.join(unknown)}")

    os.makedirs(o.out, exist_ok=True)
    failed = []
    for row in rows:
        dest = os.path.join(o.out, row)
        os.makedirs(dest, exist_ok=True)
        env = dict(os.environ, GM_DUMP_DIR=dest, PYTHONPATH=REPO,
                   PYTHONUNBUFFERED="1")
        print(f"==> {row}", flush=True)
        p = subprocess.run([sys.executable, "-m", "paper_eval.run_eval", row],
                           cwd=REPO, env=env)
        written = len([f for f in os.listdir(dest) if f.endswith(".graphmend.py")])
        print(f"    {written} transformed module(s) -> {dest}", flush=True)
        if p.returncode != 0 or written == 0:
            failed.append(row)

    if failed:
        print(f"\n{len(failed)} row(s) produced nothing: {', '.join(failed)}")
        return 1
    print(f"\n{len(rows)} row(s) regenerated under {o.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
