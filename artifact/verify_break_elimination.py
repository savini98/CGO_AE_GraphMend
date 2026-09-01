"""Does GraphMend eliminate graph breaks without changing the output?

    python artifact/verify_break_elimination.py             # every model
    python artifact/verify_break_elimination.py t5-small    # a subset
    python artifact/verify_break_elimination.py --offline   # no downloads

Runs every model twice, GraphMend off then on, and reports what the transform
removed and whether the result survived it. Exit status is non-zero if any row
fails to run, changes its output, or if no row ran at all.

The output comparison is the load-bearing half. Eliminating a graph break while
altering the result is not a fix, so a row whose two arms disagree fails
outright rather than being reported as a successful reduction.

Correctness is THREE-STATE, not two, and the distinction matters. A row can be
`identical` (both arms produced the same fingerprint), `CHANGED` (they differ,
which fails the run), or `n/a` (no fingerprint was available to compare). The
third case is neither passed nor failed: a row with nothing to compare is not
evidence either way.

Both arms go through `jac run` with a jac.toml differing only in
`graphmend_claim_imports`, so what is measured is the compiler's own
transformation of imported model code, not a hand-edited model file. See
`paper_eval/README.md` for why the entry program has to be Jac-compiled.

WHY ROWS TAKE DIFFERENT PATHS. A break count is a property of the code
TorchDynamo actually traces, so it depends on how a model is built and what it
is fed. Most rows are insensitive to that and the small random-weight harness
in `paper_eval/` measures them directly. Five are sensitive and use a
reference-fidelity build in `gpu/bench.py` instead, each for a reason found by
reading the reference scripts rather than guessed:

  * BART family (bart-base, bart-large-cnn, rebel-large, opus-mt-fr-en). The
    guard at modeling_bart.py:568 leads with `dtype == torch.float16`, a static
    Python bool that Dynamo folds to False in fp32, so the data-dependent
    breaks do not exist there: 3 instead of 7. The batch matters independently,
    an `attention_mask` and a decoder input of length ONE.

  * grounding-dino. Not dtype, which the reference pins to fp32 deliberately.
    It needs the real config, a batch built by the model's own processor from a
    real image, and `dynamic` left at its default.

NEITHER NEEDS A GPU. What sets those counts is dtype and input, not the device;
measured on CPU, bart-base reads 3 breaks in fp32 and 7 in fp16.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Every row this can measure. Only the names are used: the published counts
# live in artifact/README.md, and this script reports what GraphMend did
# rather than grading itself against a table.
MODELS = (
    "t5-small", "t5-base", "t5-3b", "flan-t5-large",
    "inclusively-reformulation-it5", "whisper", "whisper-small", "whisper-base",
    "bart", "bart-base", "rebel-large", "opus-mt-fr-en", "biogpt",
    "blenderbot-400M-distill", "PegasusForCausalLM", "layoutlmv3-base",
    "Phi-4-mini-instruct", "grounding-dino", "grounding-dino-base",
    "longformer-base-4096", "clap-htsat-fused", "chronos-bolt-small",
    "MoLFormer-XL-both10pct", "Florence-2", "Qwen-Audio-Chat",
    "moe-minicpm-x4-base", "stella-en-400M-v5",
)

# Rows the small-config harness does not build the way the reference does,
# routed to gpu/bench.py. The value is the key bench.py knows them by. NONE OF
# THESE NEED A GPU: what changes their count is dtype and batch, not device.
REF_BUILD = {
    "bart": "bart-large-cnn", "bart-base": "bart-base",
    "rebel-large": "rebel-large", "opus-mt-fr-en": "opus-mt-fr-en",
    "grounding-dino": "grounding-dino-tiny",
    "grounding-dino-base": "grounding-dino-base",
}

# Rows that download weights or Hub remote code.
NETWORK_ROWS = {"MoLFormer-XL-both10pct", "Florence-2", "Qwen-Audio-Chat",
                "chronos-bolt-small", "moe-minicpm-x4-base",
                "stella-en-400M-v5"}


def harness_dir():
    """The directory containing `paper_eval/`.

    Checked rather than assumed: this package is vendored under `jac/` in the
    standalone artifact repository and sits at the repository root in the
    upstream branch, and hardcoding either one breaks silently on the other.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    for cand in (os.path.join(repo, "jac"), repo):
        if os.path.isdir(os.path.join(cand, "paper_eval")):
            return cand
    return None


def _fail(label, rc, lines):
    """Report a dead subprocess with its reason, not just an empty result."""
    print(f"  {label} failed (exit {rc}).")
    for line in [l for l in lines if l.strip()][-8:]:
        print(f"    {line[:160]}")


def _stream(cmd, cwd, env, label):
    """Run a child, echoing its output live, and return (lines, returncode).

    subprocess.run(capture_output=True) holds everything until the child exits,
    so a sweep that takes tens of minutes prints nothing until the very end and
    a reviewer cannot tell a working run from a hung one. Merge stderr into
    stdout, forward every line as it arrives, and keep the lines for parsing.
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=cwd, env=env)
    lines = []
    for line in p.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        # ROW lines are the per-model results; show them plainly and prefix the
        # rest so the child's chatter is distinguishable from our own output.
        print(line if line.startswith("ROW ") else f"  | {line}", flush=True)
    p.wait()
    return lines, p.returncode


def run_small(keys, jac):
    """Rows measured by `paper_eval.run_eval`, as {key: (before, after, ok)}.

    `ok` is True, False, or None for "no fingerprint to compare".
    """
    if not keys:
        return {}
    # PYTHONUNBUFFERED: stdout here is a pipe, so without it Python
    # block-buffers and a long run emits nothing until it exits.
    env = dict(os.environ, PYTHONPATH=jac, PYTHONUNBUFFERED="1")
    lines, rc = _stream([sys.executable, "-m", "paper_eval.run_eval", *keys],
                        jac, env, "paper_eval.run_eval")
    out = {}
    for line in lines:
        # run_eval prints each row twice: once as `ROW key ...` the moment it
        # is measured, and once in the closing table without the prefix. Accept
        # either, so a result is picked up even if the run is cut short before
        # the table is written. Re-reading the same row is idempotent.
        m = re.match(r"^(?:ROW\s+)?(\S+)\s+(\d+)\s+(\d+)\s+\d+%\s+(\S+)", line)
        if m and m.group(1) in MODELS:
            flag = m.group(4)
            ok = True if flag == "yes" else False if flag == "NO" else None
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)), ok)
    # A non-zero exit with no parsed rows means the harness died: an OOM kill
    # at the memory ceiling, a missing model, an import error. run_eval writes
    # that reason to stderr, so surface it instead of leaving bare ERR rows.
    if rc != 0 and not out:
        _fail("paper_eval.run_eval", rc, lines)
    return out


def run_reference(keys, jac):
    """Rows measured by `gpu/bench.py --count`, same return shape."""
    if not keys:
        return {}
    bench = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gpu", "bench.py")
    env = dict(os.environ, PYTHONPATH=jac, PYTHONUNBUFFERED="1")
    lines, rc = _stream([sys.executable, bench, "--count", "--json",
                         *[REF_BUILD[k] for k in keys]],
                        jac, env, "gpu/bench.py --count")
    out = {}
    inv = {v: k for k, v in REF_BUILD.items()}
    for line in reversed([l for l in lines if l.strip()]):
        try:
            data = json.loads(line)
        except (ValueError, TypeError):
            continue
        for bench_key, r in data.items():
            if bench_key in inv and not r.get("off", {}).get("error"):
                ho, hn = r["off"].get("out_hash"), r["on"].get("out_hash")
                # None when either arm produced no fingerprint, matching the
                # small-config path. Collapsing it into False here reported a
                # CHANGED output, and a failed claim, for a row that simply
                # had nothing to compare.
                ok = None if (ho is None or hn is None) else (ho == hn)
                out[inv[bench_key]] = (r["off"]["breaks"], r["on"]["breaks"], ok)
        break
    if rc != 0 and not out:
        _fail("gpu/bench.py --count", rc, lines)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip rows that download weights or remote code")
    ap.add_argument("models", nargs="*",
                    help="restrict to these rows (default: all). Useful for a "
                         "quick check before the full sweep, which compiles "
                         "every model twice.")
    o = ap.parse_args()

    unknown = [m for m in o.models if m not in MODELS]
    if unknown:
        sys.exit(f"not known rows: {', '.join(unknown)}\n"
                 f"known: {', '.join(sorted(MODELS))}")

    jac = harness_dir()
    if jac is None:
        sys.exit("cannot find paper_eval/ next to this artifact directory")

    skip = set(MODELS) - set(o.models) if o.models else set()
    if o.offline:
        skip |= NETWORK_ROWS

    ref_keys = [k for k in REF_BUILD if k not in skip]
    small_keys = [k for k in MODELS if k not in skip and k not in REF_BUILD]

    print(f"routing {len(small_keys)} row(s) to the small-config harness and "
          f"{len(ref_keys)} to the reference-fidelity build")
    print("this takes a while: every row compiles the model twice\n")

    got = {}
    got.update(run_small(small_keys, jac))
    got.update(run_reference(ref_keys, jac))

    print(f"{'model':32s} {'breaks':>7s} {'fixed':>6s} {'left':>5s} "
          f"{'rate':>6s} {'output':>10s}")
    print("-" * 74)
    errors = out_diff = tot_b = tot_a = measured = 0
    for key in sorted(MODELS):
        if key in skip:
            print(f"{key:32s} {'-':>7s} {'-':>6s} {'-':>5s} {'-':>6s} "
                  f"{'skipped':>10s}")
            continue
        if key not in got:
            print(f"{key:32s} {'ERR':>7s} {'-':>6s} {'-':>5s} {'-':>6s} "
                  f"{'-':>10s}")
            errors += 1
            continue
        b, a, okout = got[key]
        fixed = b - a
        rate = round(100 * fixed / b) if b else 0
        tot_b += b
        tot_a += a
        measured += 1
        if okout is False:
            out_diff += 1
        oc = "identical" if okout else ("CHANGED" if okout is False else "n/a")
        print(f"{key:32s} {b:7d} {fixed:6d} {a:5d} {rate:5d}% {oc:>10s}")

    print("-" * 74)
    tot_fixed = tot_b - tot_a
    tot_rate = round(100 * tot_fixed / tot_b) if tot_b else 0
    print(f"{'TOTAL':32s} {tot_b:7d} {tot_fixed:6d} {tot_a:5d} {tot_rate:5d}%")
    print()

    # A run that measured nothing establishes nothing. Without this, asking for
    # only network rows under --offline skips every one of them and then
    # reports "0 of 0 breaks eliminated ... BIT-IDENTICAL" and exits 0, which
    # is the silently-measures-nothing failure this artifact warns about twice.
    if measured == 0:
        print("No rows were measured, so nothing was established. Check the "
              "model names and the --offline filter.")
        return 1

    if errors:
        print(f"{errors} row(s) failed to run. The claim is not established "
              f"for those.")
    if out_diff:
        print(f"{out_diff} row(s) CHANGED THEIR OUTPUT. Eliminating a graph "
              f"break while altering the result is not a fix, so the claim "
              f"FAILS.")
    if not errors and not out_diff:
        print(f"GraphMend eliminated {tot_fixed} of {tot_b} graph breaks "
              f"({tot_rate}%) across {measured} models, and every row carrying "
              f"an output comparison produced a BIT-IDENTICAL result with the "
              f"transform on and off.")
        print("Breaks that remain are the categories the paper places out of "
              "scope: dynamic-shape operators and tensor.item() calls.")
    return 1 if (errors or out_diff) else 0


if __name__ == "__main__":
    sys.exit(main())
