# GraphMend: CGO 2027 Artifact

**GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2**

## Overview

GraphMend is a compiler technique that eliminates PyTorch `torch.compile` graph
breaks by rewriting Python source before it is lowered to bytecode. It is
implemented as two passes in the Jac/jaclang compiler pipeline and applies three
transformation rules, each guarded by a conservative legality analysis over the
AST, control-flow graph and symbol table:

- **`[Trap]`** -- predicated trap lowering, replacing `if not C: raise E(msg)`
  validation guards with `torch._assert_async`
- **`[Where]`** -- predicated data-dependent control flow, rewriting `if`/`else`
  on tensor values into a single `torch.where` selection
- **`[Defer]`** -- graph-epilogue deferred side effects, buffering `print` and
  logger calls and flushing them after the compiled region

Models run unmodified: `jac run model.py` in place of `python model.py`.

**Start here, then read [`artifact/README.md`](artifact/README.md)**, which is
the artifact guide, the claim-to-command mapping, and the measured results.

## How this artifact is put together

The compiler is not vendored here. The artifact ships the upstream toolchain as
a frozen submodule plus the diff that makes it GraphMend:

| | |
|---|---|
| [`jaseci/`](jaseci) | Upstream `jaseci-labs/jaseci`, a git submodule pinned to `e2b6b9f4bdec510622410f046c8bd5427980c33f` (jaclang 0.36.1) |
| [`patches/graphmend.patch`](patches/graphmend.patch) | Everything GraphMend adds: 203 files, 166 of them new, 0 deletions |
| [`scripts/setup.sh`](scripts/setup.sh) | Checks out the pin, applies the patch, fetches the typeshed stubs. Idempotent |

The split is the point. The patch **is** the contribution, and it is readable:
163 of the 166 new files are the GraphMend passes and their tests, and the 37
modified files are the integration points in the compiler driver, the pass
schedule, the Python front end, the project config and the runtime. A vendored
copy of a 4,700-file compiler tells a reviewer none of that.

## Quick start

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # ~5 min
docker run --rm graphmend-cpu
```

`--recurse-submodules` matters: without it `jaseci/` is empty and the build
fails at `COPY jaseci`. An existing clone is fixed with
`git submodule update --init`.

Natively, without Docker:

```bash
bash scripts/setup.sh        # submodule + patch + typeshed stubs
bash scripts/run_quick.sh    # functional check, ~10-20 min
bash scripts/run_full.sh     # full reproduction
```

**Give Docker at least 12 GB of memory.** `Phi-4-mini-instruct` peaks near
9.7 GB while GraphMend compiles the imported `transformers` modeling code, and
below that the container is SIGKILLed. Check the current limit with
`docker info | grep "Total Memory"`.

Expect `All checks passed.` and exit status 0: four rule-level suites
(18 tests) and five model rows, on CPU, with no network and no weight download.
`--quick` runs a shorter set.

The full 21-row offline sweep, which the default run does not include:

```bash
docker run --rm --entrypoint bash graphmend-cpu \
  -lc 'cd /opt/artifact && python -m paper_eval.run_eval'
```

## What this artifact establishes

The paper's contribution is the elimination of graph breaks by source-level
transformation, with observable behaviour preserved. That is what this artifact
reproduces, and **all of it runs on CPU** -- no GPU, no model weights, and for
21 of the 27 rows, no network.

| | Claim | Paper |
|---|---|---|
| **C1** | Each of the three rules collapses a region that hands TorchDynamo 2+ FX graphs into exactly 1 | §4.3 |
| **C2** | Graph breaks are eliminated at Table 2's fix rates on all 27 benchmark models | §5.1, Table 2 |
| **C3** | The transformed model produces identical output | §5.1 |
| **C4** | `torch.compile(fullgraph=True)` succeeds only after the transformation | §5.6 |

Measured inside the image above, on the paper's environment (torch 2.12.1,
transformers 4.52.4): the 21 offline rows go from **89 breaks to 19**, and the
output fingerprint is identical between the two arms on every row. 25 of the 27
rows match Table 2's fix rate exactly; the two grounding-dino rows read 56%
against 58%, which is the small-config build showing up in a rate. Four rows are
expected *not* to reach zero, and reproduce Table 2 exactly for that reason.

All three rules have real-model demonstrations: `[Defer]` on 17 rows, `[Where]`
on Phi-4-mini-instruct (the Figure 3 worked example), Florence-2 and
Qwen-Audio-Chat, and `[Trap]` on MoLFormer-XL and grounding-dino.

The full row-by-row table, with Table 2's count beside every measured count, is
in [`artifact/README.md`](artifact/README.md#measured-results).

## Latency

The speedups of §5.2 to §5.4 follow from eliminating breaks rather than
standing on their own, and they need the NVIDIA hardware of §5, so they are not
claims a reviewer must reproduce. We ship no recorded traces and no saved
results — an output file we produced is not something a reviewer can check — so
the latency path is a script that measures on your own card:

```bash
bash artifact/gpu/run_reproducible.sh          # needs one CUDA device
```

It gates on what is hardware-independent: graph breaks reaching zero, and
CUDA-graph launches per forward collapsing to one (t5-small 4 → 1, MoLFormer-XL
50 → 1, Phi-4-mini 5 → 1), with a wide 1.5× floor on cold start. Steady state
and throughput are printed but not gated. A different GPU will not land on
Table 2's magnitudes and is not meant to. That split is the declared tolerance:
exact where the result is deterministic, a floor where it depends on the card,
and no gate where any fixed threshold would fail honest hardware.

## What this artifact does not cover

- The 195-model survey of §1 and §5, which selected the benchmark suite.
- Table 2's steady-state and throughput magnitudes. Both are measured on your
  hardware; neither carries a validation threshold.
- Absolute break counts on 8 of the 27 rows read lower than Table 2's, because
  the harness builds small random-weight configs rather than full pretrained
  models. Fix rate, which is what Table 2's `Fixed(%)` column reports, is
  unaffected. Both numbers are listed per row in the artifact guide.

## Layout

| Path | What it is |
|---|---|
| [`artifact/`](artifact/) | The artifact-evaluation package: guide, results, appendix, Dockerfiles, one-command runner |
| [`artifact/run_all.sh`](artifact/run_all.sh) | Kick the tires: PASS/FAIL per check, non-zero exit on failure |
| [`artifact/gpu/run_reproducible.sh`](artifact/gpu/run_reproducible.sh) | The GPU counterpart: measures latency on your own card, gates on the mechanism, non-zero exit on failure |
| [`jaseci/`](jaseci) | Upstream jaclang toolchain, submodule pinned to `e2b6b9f4b` |
| [`patches/graphmend.patch`](patches/graphmend.patch) | The GraphMend diff against that commit: the passes, their integration points, and the rule-level test suites |
| [`scripts/setup.sh`](scripts/setup.sh) | Assembles the two into a working toolchain |
| [`paper_eval/`](paper_eval/) | The reproduction harness: per-model builders, the two-arm runner, the measurement entry program |

## Requirements

Docker is the supported path and pins everything. To run without it you need
Python 3.13, `torch==2.12.1`, `transformers==4.52.4`, `numpy==2.4.6` and
`torchvision==0.27.1`, plus `git` for the submodule and the patch. Then:

```bash
bash scripts/setup.sh
```

That also materializes the typeshed stdlib stubs, which are gitignored.
Without them no Jac compilation works at all, and the failure appears as
`TypeshedUnavailableError` on the first real compile rather than at import.

## License

MIT, see [`LICENSE`](LICENSE).
