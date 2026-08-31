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

## Quick start

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # ~5 min
docker run --rm graphmend-cpu
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
  -lc 'cd /opt/jaseci/jac && python -m paper_eval.run_eval'
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

The speedups of §5.2 to §5.4 are a consequence of eliminating breaks rather than
an independent mechanism, and they require the specific NVIDIA hardware of §5.
We provide them as supporting evidence rather than as claims a reviewer must
reproduce.

The published cold-start and steady-state numbers were read from PyTorch
profiler traces, so the traces themselves are shipped here and the strongest
check needs **no GPU and no model download at all**:

```bash
python artifact/gpu/from_trace.py --dir artifact/traces/3090
```

That re-derives Table 2's cold-start column to two decimals on 22 of the 24
trace pairs, and the CUDA-graph launch counts exactly -- MoLFormer-XL 24.71x and
50 launches to 1, bart-large-cnn 21.07x and 30 to 1, t5-small 3.49x and 4 to 1.
For reviewers with an NVIDIA GPU, `artifact/gpu/run_reproducible.sh` re-measures
on device with fixed expected values and a real exit status.

## What this artifact does not cover

- The 195-model survey of §1 and §5, which selected the benchmark suite. This
  artifact supports the results measured on the 27 models it produced.
- §5.7, GraphMend's own compilation overhead.
- End-to-end serving in vLLM (§5.6). C4 covers the requirement vLLM and SGLang
  impose, `torch.compile(fullgraph=True)`, which is the part attributable to
  GraphMend.
- Absolute break counts on 8 of the 27 rows read lower than Table 2's, because
  the harness builds small random-weight configs rather than full pretrained
  models. Fix rate, which is what Table 2's `Fixed(%)` column reports, is
  unaffected. Both numbers are listed per row in the artifact guide.

## Layout

| Path | What it is |
|---|---|
| [`artifact/`](artifact/) | The artifact-evaluation package: guide, results, appendix, Dockerfiles, one-command runner |
| [`artifact/run_all.sh`](artifact/run_all.sh) | Kick the tires: PASS/FAIL per check, non-zero exit on failure |
| [`artifact/gpu/from_trace.py`](artifact/gpu/from_trace.py) | Re-derives the published latency numbers from the shipped profiler traces. No GPU needed |
| [`artifact/traces/3090/`](artifact/traces/3090/) | The 24 profiler trace pairs the latency numbers were read from |
| [`jac/`](jac/) | The jaclang toolchain, including the GraphMend passes |
| [`jac/jaclang/compiler/passes/graphmend/`](jac/jaclang/compiler/passes/graphmend/) | The three transformation rules and their legality analysis |
| [`jac/paper_eval/`](jac/paper_eval/) | The reproduction harness: per-model builders, the two-arm runner, the measurement entry program |
| [`jac/tests/compiler/passes/`](jac/tests/compiler/passes/) | The rule-level graph-count suites |

## Requirements

Docker is the supported path and pins everything. To run without it you need
Python 3.13, `torch==2.12.1`, `transformers==4.52.4`, `numpy==2.4.6` and
`torchvision==0.27.1`, plus the typeshed stubs, which are gitignored and are
materialized by:

```bash
python artifact/fetch_typeshed.py jac
```

Without them no Jac compilation works at all, and the failure appears as
`TypeshedUnavailableError` on the first real compile rather than at import.

## License

MIT, see [`LICENSE`](LICENSE).
