# GraphMend, CGO 2027 artifact

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

This repository is the artifact. It is self-contained: the toolchain is
vendored here rather than pulled from elsewhere, so nothing outside this clone
and a pinned set of PyPI wheels is needed.

**Start here, then read [`artifact/README.md`](artifact/README.md).** That file
is the detailed guide, including two ways to accidentally measure nothing.
[`artifact/RESULTS.md`](artifact/RESULTS.md) is the row-by-row record.

## Kick the tires

```bash
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # ~5 min
docker run --rm graphmend-cpu
```

Expect `All checks passed.` and exit status 0: four rule-level suites
(18 tests) and five model rows, on CPU, with no network and no weight download.
`--quick` runs a shorter set.

The full 21-row offline sweep, which the default run does not include:

```bash
docker run --rm --entrypoint bash graphmend-cpu \
  -lc 'cd /opt/jaseci/jac && python -m paper_eval.run_eval'
```

## What this artifact establishes

**Break elimination reproduces on all 27 rows of the paper's Table 2**, measured
inside the image above, on the paper's environment (torch 2.12.1, transformers
4.52.4). The 21 offline rows go from **89 breaks to 19**, a 78% reduction, and
the output fingerprint is identical between the two arms on every row.

All three rules have real-model demonstrations: `[Defer]` on 17 rows, `[Where]`
on Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, `[Trap]` on MoLFormer-XL
and grounding-dino.

**Cold start (C8) reproduces on the three GPU models measured**, using the
authors' own metric (first iteration minus compilation, excluded from both
arms): t5-small 3.29x, MoLFormer-XL 2.22x, Phi-4-mini 5.92x, with CUDA-graph
launches per forward going 4 to 1, 50 to 1, and 5 to 1 respectively.

It also reproduces through a second, independent path: the authors' own
`fx-graph-research` MoLFormer script, run on an RTX 3090 under the paper's
torch 2.12.1, matches their recorded warm timing to about one percent (118.5 ms
against 117.8 ms), confirms 5 graph breaks in the original arm and 0 in the
fixed arm, and reports a **6.48x** cold-start ratio with both arms compiling
from an equally fresh cache.

**Throughput (C10) is measured, and what it shows is a mechanism rather than a
single number.** The gain tracks how many CUDA-graph launches the transform
eliminates, and shrinks as the batch grows:

| model | launches | batch 1 | batch 8 | large batch |
|---|---|---|---|---|
| t5-small | 4 -> 1 | 0.984x | 1.000x | 1.001x (b256) |
| Phi-4-mini | 5 -> 1 | 1.008x | 1.005x | 1.002x (b16) |
| **MoLFormer-XL** | **50 -> 1** | **1.70x** | 1.017x | 1.009x (b512) |

MoLFormer sheds 49 of its 50 launches and gains 70% at batch 1, and nothing by
batch 512 where compute dominates. Models shedding three or four launches gain
nothing at any batch size. The batch-1 figure is four runs (1.729x, 1.616x,
1.692x, 1.755x) against a noise band of about half a percent.

**The CUDA image is GPU-verified, and the GPU claims reproduce from it.**
Measured inside the image on an RTX 3090: break counts t5-small 3 to 0 and
MoLFormer-XL 5 to 0, and cold start on t5-small at **3.82x** (2018.0 ms to
528.5 ms, launches 4 to 1) against 3.29x measured natively on the same machine.
It does not need the NVIDIA Container Toolkit: torch's cu126 wheel vendors its
own CUDA runtime, so passing the driver and the device nodes through directly is
enough, with no root and no Docker restart. The exact invocation, including the
second libcuda mount that Triton needs, is in the GPU section of
[`artifact/RESULTS.md`](artifact/RESULTS.md).

## What it does not

Stated here rather than left for a reviewer to discover:

- **One row disagrees.** grounding-dino measures 56% against Table 2's 58%.
- **Cold start is cache-sensitive, and the benchmark is opinionated about it.**
  Compilation dominates the first measured window, so both arms have to compile
  with the same TorchInductor cache state or the ratio moves by a factor of
  several. `gpu/bench.py` gives each arm its own cache directory; a run that
  lets them share the default does not compare like for like. See the GPU
  section of [artifact/RESULTS.md](artifact/RESULTS.md).
- The repository's own test suite is **272 passed, 2 failed**. Both failures are
  cache-hit assertions that reproduce identically on the merge-base commit, so
  they pre-date this work.

## Layout

| Path | What it is |
|---|---|
| [`artifact/`](artifact/) | The artifact-evaluation package: guide, results, appendix, Dockerfiles, one-command runner |
| [`jac/`](jac/) | The vendored jaclang toolchain, including the GraphMend passes |
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
