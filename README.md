# GraphMend: CGO 2027 Artifact

Artifact for **"GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2."**

GraphMend rewrites Python source before bytecode generation to eliminate
`torch.compile` graph breaks. It implements three transformations: `[Trap]`
for validation guards, `[Where]` for supported data-dependent control flow,
and `[Defer]` for supported `print`/logger side effects.

## What this artifact validates

| Claim | Validation | Hardware |
|---|---|---|
| **C1. Graph-break repair + correctness** | GraphMend removes fixable graph breaks while preserving model behavior. | CPU |
| **C2. Full-graph capture** | Fully repaired models can be captured with `torch.compile(fullgraph=True)`. | CPU |
| **C3. Performance consequence** | Removing graph breaks reduces graph fragmentation and exposes downstream PyTorch/CUDA performance benefits. | NVIDIA GPU |

**C1 and C2 are the core artifact evaluation and require no GPU.**
C3 is hardware dependent; exact latency values are not expected to match the
paper on a different GPU.

## Quick start

```bash
git clone --recurse-submodules <artifact-url>
cd CGO_AE_GraphMend

docker build -f artifact/Dockerfile -t graphmend .
docker run --rm graphmend c1 t5-small          # one row, a few minutes
```

Expected output: a single row reading 3 breaks found, 3 eliminated, and an
identical output fingerprint. Exit status is 0 only if the row ran and both
arms agreed.

One image covers all three claims:

```bash
docker run --rm graphmend                 # C1 then C2, every model
docker run --rm graphmend c1              # C1 only
docker run --rm graphmend c2              # C2 only
docker run --rm --gpus all graphmend c3   # C3, the 10-model sample
```

Arguments after the selector are forwarded, so `graphmend c1 --offline` skips
the six rows that download weights, and `graphmend c1 t5-small biogpt` runs a
subset.

> Give Docker at least **20 GB RAM** and **35 GB disk**. GraphMend compiles the
> imported `transformers` modeling code, and that is what sets the floor rather
> than the weights: `grounding-dino-base` peaks above 11.7 GB. Below the floor
> the container is SIGKILLed and the row reports `ERR` with exit 137.

If the repository was cloned without submodules:

```bash
git submodule update --init
```

## C1 + C2: the core evaluation

```bash
docker run --rm graphmend            # both claims, all 27 models
docker run --rm graphmend --offline  # 21 models, no downloads
```

For each model the run reports graph-break counts before and after GraphMend,
and an output fingerprint for semantic-equivalence checking. Rows print as they
finish. C2 then checks full-graph capture on the rows C1 repairs completely.

Correctness is the load-bearing half: eliminating a graph break while altering
the result is not a fix, so a row whose two arms disagree **fails the run**
rather than counting as a reduction.

The paper reports **107 of 147 graph breaks removed (73%)** and **21 of 27
models fully repaired**. This artifact measures **105 of 139 (76%)**, with the
same 21 models fully repaired, 3 partially, and 3 unrepaired. The row-by-row
comparison with Table 2 is in
[`artifact/README.md`](artifact/README.md#results).

### Known differences from Table 2

Three rows report a lower absolute break count than Table 2 while matching its
**fix rate**, which is what the claim concerns: `chronos-bolt-small` (4 against
6), `clap-htsat-fused` (2 against 4) and `moe-minicpm-x4-base` (11 against 15).
That accounts for the whole 139-against-147 difference.

`grounding-dino` reports 59% against 58% — one break in seventeen.

Break counts are configuration-dependent, which is why six rows are built
through `artifact/gpu/bench.py` at the precision and batch shape Table 2 used.
The BART family is the reason: its overflow guard leads with
`dtype == torch.float16`, a static bool that Dynamo folds away in FP32, so
those rows read 3 breaks instead of 7 unless the dtype matches. Built that way
they reproduce Table 2 exactly.

## C3: GPU performance

```bash
docker run --rm --gpus all graphmend c3          # stage 1: 10-model sample
docker run --rm --gpus all graphmend c3 --full   # stage 2: every model
```

**Two stages.** Stage 1 is a 10-model sample chosen for coverage rather than
speed: every rule fires at least once, every input modality appears (text,
speech, vision, time series), and both ends of the agreement range are present.
Stage 2 runs every model at roughly four times the cost. A reviewer who wants
the claim without the wait runs stage 1; it is the same measurement over fewer
models.

**The compiler is not in the measured window.** Both arms are plain CPython
importing plain Python: the original arm is stock `transformers`, and the fixed
arm is the same tree with GraphMend's *output* copied over the modules it
transformed, from [`artifact/fixed_models/`](artifact/fixed_models/). This is
the paper's own methodology, and it is why an arm takes seconds rather than the
minutes a compile inside the profiled region would add.

The experiment checks the mechanism behind the paper's speedups: after repair,
the model executes with fewer compiled/CUDA-graph regions and launches.

| Model | Before | After |
|---|---:|---:|
| `t5-small` | 4 | 1 |
| `MoLFormer-XL` | 50 | 1 |
| `Phi-4-mini-instruct` | 5 | 1 |

It also reports cold-start and steady-state latency. These depend on the GPU
and batch size, so the expected result is the trend rather than a match to the
RTX 3090, A40 or H100 numbers in the paper. Throughput (Figure 9) is separate:

```bash
bash scripts/run_throughput.sh
```

The fixed sources are compiler output, and that is checkable rather than
asserted — every file ships beside its `.original.py`:

```bash
python artifact/gen_fixed_models.py   # regenerate from the compiler, then diff
python artifact/verify_fixed.py       # gate: they must remove the same breaks
```

The full three-GPU performance campaign is **not required** for the core
artifact evaluation.

## Requirements

**Recommended:** Docker, 20 GB RAM, 35 GB disk.

**C3 only:** one NVIDIA GPU with a CUDA-capable PyTorch environment. The GPU
does not need to match the hardware used in the paper.

Docker is the supported setup and pins the software environment: Python 3.13,
`torch==2.12.1+cu126`, `transformers==4.52.4`, `numpy==2.4.6`,
`torchvision==0.27.1`, `triton==3.7.1`. For native execution:

```bash
bash scripts/setup.sh                                # submodule + patch + stubs
bash artifact/run_break_analysis.sh --c1 t5-small    # one row
bash artifact/run_break_analysis.sh                  # C1 and C2, every model
python artifact/run_latency_analysis.py              # C3, the 10-model sample
```

## Repository layout

| Path | Purpose |
|---|---|
| `artifact/Dockerfile` | The image: one build, all three claims |
| `artifact/run.sh` | Image entry point: `c1`, `c2`, `c3` |
| `artifact/run_break_analysis.sh` | C1 and C2; `--c1` / `--c2` to run one |
| `artifact/run_latency_analysis.py` | C3; `--full` for stage 2 |
| `artifact/fixed_models/` | GraphMend's output, beside the originals, for C3 |
| `paper_eval/` | Per-model evaluation harness |
| `jaseci/` | Pinned upstream Jaseci/jaclang submodule |
| `patches/graphmend.patch` | GraphMend compiler changes (203 files, 166 new) |
| `scripts/setup.sh` | Applies the patch and fetches the typeshed stubs |

There is one script per claim and no second route to any of them, so a reviewer
never has to work out which of two runners measures the thing being claimed.

The Jaseci submodule is pinned to
`e2b6b9f4bdec510622410f046c8bd5427980c33f`, **unpatched**; `scripts/setup.sh`
applies `patches/graphmend.patch` on top, so what GraphMend adds is inspectable
as a diff rather than vendored.

## Scope

This artifact focuses on GraphMend itself: graph-break repair, semantic
preservation, full-graph capture, and the downstream effect of removing graph
breaks. It does not rerun the initial 195-model survey or require the complete
three-GPU latency campaign.

## License

MIT. See [`LICENSE`](LICENSE).
