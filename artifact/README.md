# GraphMend artifact (CGO 2027)

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

Models run unmodified: `jac run model.py` in place of `python model.py`.

## Contents

- [Getting started](#getting-started)
- [Two ways to measure nothing](#two-ways-to-measure-nothing)
- [Two workflows](#two-workflows)
- [Claims validation](#claims-validation)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Troubleshooting](#troubleshooting)

---

## Getting started

### Prerequisites

- **Docker**, or Python 3.13 with `torch==2.12.1`, `transformers==4.52.4`,
  `numpy==2.4.6`, `torchvision==0.27.1` and `git`
- **20 GB of memory.** GraphMend compiles the imported `transformers` modeling
  code, and that is what sets the floor, not the weights: `grounding-dino-base`
  peaks above 11.7 GB and `chronos-bolt-small` was still growing past 8.8 GB
  when starved. Below the floor the container is SIGKILLed and the row reports
  `ERR` with exit 137. Check with `docker info | grep "Total Memory"`
- **35 GB of disk**, for the image, the toolchain cache and the model weights
- **No GPU for C1 and C2.** C3 needs an NVIDIA card. Six of the 27 rows
  download weights or Hub code; `--offline` skips them

The toolchain is not vendored here. It is upstream `jaseci-labs/jaseci` as a
submodule frozen at `e2b6b9f4bdec510622410f046c8bd5427980c33f` (jaclang 0.36.1)
plus [`../patches/graphmend.patch`](../patches/graphmend.patch), which is
everything GraphMend adds: 203 files, 166 of them new, no deletions.

### Option 1: Docker (recommended)

One image covers all three claims.

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
docker build -f artifact/Dockerfile -t graphmend .

docker run --gpus all --memory=20g graphmend            # C1 then C2
docker run --gpus all --memory=20g graphmend c1         # C1 only
docker run --gpus all --memory=20g graphmend c2         # C2 only
docker run --gpus all --memory=20g graphmend c3         # C3, the 10-row sample
docker run --gpus all --memory=20g graphmend c3 --full  # C3, every row
docker run --rm -it graphmend bash                      # a shell in the image
```

Arguments after the selector are forwarded, so `graphmend c1 t5-small` and
`graphmend c1 --offline` work.

**A GPU is optional for C1 and C2.** Drop `--gpus all` and they measure the
same numbers, with one exception: `stella-en-400M-v5` uses its unpadding path
only when a device is visible, and without one it takes the model card's
fallback and measures a different row. C3 needs a card.

Add `-v ~/.cache/huggingface:/hf -e HF_HOME=/hf` to keep the six network rows'
downloads across runs; without it each run re-fetches them.

`--recurse-submodules` matters: without it `jaseci/` is empty and the build
fails at `COPY jaseci`. Fix an existing clone with `git submodule update
--init`.

### Option 2: Direct installation

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
bash scripts/setup.sh        # applies the patch, fetches the typeshed stubs
bash artifact/run_break_analysis.sh --c1 t5-small   # quick check
```

`setup.sh` is idempotent. It also materializes the typeshed stdlib stubs, which
are gitignored; without them no Jac compilation works at all.

### Verification

```bash
bash artifact/run_break_analysis.sh --c1 t5-small    # one row, a few minutes
```

Expect a single row reading `3` breaks found, `3` eliminated, and an identical
output. Exit status is 0 only if the row ran and its two arms agreed.

---

## Two ways to measure nothing

Both produce a run that completes cleanly, prints a well formed results table,
and reports that GraphMend fixed nothing. Neither prints any diagnostic. The
shipped harness gets both right; a script you write yourself will hit at least
one.

**1. `graphmend_claim_imports` defaults to `false`.** GraphMend is on by
default; claiming *imported third-party code* is a separate opt-in and is not:

```toml
[run]
graphmend = true                 # default: true
graphmend_claim_imports = true   # default: FALSE, and this is the one you need
```

Every model here is imported third-party code (`transformers.models.*`, or
`transformers_modules.*` for Hub remote code), so without the opt-in both arms
measure the same thing and every row reads `N -> N, 0% fixed`.

There is no CLI switch. Both keys are read from the **nearest ancestor
`jac.toml`** — nearest wins, and files further up are not merged in. The same
rule governs `[dev] jaclang_source`, which points at the patched toolchain; a
`jac.toml` that sets the `[run]` keys but omits `[dev]` switches GraphMend on
while running a `jaclang` that predates it. Both shipped `jac.toml` files carry
both stanzas.

**2. The entry program must be run by `jac run`, not by `python`.** `[Defer]`
buffers logger calls, and the hook that drives it is injected at the
`torch.compile(...)` assignment site — a source transformation, so it only
happens in a module Jac compiled. Under plain CPython no hook is registered,
every deferred call runs inline, and every logger break survives, while the
modeling code genuinely was transformed so nothing looks wrong.

[`minimal_example.py`](minimal_example.py) is a template that gets both right.

---

## Two workflows

```bash
bash scripts/run_quick.sh    # functional check, ~10-20 min, CPU, no network
bash scripts/run_full.sh     # full reproduction, hours; GPU steps auto-skip
```

`run_quick.sh` answers the artifact-evaluation question -- all three
transformations execute, graph breaks disappear, outputs stay equal -- without
reproducing the model suite or any performance number. `run_full.sh` runs each
per-claim script below in turn, skipping the two GPU steps where no CUDA device
is visible rather than failing them.

## Claims validation

Each claim has one script. Run from the repository root after `bash scripts/setup.sh`.

### C1: GraphMend automatically repairs fixable FX graph breaks while preserving model behavior

*Paper §5.1, Table 2.* Every model runs twice, GraphMend off then on, through
`jac run` with a `jac.toml` differing only in `graphmend_claim_imports` — so
what is measured is the compiler transforming imported model code, not a
hand-edited model file.

```bash
bash artifact/run_break_analysis.sh --c1              # every model
bash artifact/run_break_analysis.sh --c1 --offline    # skip rows needing downloads
bash artifact/run_break_analysis.sh --c1 t5-small     # a subset, for a quick check
```

**Expected:** a row per model giving breaks found, breaks eliminated, and
whether the two arms produced a bit-identical output. Rows print as they
finish. Not every break is fixable: dynamic-shape operators and `tensor.item()`
calls are out of scope, so some rows are repaired partially or not at all, and
the claim is about the fixable ones.

Correctness is the load-bearing half. Eliminating a graph break while altering
the result is not a fix, so a row whose arms disagree **fails the run** rather
than counting as a successful reduction. A row with no fingerprint to compare
reports `n/a` and is neither passed nor failed.

To see why a particular break survives, which is what distinguishes a correctly
unfixed row from a defect:

```bash
python -m paper_eval.run_why longformer-base-4096 on
```

### C2: GraphMend enables full-graph capture for models whose graph breaks are completely repaired

*Paper §5.6.* This is what C1 buys. vLLM and SGLang require
`torch.compile(fullgraph=True)`, which a single break defeats, so a model whose
breaks are gone should capture where it previously could not.

```bash
bash artifact/run_break_analysis.sh --c2
```

**Expected:** `fullgraph=True` failing on every original model and succeeding on
every transformed one. The check is asymmetric by construction — a both-pass
result is a failure, not a win. Rows that C1 repairs only partially are
excluded: they still contain breaks, so capture must fail for them and that
says nothing about this claim. `backend="eager"` isolates Dynamo's capture from
backend compilation, so it is deterministic and needs no GPU.

Both claims in one pass, which is the default:

```bash
bash artifact/run_break_analysis.sh
```

Exit status is 0 only if every row ran, no row changed its output, and
full-graph capture flipped on every row it was checked for.

### C3: eliminating graph breaks enables downstream cold-start and steady-state gains

*Paper §5.2-5.4, Table 2 and Figure 9.* Removing breaks gives PyTorch larger
regions to optimize, so this measures what C1's transformation buys.

**The compiler is not in the measured window.** Both arms are plain CPython
importing plain Python: the original arm is stock `transformers`, and the fixed
arm is the same tree with GraphMend's *output* copied over the modules it
transformed, from [`fixed_models/`](fixed_models/). That is the paper's own
methodology, and it is why each arm takes seconds rather than the minutes a
compile would add inside the profiled region.

**Two stages.** Stage 1 is a 10-row sample; stage 2 is every row and costs
about four times as long.

```bash
python artifact/run_latency_analysis.py            # stage 1, the 10-row sample
python artifact/run_latency_analysis.py --full     # stage 2, every row
python artifact/run_latency_analysis.py --list     # what would run, and why not
```

The sample is chosen for coverage rather than speed: every rule fires at least
once (`[Trap]` on MoLFormer and grounding-dino-base, `[Where]` on Phi-4-mini
and Qwen-Audio-Chat, `[Defer]` on the rest), every input modality appears
(text, speech, vision, time series), and both ends of the agreement range are
present.

**Expected:** cold start in the single to low double digits and inversely
related to batch size — the same model reads 3.49× at batch 1345 and 15.66× at
8×128 — with steady state within a few percent of 1.0×. Not compared to Table 2
row by row, because the paper's figures hold at its own batch sizes. Needs an
NVIDIA card.

**The fixed sources are compiler output, and that is checkable rather than
asserted.** Every file ships beside its `.original.py`, and

```bash
python artifact/gen_fixed_models.py        # regenerate, then diff
python artifact/verify_fixed.py            # gate: they must remove the same breaks
```

`verify_fixed.py` is a prerequisite, not an extra: a fixed source that leaves
breaks behind still produces a tidy speedup number, and that number means
nothing.

---

## Results

Measured inside `artifact/Dockerfile` on torch 2.12.1 and transformers
4.52.4, so `docker run` executes the same binary that produced them.

| | |
|---|---|
| Rows measured | 27 of 27 |
| Matching Table 2's fix rate | 25 of 27 |
| Matching Table 2's break count | 19 of 27 |
| Output fingerprint identical | 27 of 27 |

The CPU sweep runs in **fp32**, and that is what makes its absolute counts
differ from Table 2's: 122 breaks against 147, with 17 of 26 rows matching
exactly. The largest group that differs is the BART family, whose guard at
`modeling_bart.py:568` leads with `hidden_states.dtype == torch.float16` — a
static bool that Dynamo folds to False in fp32, so those data-dependent breaks
do not exist there. Measured on the GPU path in fp16, with the reference batch
shape, those four rows match Table 2 exactly:

| Model | fp32 CPU row | GPU row (`bench.py --count`) | Table 2 |
|---|---|---|---|
| bart-base | 3 | **7 → 0** | 7 |
| bart-large-cnn | 3 | **7 → 0** | 7 |
| rebel-large | 3 | **7 → 0** | 7 |
| opus-mt-fr-en | 3 | **6 → 0** | 6 |

The fp32 rows are kept as they are on purpose: the correctness claim is that
**FP32** outputs are bit-identical, and that is what `output_ok` checks. The two
paths measure different things deliberately. Both counts are given per row.

### The 21 offline rows

| Model | Breaks, Table 2 | Breaks, here | After | Fixed | Table 2 | Rule |
|---|---|---|---|---|---|---|
| t5-small | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-base | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| t5-3b | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| flan-t5-large | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| inclusively-reformulation-it5 | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-large-v3 | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-small | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| whisper-base | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| bart-large-cnn | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| bart-base | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| rebel-large | 7 | 3 | 0 | 100% | 100% | `[Defer]` |
| biogpt | 2 | 2 | 0 | 100% | 100% | `[Defer]` |
| blenderbot-400M-distill | 3 | 3 | 0 | 100% | 100% | `[Defer]` |
| opus-mt-fr-en | 6 | 3 | 0 | 100% | 100% | `[Defer]` |
| tiny-random-PegasusForCausalLM | 2 | 2 | 0 | 100% | 100% | `[Defer]` |
| layoutlmv3-base | 2 | 2 | 0 | 100% | 100% | `[Defer]` |
| **Phi-4-mini-instruct** | **5** | **5** | **0** | **100%** | **100%** | **`[Where]`** |
| **grounding-dino-tiny** | 17 | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| **grounding-dino-base** | 17 | **16** | **7** | **56%** | 58% | **`[Trap]`** |
| longformer-base-4096 | 5 | 5 | 3 | 40% | 40% | `[Defer]` |
| clap-htsat-fused | 4 | 2 | 2 | 0% | 0% | none applicable |
| **TOTAL** | | **89** | **19** | **78%** | | |

### The 6 network rows

| Model | Breaks, Table 2 | Breaks, here | After | Fixed | Table 2 | Rule |
|---|---|---|---|---|---|---|
| chronos-bolt-small | 6 | 4 | 0 | 100% | 100% | `[Defer]` |
| **MoLFormer-XL-both10pct** | **5** | **5** | **0** | **100%** | **100%** | **`[Trap]`** |
| **Florence-2-large** | **7** | **7** | **0** | **100%** | **100%** | **`[Where]`** |
| **Qwen-Audio-Chat** | **2** | **2** | **0** | **100%** | **100%** | **`[Where]`** |
| moe-minicpm-x4-base | 15 | 11 | 11 | 0% | 0% | none applicable |
| stella-en-400M-v5 | 4 | 4 | 4 | 0% | 0% | none applicable |

**Four rows are expected not to reach zero, and reproduce for that reason.**
longformer at 40%, and clap, moe-minicpm and stella at 0%: those are Table 2's
own numbers, and what survives is the paper's declared out-of-scope category
(`tensor.item()`, dynamic-shape and data-dependent operators). A clean sweep on
any of them would be the anomaly.

**grounding-dino reads 56% where Table 2 reads 58%** — 16 breaks to 7 against
17 to 7. It is the only row whose reported percentage differs, by one break out
of seventeen. `[Trap]` fires as expected and the three surviving sites are all
in the paper's out-of-scope categories. Of the five rows whose absolute count
still differs, the fix rate matches exactly on chronos-bolt-small, clap and
moe-minicpm.

**stella reproduces its 0% row only on CUDA with xformers**, because its breaks
live behind unpadding. The CPU fallback measures a different, smaller break set.

Each rule has real-model demonstrations: `[Defer]` on 17 rows, `[Where]` on
Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, `[Trap]` on MoLFormer-XL
and grounding-dino. To confirm which fired, look for the markers GraphMend
leaves in the generated bytecode: `__gm_cond_<n>` for `[Where]`,
`__jac_log_emit__` for `[Defer]`, `__jac_tensor_eq_assert__` for `[Trap]`.

### Not covered

- The 195-model survey of §1 and §5, which selected the benchmark suite. This
  artifact supports the results measured on the 27 models it produced.
- Table 2's steady-state and throughput magnitudes. The scripts measure both on
  your hardware; neither carries a validation threshold, for the reason given
  under Expected variation.

---

## Repository structure

```
CGO_AE_GraphMend/
├── README.md               # start here
├── jac.toml                # GraphMend keys + the toolchain pointer
├── jaseci/                 # upstream jaclang, submodule pinned at e2b6b9f4b
├── patches/graphmend.patch # everything GraphMend adds (203 files)
├── scripts/                # one script per claim, plus the two workflows
│   ├── setup.sh            #   pin check + patch + typeshed stubs; idempotent
│   ├── run_quick.sh        #   functional workflow, ~10-20 min
│   ├── run_full.sh         #   full reproduction
│   ├── run_latency.sh           # Table 2, Sections 5.2-5.3   (GPU)
│   ├── run_throughput.sh        # Figure 9, Section 5.4       (GPU)
│   ├── run_compiler_overhead.sh # Figure 10, Section 5.7
│   └── make_archive.sh     #   self-contained tarball for the Zenodo deposit
├── paper_eval/             # reproduction harness
│   ├── registry.py         #   per-model builders and inputs
│   ├── run_eval.py         #   two-arm runner (C1)
│   ├── run_fullgraph.py    #   full-graph capture (C2)
│   ├── run_why.py          #   per-break cause reporting
│   ├── run_overhead.py     #   compiler overhead (Section 5.7, not a claim)
│   ├── entry.py            #   the measurement program, run under `jac run`
│   └── _paths.py           #   where the toolchain and the harness live
└── artifact/               # this package
    ├── README.md           #   this file
    ├── appendix.tex        #   artifact appendix, for the paper
    ├── Dockerfile           #   one image, all three claims
    ├── run.sh               #   entry point: c1 / c2 / c3
    ├── run_all.sh          #   PASS/FAIL kick-the-tires runner
    ├── run_break_analysis.sh    # C1 and C2; --c1 / --c2 to run one
    ├── run_latency_analysis.py  # C3; --full for stage 2
    ├── verify_fixed.py          #   gate the fixed sources before timing them
    ├── gen_fixed_models.py      #   regenerate fixed_models/ from the compiler
    ├── fixed_models/            #   GraphMend's output, beside the originals
    ├── verify_break_elimination.py  # the measurement it drives
    ├── jac.toml            #   config for commands typed in this directory
    ├── minimal_example.py  #   smallest correct own-script template
    └── gpu/                #   bench.py, from_trace.py, run_reproducible.sh
```

---

## Troubleshooting

**Every row reads `N -> N, 0% fixed`.** One of the two ways to measure nothing,
above. `run_all.sh` detects the all-unchanged case and says so.

**`no GraphMend passes under .../jaseci/jac`.** The submodule is not
initialised or the patch is not applied:

```bash
git submodule update --init && bash scripts/setup.sh
```

**A row reads `ERR` and the container exited 137.** SIGKILL from the OOM
killer. Give Docker at least 20 GB. The heaviest rows are the ones that set
this floor: `grounding-dino-base` peaks above 11.7 GiB and `chronos-bolt-small`
was still climbing past 8.8 GiB when starved at a lower cap. The cost is
GraphMend compiling the imported modeling code, not the weights, so it does not
shrink with a smaller batch.

**`import jaclang` resolves to a released jaclang.** Every released `jaclang`
predates GraphMend. Check with:

```bash
python -c "import jaclang; print(jaclang.__file__)"    # want .../jaseci/jac/jaclang
```

**The first run is very slow and looks hung.** Expected. The cold Jac compiler
bootstrap plus recompiling imported modeling code is minutes, not seconds.
`run_eval` prints each model name to stderr as it starts.

**A row shows `MISMATCH` next to the input shape.** The two arms did not see
the same input, so `output_ok` on that row means nothing.
