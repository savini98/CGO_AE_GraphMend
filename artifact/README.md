# GraphMend artifact (CGO 2027)

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting source before it is compiled,
using three rules: `[Trap]` (validation-guard lowering), `[Where]` (predicated
control flow) and `[Defer]` (side-effect deferral).

Models run unmodified: `jac run model.py` in place of `python model.py`.

## Contents

- [Getting started](#getting-started)
- [Two ways to measure nothing](#two-ways-to-measure-nothing)
- [Claims validation](#claims-validation)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Troubleshooting](#troubleshooting)

---

## Getting started

### Prerequisites

- **Docker**, or Python 3.13 with `torch==2.12.1`, `transformers==4.52.4`,
  `numpy==2.4.6`, `torchvision==0.27.1` and `git`
- **12 GB of memory.** `Phi-4-mini-instruct` peaks near 9.7 GB while GraphMend
  compiles the imported `transformers` modeling code. Below that the container
  is SIGKILLed and the row reports `ERR` with exit 137. Check with
  `docker info | grep "Total Memory"`
- **No GPU, no model weights, and no network** for the claims below

The toolchain is not vendored here. It is upstream `jaseci-labs/jaseci` as a
submodule frozen at `e2b6b9f4bdec510622410f046c8bd5427980c33f` (jaclang 0.36.1)
plus [`../patches/graphmend.patch`](../patches/graphmend.patch), which is
everything GraphMend adds: 203 files, 166 of them new, no deletions.

### Option 1: Docker (recommended)

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .   # ~5 min
docker run --rm graphmend-cpu                                # runs run_all.sh
docker run --rm graphmend-cpu --quick                        # shorter
docker run --rm -it --entrypoint bash graphmend-cpu          # poke around
```

`--recurse-submodules` matters: without it `jaseci/` is empty and the build
fails at `COPY jaseci`. Fix an existing clone with `git submodule update
--init`.

### Option 2: Direct installation

```bash
git clone --recurse-submodules <url> && cd CGO_AE_GraphMend
bash scripts/setup.sh        # applies the patch, fetches the typeshed stubs
bash artifact/run_all.sh     # 4 rule suites + 5 models
```

`setup.sh` is idempotent. It also materializes the typeshed stdlib stubs, which
are gitignored; without them no Jac compilation works at all.

### Verification

```bash
bash artifact/run_all.sh --suites    # fastest real check, ~1-3 min
```

Expect `All checks passed.` and exit status 0. `run_all.sh` prints a
`PASS`/`FAIL` line per check and exits non-zero if any check fails.

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

## Claims validation

The paper's contribution is the elimination of graph breaks by source-level
transformation, with observable behaviour preserved. That is what this artifact
validates. Run from the repository root after `bash scripts/setup.sh`.

### Claim 1: each rule collapses a broken region into one FX graph

*Paper §4.3.* A region that hands TorchDynamo two or more FX graphs
untransformed hands it exactly one after the rewrite.

```bash
bash artifact/run_all.sh --suites
```

**Expected:** `18 passed`, `0 skipped` — `[Trap]` 6, `[Where]` 3, `[Defer]` 7,
import-claiming 2. CPU, ~1-3 min.

### Claim 2: graph breaks are eliminated at Table 2's fix rates

*Paper §5.1, Table 2.* Each model's forward pass runs through a counting
TorchDynamo backend in two isolated subprocesses, GraphMend off then on.

```bash
python -m paper_eval.run_eval                              # 21 offline rows
python -m paper_eval.run_eval Phi-4-mini-instruct          # Figure 3 example
python -m paper_eval.run_eval longformer-base-4096         # a partial row
python -m paper_eval.run_why longformer-base-4096 on       # why a break survives
```

**Expected:** the table below, ending `TOTAL 89 19 78%`. CPU, ~1.5-3 h on a
cold cache; individual rows are minutes. The six rows needing network and
`trust_remote_code` are opt-in and run by name.

### Claim 3: the transformed model produces identical output

*Paper §5.1.* Same command as Claim 2: it compares a SHA-256 fingerprint of the
output tensor between arms, with both arms pinned to the same weights.

**Expected:** `output_ok` is `yes` on **every** row, including rows that fix
nothing.

### Claim 4: full-graph capture for serving

*Paper §5.6.* vLLM and SGLang require `torch.compile(fullgraph=True)`, which a
single graph break defeats.

```bash
python -m paper_eval.run_fullgraph
```

**Expected:** `off failed / on ok` on every row. Asymmetric by construction —
both-pass or both-fail is a FAIL. `backend="eager"` isolates capture from
compilation, so it is deterministic and needs no GPU. ~10 min.

### Latency

*Paper §5.2-5.4.* The speedups follow from eliminating breaks rather than
standing on their own, and they need the NVIDIA hardware of §5, so they are not
claims a reviewer must reproduce. We ship no recorded traces or saved results —
an output file we produced is not something a reviewer can check — so this is a
script that measures on your own card:

```bash
bash artifact/gpu/run_reproducible.sh          # needs one CUDA device
```

It builds each model from real pretrained weights, gives each arm a private
TorchInductor cache so cold start is genuinely cold, and gates only on what is
hardware-independent: graph breaks reaching zero, and CUDA-graph launches per
forward collapsing to one (t5-small 4 → 1, MoLFormer-XL 50 → 1, Phi-4-mini
5 → 1). Cold start is gated with a wide 1.5× floor rather than an expected
value. Steady state and throughput are printed but not gated. A different GPU
will not land on Table 2's magnitudes and is not meant to.

[`gpu/bench.py`](gpu/bench.py) can be called on one model
(`--count`, `--json`, `--paper-batch`, `--save-traces`), and
[`gpu/from_trace.py`](gpu/from_trace.py) reports cold start, steady state and
launch counts from a trace pair `bench.py --save-traces` wrote.

---

## Results

Measured inside `artifact/Dockerfile.cpu` on torch 2.12.1 and transformers
4.52.4, so `docker run` executes the same binary that produced them.

| | |
|---|---|
| Rows measured | 27 of 27 |
| Matching Table 2's fix rate | 25 of 27 |
| Matching Table 2's break count | 19 of 27 |
| Output fingerprint identical | 27 of 27 |

The harness builds each model from a small random-weight config so it is fast
and offline. Graph breaks are structural — they are code paths — so fix rate and
correctness carry over, but a shallower model can expose fewer break sites.
Both counts are given per row.

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

**grounding-dino reads 56% where Table 2 reads 58%** — 16 breaks to 7 is
56.25%. `[Trap]` fires as expected and the three surviving sites are all in the
paper's out-of-scope categories.

**stella reproduces its 0% row only on CUDA with xformers**, because its breaks
live behind unpadding. The CPU fallback measures a different, smaller break set.

Each rule has real-model demonstrations: `[Defer]` on 17 rows, `[Where]` on
Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, `[Trap]` on MoLFormer-XL
and grounding-dino. To confirm which fired, look for the markers GraphMend
leaves in the generated bytecode: `__gm_cond_<n>` for `[Where]`,
`__jac_log_emit__` for `[Defer]`, `__jac_tensor_eq_assert__` for `[Trap]`.

### Not covered

- The 195-model survey of §1 and §5, which selected the benchmark suite.
- §5.7, GraphMend's own compilation overhead.
- Table 2's steady-state and throughput columns. Claim 4 covers the serving
  requirement attributable to GraphMend; the latency script above measures on
  your hardware rather than reproducing the paper's magnitudes.

---

## Repository structure

```
CGO_AE_GraphMend/
├── README.md               # start here
├── jac.toml                # GraphMend keys + the toolchain pointer
├── jaseci/                 # upstream jaclang, submodule pinned at e2b6b9f4b
├── patches/graphmend.patch # everything GraphMend adds (203 files)
├── scripts/setup.sh        # pin check + patch + typeshed stubs; idempotent
├── paper_eval/             # reproduction harness
│   ├── registry.py         #   per-model builders and transform scope
│   ├── run_eval.py         #   two-arm runner (Claims 2 and 3)
│   ├── run_fullgraph.py    #   full-graph capture (Claim 4)
│   ├── run_why.py          #   per-break cause reporting
│   ├── entry.py            #   the measurement program, run under `jac run`
│   └── _paths.py           #   where the toolchain and the harness live
└── artifact/               # this package
    ├── README.md           #   this file
    ├── APPENDIX.md         #   two-page appendix for the paper
    ├── Dockerfile.cpu      #   the supported path; pins torch and transformers
    ├── Dockerfile.cuda     #   GPU image for the latency script
    ├── run_all.sh          #   PASS/FAIL kick-the-tires runner
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
killer. Give Docker at least 12 GB. Measured on aarch64: 9.69 GiB SIGKILLs
Phi-4, 11.65 GiB passes everything. Note that 8.8 GB, the peak resident size of
the same row natively on x86-64, is not enough here — the cost is GraphMend
compiling the imported modeling code, not the weights.

**The rule suites report `skipped`.** torch is not importable by the
interpreter running the toolchain. `run_all.sh` fails on this rather than
passing an empty session.

**The rule suites report `N error` with `DeprecationWarning:
torch.jit.script_method is deprecated`.** Seen on native runs against a torch
CUDA build: the warning comes from inside torch and the Jac test runner
escalates it. No rule is involved. The container is the supported path and
gives `18 passed, 0 skipped`.

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
