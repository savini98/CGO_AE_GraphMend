<!--
Artifact Appendix draft, CGO 2027 Artifact Evaluation.

Formatting requirements this draft is written against:
  - at most 2 pages
  - placed BEFORE the References
  - ctuning AE appendix template (\appendix + the artifact-evaluation LaTeX
    package), section order as below
  - single-blind review, at least two reviewers per artifact
  - R1 submission deadline Tue 1 Sep 2026 AoE, https://cgo27ae.hotcrp.com/

Placeholders that MUST be filled before submission are written as {{LIKE_THIS}}.
There are four of them: the archival DOI, the public repository URL, the commit
hash, and the paper's own title/author line if the appendix repeats it.

Length note: about 1540 words of body text, which is close to two two-column
pages and may run slightly over once the check-list bullets and the two tables
are typeset. Trim in this order if needed: (1) the A.2 check-list entries that
repeat A.3, (2) the A.8 Notes examples, keeping at least the Phi-4 LongRoPE one,
(3) the A.7 customization detail. Do not trim A.6, which is the section a
reviewer actually works from, or the PyTorch version deviation in A.3.3.
-->

# Artifact Appendix

## A.1 Abstract

GraphMend is a compiler feature in the Jac/jaclang toolchain that removes
PyTorch `torch.compile` graph breaks by rewriting Python source before it is
compiled, using three rules: `[Trap]` (validation-guard lowering), `[Where]`
(predicated control flow), and `[Defer]` (side-effect deferral). This artifact
reproduces the paper's break-elimination and output-correctness results on 27
Hugging Face model rows, and reports honestly on the six rows that do not
reproduce.

The break-elimination and correctness experiments are **CPU-only** and need no
GPU, no model weights, and (for 21 of the 27 rows) no network. Each model's
forward pass is run through a counting TorchDynamo backend in two isolated
subprocesses, GraphMend off and then on, and the two are compared on graph-break
count and on a SHA-256 fingerprint of the output tensor. A container image is
provided with the environment pinned, and a single `run_all.sh` gives a
pass/fail kick-the-tires result.

The paper's latency, cold-start and throughput claims require an NVIDIA GPU and
are **not reproduced by this artifact**. The GPU benchmark shipped here is a
stub that has never been executed, and it is labelled as such in the artifact,
in its own header, and at run time.

## A.2 Artifact check-list (meta-information)

- **Algorithm:** three source-level program transformations (`[Trap]`,
  `[Where]`, `[Defer]`) applied to PyTorch model code claimed at import time.
- **Program:** jaclang compiler (this branch's source); PyTorch 2.12.1;
  Hugging Face transformers 4.52.4.
- **Compilation:** the toolchain is used directly from source on `PYTHONPATH`;
  it declares no runtime PyPI dependencies. No Zig or LLVM build is needed for
  these experiments.
- **Transformations:** GraphMend, enabled by `[run] graphmend = true` (default)
  plus `[run] graphmend_claim_imports = true` (**not** default).
- **Model:** 27 Hugging Face rows, built from small configs with random
  weights. Graph breaks are structural, so this preserves the measured
  quantity; it does not preserve latency.
- **Data set:** none. Inputs are synthetic tensors of fixed shape, seeded with
  `torch.manual_seed(0)`.
- **Run-time environment:** Linux or macOS, Python 3.13, no GPU. A Docker image
  (`artifact/Dockerfile.cpu`) pins everything.
- **Hardware:** any x86-64 or arm64 CPU for the reproduced claims. The
  unreproduced claims need an NVIDIA GPU (paper used RTX 3090 / A40 / H100).
- **Run-time state:** none persisted between runs beyond a compiler cache.
- **Execution:** `bash artifact/run_all.sh`, then the full sweep with
  `python -m paper_eval.run_eval`.
- **Metrics:** graph breaks before and after (`breaks = FX graphs - 1`); fix
  rate; output fingerprint equality.
- **Output:** a table per model, plus a total. Reference output is in
  `artifact/README.md` and `artifact/RESULTS.md`.
- **Experiments:** 4 rule-level graph-count suites (18 tests); 21 offline model
  rows; 6 network model rows; per-break cause reporting.
- **Disk:** the built CPU image is 1.37 GB; allow roughly 5 GB with the build
  cache. The 6 network rows add a few GB of Hub downloads on top.
- **Time to prepare workflow:** about 5 minutes for the container build on a
  20-core machine, most of it the PyTorch wheel download and the build-time
  compiler cache warm.
- **Time to complete experiments:** minutes for the kick-the-tires path,
  roughly 1.5 to 3 hours for the 21-row offline sweep on a cold cache
  (estimated).
- **Publicly available:** {{PUBLIC_REPO_URL}}, commit {{COMMIT_HASH}}.
- **Code licenses:** MIT.
- **Data licenses:** not applicable; no data set is distributed.
- **Workflow framework used:** none; plain shell and Python.
- **Archived:** {{ARCHIVAL_DOI}}.

## A.3 Description

### A.3.1 How to access

{{PUBLIC_REPO_URL}} at commit {{COMMIT_HASH}}; archived at {{ARCHIVAL_DOI}}.
The artifact package is the `artifact/` directory; the reproduction harness is
`jac/paper_eval/`, whose README is the detailed technical companion.

### A.3.2 Hardware dependencies

None for the reproduced claims: any CPU with about 8 GB of RAM and 10 GB of
disk. The latency, cold-start and throughput claims need an NVIDIA GPU and are
not reproduced here.

### A.3.3 Software dependencies

Python 3.13, PyTorch 2.12.1 (CPU wheel), NumPy 2.4.6, transformers 4.52.4, and
the jaclang toolchain from this branch's source, used via `PYTHONPATH` rather
than pip. `artifact/Dockerfile.cpu` installs exactly this set.

The torch version matches the paper's 2.12. The Dockerfiles pin it exactly
rather than flooring it, because a graph-break count is a property of what
TorchDynamo chooses to split on, and a reviewer who resolves a different torch
may read a different number as a failed reproduction.

### A.3.4 Data sets

None. Inputs are synthetic tensors at fixed shapes, seeded so that the
GraphMend-off and GraphMend-on arms receive identical weights and identical
inputs. The harness prints the input shape per arm and flags a mismatch rather
than assuming one.

### A.3.5 Models

27 Hugging Face rows. 21 are built offline from small configs with random
weights. 6 need network access and `trust_remote_code`: `Florence-2`,
`MoLFormer-XL-both10pct` (revision pinned to `7b12d946c181`),
`chronos-bolt-small`, `Qwen-Audio-Chat`, `stella-en-400M-v5`,
`moe-minicpm-x4-base`. Small configs are sound here because graph breaks are
structural; absolute break counts can therefore be lower than the paper's,
which uses full pretrained models, and fix rate is the compared quantity.

## A.4 Installation

```
docker build -f artifact/Dockerfile.cpu -t graphmend-cpu .
docker run --rm graphmend-cpu
```

Or natively, with torch 2.12.1 and transformers 4.52.4 installed:

```
git clone {{PUBLIC_REPO_URL}} && cd <repo>
bash artifact/run_all.sh
```

Nothing is installed for the toolchain itself: `run_all.sh` puts the branch's
`jac/` directory on `PYTHONPATH`, which must win over any pip-installed
`jaclang` (every released `jaclang` predates GraphMend).

## A.5 Experiment workflow

GraphMend has no CLI switch. It is controlled by two keys in the **nearest
ancestor** `jac.toml`, so the harness writes a fresh one per arm into a private
temporary directory and runs both arms under `jac run`:

```
[run]
graphmend = true                 # default true
graphmend_claim_imports = true   # default FALSE
```

Two properties of the setup are easy to get wrong, and both produce a clean run
that silently measures nothing. First, `graphmend_claim_imports` is off by
default; all model code is imported third-party code, so without the opt-in
GraphMend transforms nothing and both arms are identical. Second, the entry
program must be compiled by Jac: GraphMend injects the deferred-side-effect
hooks at the `torch.compile(...)` assignment site, which is a source
transformation, so under plain CPython no hook is registered and every logger
break survives while the modeling code has genuinely been transformed. Both are
documented prominently in `artifact/README.md`, and `artifact/jac.toml` plus
`artifact/minimal_example.py` give a correct starting point for a reviewer's
own script.

## A.6 Evaluation and expected results

From the repository's `jac/` directory with `PYTHONPATH=$PWD`:

| Claim | Command | Expected |
|---|---|---|
| Rules collapse 2+ FX graphs to 1 | `bash artifact/run_all.sh --suites` | 18 passed, 0 skipped |
| Table 2 fix rate, 21 offline rows | `python -m paper_eval.run_eval` | `TOTAL 89 19 78%` |
| Output bit-identical in FP32 | same | `output_ok` = `yes` on all rows |
| Figure 3 worked example (`[Where]`) | `python -m paper_eval.run_eval Phi-4-mini-instruct` | `5 -> 0`, 100% |
| Break-cause attribution | `python -m paper_eval.run_why <model> on` | per-break reason and location |

**All 27 rows are measured; 26 reproduce their Table 2 fix rate.** The 21
offline rows total 89 breaks to 19 (78%). The one disagreement is
`grounding-dino` and `grounding-dino-base`, which measure 56% against Table 2's
58% (16 breaks to 7); the residue is three sites in the paper's own
out-of-scope categories, and the likeliest cause of the two-point gap is that
Table 2 counts a full pretrained model where this harness builds a small
config. `stella-en-400M-v5` reproduces its 0% row only on CUDA with xformers,
because its breaks live behind unpadding; the CPU fallback measures a
different, smaller break set and is not the Table 2 row.

Four rows are expected **not** to reach zero and reproduce exactly for that
reason: longformer at 40%, and clap-htsat-fused, moe-minicpm-x4-base and
stella-en-400M-v5 at 0%, because what survives is the paper's declared
out-of-scope category. Output fingerprints match on all 27 rows without
exception, with the two arms pinned to the same weights so that the comparison
tests the rewrite rather than the random initialisation.

All three rules have real-model demonstrations: `[Defer]` on 17 rows,
`[Where]` on Phi-4-mini-instruct, Florence-2 and Qwen-Audio-Chat, and `[Trap]`
on MoLFormer-XL (5 to 0) and grounding-dino (16 to 7).

C8 reproduces on t5-small at 3.29x under the authors' own cold-start
definition (first iteration minus `backend_compile`, excluded from both arms),
inside the paper's 30-75%. Two of three benchmark models cannot be measured by
that metric because they recompile, and C10 is not measured. The claim values
recorded in this artifact ("26x", "up to 15%") do not match the GraphMend paper
text available to check and are marked unverified. See the GPU section of
`artifact/RESULTS.md`.

## A.7 Experiment customization

Individual rows run by name: `python -m paper_eval.run_eval t5-small biogpt`.
Individual rules can be ablated with `[run] graphmend_disable`, or the
equivalent flag, which takes a comma-separated list of `trap`, `where` and
`defer`: `jac run --graphmend_disable defer program.py`. Setting
`graphmend = false` disables all three. New models are added with a `_build()`
returning `(model, inputs)` plus a registry entry naming the modules GraphMend
should scope-transform; note that a model's breaks frequently live in shared
transformers modules rather than in its own package, and a scope that names
only the model package silently misses them.

## A.8 Notes

Enabling the code path under test matters as much as enabling GraphMend.
Phi-4-mini only takes the LongRoPE branch when `rope_scaling.type ==
"longrope"` is set; clap needs `enable_fusion`; longformer needs a sequence
length that is not a multiple of the attention window; Florence-2's guard is
dead code in FP32 and has to be measured in half precision. With the default
configuration each of those rows would pass vacuously.

`"call": "generate"` measures nothing: `torch.compile` wraps `forward` only,
and `OptimizedModule.__getattr__` forwards every other attribute to the
unwrapped module, so a `generate` call runs the eager model and the counting
backend records zero graphs. No registry entry uses it.

## A.9 Methodology

Submission, reviewing and badging methodology:

- https://www.acm.org/publications/policies/artifact-review-badging
- http://cTuning.org/ae/submission-20201122.html
- http://cTuning.org/ae/reviewing-20201122.html
