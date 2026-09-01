# GraphMend artifact scripts

Everything needed to run the three claims: the Dockerfile, one script per
claim, and the GraphMend-fixed model sources that C3 measures.

The [top-level README](../README.md) says what each claim is. This file says
what is in this directory and how to run it.

## What is here

```text
artifact/
├── Dockerfile                     # one image, all three claims
├── run.sh                         # container entry point: c1 / c2 / c3
├── run_break_analysis.sh          # C1 and C2
├── verify_break_elimination.py    #   the measurement it drives
├── run_latency_analysis.py        # C3
├── gpu/                           #   bench.py and from_trace.py, used by C3
├── fixed_models/                  # GraphMend's output, next to the originals
├── gen_fixed_models.py            #   regenerate fixed_models/ from the compiler
├── verify_fixed.py                #   check fixed_models/ before timing them
├── minimal_example.py             # smallest correct GraphMend script
├── fetch_typeshed.py              # stdlib stubs the Jac compiler needs
└── jac.toml                       # GraphMend settings for this directory
```

## Setup

### Docker (recommended)

```bash
git clone --recurse-submodules <artifact-url>
cd CGO_AE_GraphMend
docker build -f artifact/Dockerfile -t graphmend .
```

`--recurse-submodules` matters. Without it `jaseci/` is empty and the build
fails. On an existing clone, run `git submodule update --init`.

### Without Docker

```bash
bash scripts/setup.sh
```

This applies [`patches/graphmend.patch`](../patches/graphmend.patch) to the
pinned `jaseci/` submodule and fetches the typeshed stubs. Nothing compiles
without it. Running it again is safe.

You also need Python 3.13 with `torch==2.12.1`, `transformers==4.52.4`,
`numpy==2.4.6` and `torchvision==0.27.1`.

## Quick check

One model, a few minutes:

```bash
docker run --rm --memory=20g graphmend c1 t5-small     # in Docker
bash artifact/run_break_analysis.sh --c1 t5-small      # or directly
```

Expect one row reading 3 breaks found, 3 eliminated, and an identical output.
Exit status is 0 only if the row ran and both arms agreed.

## Running the claims

There is one script per claim. Run the direct commands from the repository
root.

**C1, fixing graph breaks.** Every model runs twice, GraphMend off then on,
and each row reports breaks found, breaks eliminated, and whether the two arms
produced the same output. A row whose arms disagree fails the run.

```bash
docker run --memory=20g graphmend c1
bash artifact/run_break_analysis.sh --c1
bash artifact/run_break_analysis.sh --c1 --offline    # skip rows that download
bash artifact/run_break_analysis.sh --c1 t5-small     # a subset
```

To see why a particular break survives:

```bash
python -m paper_eval.run_why longformer-base-4096 on
```

**C2, full-graph capture.** Checks `torch.compile(fullgraph=True)` on both
arms. It should fail on the original and succeed on the transformed model.
Rows that C1 only partially repairs are excluded. No GPU needed.

```bash
docker run --memory=20g graphmend c2
bash artifact/run_break_analysis.sh --c2
```

Both claims in one pass, which is the default:

```bash
docker run --memory=20g graphmend
bash artifact/run_break_analysis.sh
```

**C3, GPU latency.** Compares the original and the fixed model and reports
cold-start and steady-state latency. Needs an NVIDIA card and downloads
pretrained weights.

```bash
docker run --gpus all --memory=20g graphmend c3
python artifact/run_latency_analysis.py            # the 10-model sample
python artifact/run_latency_analysis.py --full     # every model
python artifact/run_latency_analysis.py --list     # what would run, and why not
```

The default sample is 10 models chosen for coverage: every rule fires at least
once and every input modality appears. `--full` runs every model and takes
roughly four times as long.

Anything after the selector is forwarded, so `graphmend c1 t5-small` and
`graphmend c1 --offline` work. `docker run --rm -it graphmend bash` gets you a
shell in the image.

## Two mistakes that silently disable GraphMend

If a run finishes cleanly but every row reads `N -> N, 0% fixed`, it is almost
always one of these two. Neither prints a warning.

The scripts in this directory already get both right, so this only matters if
you write your own script. [`minimal_example.py`](minimal_example.py) is a
template to copy.

1. **`graphmend_claim_imports` defaults to `false`.** GraphMend is on by
   default, but claiming *imported* third-party code is a separate opt-in.
   Every model here is imported third-party code, so without it both arms
   measure the same thing and every row reads `N -> N, 0% fixed`. There is no
   CLI switch: the key is read from the nearest ancestor `jac.toml`.
2. **The entry program must be run by `jac run`, not `python`.** The `[Defer]`
   rule needs a hook that GraphMend injects at the `torch.compile(...)` call
   site, which only happens in a module Jac compiled. Under plain CPython no
   hook is registered and every logger break survives.

## The fixed model sources

C3 times stock `transformers` against the same tree with GraphMend's output
copied over the modules it transformed. That output is committed under
[`fixed_models/`](fixed_models/), each file next to its `.original.py`, so the
timed window is the model code rather than a compiler run.

The files come from the compiler, and these two scripts produce and check them:

```bash
python artifact/gen_fixed_models.py    # regenerate them, then diff
python artifact/verify_fixed.py        # check they remove the same breaks
```

## Requirements

- Docker, or Python 3.13 with the pinned packages above
- 20 GB of memory. GraphMend compiles the imported `transformers` modeling
  code, and that is what sets the floor, not the weights. Below it the
  container is killed and the row reports `ERR` with exit 137. Check with
  `docker info | grep "Total Memory"`
- 35 GB of disk, for the image, the toolchain cache and the model weights
- No GPU for C1 and C2. C3 needs an NVIDIA card
- Six of the 27 rows download weights or Hub code. `--offline` skips them. Add
  `-v ~/.cache/huggingface:/hf -e HF_HOME=/hf` to keep the downloads between
  runs

<a id="results"></a>

## Known differences from the paper

The runs print their own counts. Read them against the paper's Table 2. Three
differences are expected, so that a reviewer does not read one as a failure:

- **The CPU sweep runs in fp32**, so some absolute break counts are lower than
  Table 2's. Guards conditioned on `dtype == torch.float16` are dead code in
  fp32. On the GPU path in fp16 those rows match.
- **Four rows do not reach zero**: longformer, clap, moe-minicpm and stella.
  What survives is the paper's declared out-of-scope category. A clean sweep on
  any of them would be the anomaly. `run_why` shows which category a surviving
  break falls into.
- **stella needs CUDA with xformers** to reproduce its row, because its breaks
  live behind the unpadding path. Elsewhere it measures a different, smaller
  break set.

To confirm which rule fired on a row, look for the markers GraphMend leaves in
the generated bytecode: `__gm_cond_<n>` for `[Where]`, `__jac_log_emit__` for
`[Defer]`, and `__jac_tensor_eq_assert__` for `[Trap]`.

## Troubleshooting

**Every row reads `N -> N, 0% fixed`.** One of the two mistakes above. The C1
run detects the all-unchanged case and says so.

**`no GraphMend passes under .../jaseci/jac`.** The submodule is not
initialised or the patch is not applied:

```bash
git submodule update --init && bash scripts/setup.sh
```

**A row reads `ERR` and the container exited 137.** The OOM killer. Give Docker
at least 20 GB. The cost is GraphMend compiling the imported modeling code, not
the weights, so a smaller batch does not help.

**`import jaclang` resolves to a released jaclang.** Every released `jaclang`
predates GraphMend. Check with:

```bash
python -c "import jaclang; print(jaclang.__file__)"    # want .../jaseci/jac/jaclang
```

**The first run is very slow and looks hung.** Expected. The cold compiler
bootstrap plus recompiling imported modeling code is minutes, not seconds.
`run_eval` prints each model name as it starts.

**A row shows `MISMATCH` next to the input shape.** The two arms did not see
the same input, so the correctness result on that row means nothing.
