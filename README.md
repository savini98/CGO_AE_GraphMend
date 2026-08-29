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

## What it does not

Stated here rather than left for a reviewer to discover:

- **One row disagrees.** grounding-dino measures 56% against Table 2's 58%.
- **Throughput (C10) is not measured.**
- **The claim values in the tables are marked UNVERIFIED.** Figures quoted in
  earlier drafts of this artifact do not match the paper text available to
  check. They need reconciling against the submission.
- **`from_pretrained` does not work** under `graphmend_claim_imports`, because
  Python ingestion drops `global` and `nonlocal` statements. The measured rows
  are unaffected: they build via `from_config`, and no measured modeling file
  uses a function-scope `global`.
- **The CUDA image has never run against a physical GPU.** It builds and its
  environment is verified at build time; the machine holding the GPU used for
  this work has Docker without the NVIDIA Container Toolkit.
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
