# GraphMend: CGO 2027 Artifact

Artifact for **“GraphMend: Code Transformations for Fixing Graph Breaks in PyTorch 2.”**

GraphMend automatically rewrites Python source to eliminate `torch.compile`
graph breaks while preserving model behavior.

## What this artifact validates

| Claim | What is reproduced |
|---|---|
| **C1** | GraphMend fixes graph breaks while preserving the outputs of the original model. |
| **C2** | Models for which GraphMend eliminates all graph breaks can be captured using `torch.compile(fullgraph=True)`, a prerequisite for serving frameworks that rely on full-graph capture. |
| **C3** | Fixing graph breaks improves PyTorch 2 execution performance by reducing cold-start and steady-state forward-pass latency. |

C1 and C2 can be run on CPU or GPU. C3 requires an NVIDIA GPU.

## Quick start

```bash
git clone --recurse-submodules <artifact-url>
cd CGO_AE_GraphMend

docker build -f artifact/Dockerfile -t graphmend .
```

For a quick functional check on one model:

```bash
docker run --rm graphmend c1 t5-small
```

A successful run exits with status 0.

> Docker should have at least **20 GB RAM** and **100 GB disk** (60 GB is
> the floor). The image is 7.3 GB, one build leaves about 7 GB of build
> cache, and the model weights are roughly 15 GB across the three claims.

The Zenodo deposit also carries the built image, so the claims can be run
without a build:

```bash
docker load < graphmend-image.tar.gz
docker run --rm --gpus all --memory=20g graphmend:cgo2027 c1 t5-small
```

If the repository was cloned without submodules:

```bash
git submodule update --init
```

## C1: Fixing Graph Breaks

```bash
docker run --rm graphmend c1
```

This experiment runs GraphMend on the evaluation models and reports:

- graph breaks before and after GraphMend, and
- whether the transformed output matches the original output.

The paper reports that GraphMend removes **107 of 147 graph breaks (73%)** and
fully repairs **21 of 27 models**.

A few absolute break counts may differ because some breaks depend on dtype and
model configuration. These known differences are listed in
[`artifact/README.md`](artifact/README.md#results).
## C2: Full-Graph Capture

```bash
docker run --rm graphmend c2
```

This experiment checks whether models fully fixed by GraphMend can be
captured with:

```python
torch.compile(fullgraph=True)
```

The expected result is that models fully fixed by GraphMend succeed in
full-graph capture.

## C3: GPU Performance

C3 requires one NVIDIA GPU:

```bash
docker run --rm --gpus all graphmend c3
```

This experiment compares the original and GraphMend-fixed models and reports
cold-start and steady-state latency.

Because performance depends on the GPU and system configuration, exact latency
and speedup values may vary. The expected result is the same overall performance
trend after graph-break elimination.

By default, C3 runs on a representative 10-model subset to keep the evaluation
time practical. For each model, the experiment profiles 10 forward-pass
iterations and generates five profiler traces. To run the same experiment on
all evaluation models, use:

```bash
docker run --rm --gpus all graphmend c3 --full
```

## Requirements

For C1 and C2:

- Docker
- 20 GB RAM
- 100 GB disk (60 GB minimum)

For C3:

- the above requirements
- one NVIDIA GPU with CUDA support
- to reproduce the full hardware evaluation from the paper, access to an
  **RTX 3090, A40, and H100** is required. Running the full model set on all
  three GPUs can take substantially longer than the default C3 evaluation.

Docker pins the software environment used by the artifact.

## Repository structure

```text
.
├── artifact/              # Artifact scripts, Dockerfile, and detailed results
├── paper_eval/            # Per-model evaluation harness
├── jaseci/                # Pinned Jaseci/jaclang source
└── patches/
    └── graphmend.patch     # GraphMend compiler changes
```

## License

MIT. See [`LICENSE`](LICENSE).
