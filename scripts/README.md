# Scripts

Setup, plus the one measurement that the `c1` / `c2` / `c3` commands do not
cover.

```text
scripts/
├── setup.sh           # prepare the toolchain; run this first
├── run_throughput.sh  # end-to-end throughput, paper Figure 9 (needs a GPU)
├── make_archive.sh    # build the archival tarball (for the authors)
└── _common.sh         # shared preamble, sourced by the others
```

## setup.sh

```bash
bash scripts/setup.sh
```

Applies [`patches/graphmend.patch`](../patches/README.md) to the pinned
`jaseci/` submodule and fetches the typeshed stubs. Nothing compiles until it
has run. Running it again is safe.

Docker users do not need this. The image runs it during the build.

## run_throughput.sh

```bash
bash scripts/run_throughput.sh                      # three representative models
bash scripts/run_throughput.sh t5-small Phi-4-mini-instruct   # models you pick
```

End-to-end inference throughput, the paper's Figure 9 and Section 5.4. Needs an
NVIDIA GPU. Generative models produce 100 output tokens with greedy decoding,
as in the paper; encoders are measured in samples per second.

Figure 9 is 24 models across three GPUs. This measures the models you name on
the card you have, so it reproduces the shape of the result rather than the
individual bars: the gain tracks how many CUDA-graph launches the transform
removes, and shrinks as the batch grows.

This is the one measurement with no `docker run graphmend` shortcut. C1, C2 and
C3 are in [`artifact/README.md`](../artifact/README.md).

## make_archive.sh

```bash
bash scripts/make_archive.sh [output.tar.gz]
```

Builds the self-contained tarball for the archival deposit. `git archive` skips
submodule contents, so an archive made with it would ship an empty `jaseci/`;
this flattens the submodule in and records the pinned commit in `ARCHIVE_INFO`.

For the authors preparing a release. Reviewers do not need it.
