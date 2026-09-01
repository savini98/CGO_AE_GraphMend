# The GraphMend patch

`graphmend.patch` is the contribution of the paper. Everything GraphMend adds
to the Jac compiler is in this one file.

The toolchain itself is not vendored here. It is upstream
[`jaseci-labs/jaseci`](https://github.com/jaseci-labs/jaseci) as the `jaseci/`
submodule, frozen at the commit GraphMend was developed against. The split is
deliberate: reading the patch tells you exactly what this paper changes, which
a vendored copy of a 4,700-file compiler does not.

## What is in it

203 files, 166 of them new, no deletions. By destination:

```text
jac/jaclang/compiler/passes/graphmend/   the three rules and their analyses
jac/jaclang/compiler/{driver,frontend}/  pass schedule and Python front end
jac/jaclang/runtime/                     the deferred side-effect runtime
jac/jaclang/cli/                         the `[run]` config keys
jac/tests/compiler/passes/               152 test files and fixtures
```

The 37 modified files are the integration points. The rest is new code.

## Applying it

Run:

```bash
bash scripts/setup.sh
```

That applies the patch with `git apply -p1` inside the submodule, then fetches
the typeshed stubs. Running it again is safe: it sees an already-patched tree
and stops.

The patch only applies to the pinned commit, so if it applies, the submodule is
at the right revision.

Afterwards `git status` shows the submodule as modified. That is expected: the
patched tree is the compiler, and it is the `jaclang` every measurement
imports.
