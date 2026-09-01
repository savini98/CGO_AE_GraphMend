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

`scripts/setup.sh` does this for you, and is idempotent:

```bash
bash scripts/setup.sh
```

It applies the patch with `git apply -p1` inside the submodule, then fetches
the typeshed stubs. The patch is generated against the pinned commit and will
not apply to another one, which is the check that the submodule is at the right
revision.

After it runs, the submodule's working tree is intentionally dirty. That is the
patched compiler, and it is the `jaclang` every measurement imports.
