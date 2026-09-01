"""Single-model break-count and output measurement, run under `jac run`.

This is the entry program, and it must be compiled by Jac: GraphMend injects
the deferred-side-effect hooks at the `torch.compile(...)` assignment site, so
under plain CPython no hook is registered and every [Defer] rewrite is inert.

Prints one `GMRESULT <json>` line.
"""
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

if os.environ.get("PAPER_EVAL_DIR"):
    sys.path.insert(0, os.environ["PAPER_EVAL_DIR"])

from paper_eval.registry import MODELS

key = os.environ["GM_MODEL"]
spec = MODELS[key]

torch.manual_seed(0)
model, inputs = spec["build"]()
model.eval()

# Pin the weights across the two arms. The builders construct from random
# weights, and a shared seed only makes the arms identical while both draw from
# the RNG identically, which claiming a module can change. The first arm writes
# its state_dict and the second loads it, so the transform is the only
# difference between the runs. strict=True: a structural mismatch must fail
# loudly rather than be absorbed.
_state = os.environ.get("GM_STATE")
if _state:
    if os.path.exists(_state):
        model.load_state_dict(torch.load(_state, weights_only=True), strict=True)
    else:
        torch.save(model.state_dict(), _state)

# Paper 5.6: full-graph capture. `backend="eager"` isolates Dynamo's capture
# from backend compilation, so this measures graph-break elimination alone, on
# CPU and deterministically. Restricted to models whose breaks are on the
# forward path: where they are on the generation path, fullgraph=True over
# generate() fails in both arms on the decode loop and says nothing about
# GraphMend, so those rows are skipped.
if os.environ.get("GM_CHECK") == "fullgraph":
    if spec.get("call") == "generate":
        print("GMRESULT " + json.dumps({
            "key": key, "fullgraph": "skipped",
            "why": "breaks are on the generation path; fullgraph over "
                   "generate() fails in both arms on the decode loop",
            "error": None}))
        raise SystemExit(0)
    torch._dynamo.reset()
    _fg = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    _ok, _why = True, None
    try:
        with torch.no_grad():
            _fg(**inputs)
    except Exception as _exc:
        _ok = False
        _why = f"{type(_exc).__name__}: {str(_exc).strip().splitlines()[0][:160]}"
    print("GMRESULT " + json.dumps({
        "key": key, "fullgraph": "ok" if _ok else "failed",
        "why": _why, "error": None}))
    raise SystemExit(0)

graphs = []


def backend(gm, example_inputs):
    graphs.append(gm)
    return gm.forward


torch._dynamo.reset()
compiled = torch.compile(model, backend=backend, dynamic=False)
with torch.no_grad():
    if spec.get("call") == "generate":
        # Some models carry their breaks on the generation path rather than in a
        # bare forward: Florence-2's guards are shape comparisons Dynamo folds at
        # trace time, so a forward-only row reports 0 breaks and measures nothing.
        # Paper 5.1 claims "greedy-decoded token sequences are identical for
        # every generative model", so the comparison below is over a decoded
        # SEQUENCE rather than a single step. 16 tokens rather than 4: a
        # divergence introduced by a transform usually appears a few steps in,
        # once the sampled prefix differs, and a 4-token window can miss it.
        out = compiled.generate(**inputs, max_new_tokens=16, do_sample=False,
                                num_beams=1)
    else:
        out = compiled(**inputs)

# Output fingerprint for the correctness comparison (logits / last_hidden_state).
t = out if isinstance(out, torch.Tensor) else None
for attr in ("logits", "last_hidden_state"):
    if hasattr(out, attr):
        t = getattr(out, attr)
        break
if t is None and isinstance(out, (tuple, list)):
    t = out[0]
out_hash = (hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
            if t is not None else None)

# The paper states the original and transformed runs share a batch size and
# identical inputs; emitting the shape per arm makes that checkable rather than
# asserted, and the off/on rows must agree.
in_shape = None
for v in inputs.values():
    if hasattr(v, "shape"):
        in_shape = list(v.shape)
        break

# Which quantity `out_hash` covers, so the results table can say how many rows
# compare decoded token sequences rather than asserting it. For a generate row
# the returned tensor IS the token ids, so the hash is over the sequence.
compared = "tokens" if spec.get("call") == "generate" else "logits"

print("GMRESULT " + json.dumps({
    "key": key, "graphs": len(graphs), "breaks": max(0, len(graphs) - 1),
    "out_hash": out_hash, "in_shape": in_shape, "compared": compared,
    "error": None,
}))
