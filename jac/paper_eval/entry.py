"""Single-model break-count + output measurement, run under `jac run`.

This file is the *entry program*, and it must be compiled by Jac for the
measurement to be valid. GraphMend injects the deferred-side-effect region
hooks at the `torch.compile(...)` assignment site:

    compiled = torch.compile(model, ...)
    if hasattr(compiled, 'register_forward_pre_hook'):
        compiled.register_forward_pre_hook(__jac_se_region_open__)
        compiled.register_forward_hook(__jac_log_flush_hook__, always_call=True)

The pre-hook is what raises `_gm_se_depth`, and `log_emit` only buffers a
logger call while that depth is above zero -- otherwise it calls the logger
inline and the graph break it was meant to remove survives. So running this
under plain CPython (`python -m ...`) leaves every [Defer] rewrite inert and
every logger-break model reports 0% fixed. Invoke it as the paper does:

    jac run entry.py          # options before the filename

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

# Pin the weights across the two arms.
#
# The builders construct from a config with RANDOM weights, and the seed above
# is what is supposed to make the off and on arms the same model. That holds
# only while both arms draw from the RNG in the same order and the same number
# of times, and claiming a module can change that: Florence-2's remote
# `modeling_florence2.py` consumes the stream differently once GraphMend has
# recompiled it, so the two arms built models with different weights
# (param_hash c89e8f25 vs e756ce2a) and the output comparison reported a
# mismatch that had nothing to do with the rewrite.
#
# So the arms no longer rely on the seed agreeing. The first arm writes its
# state_dict out and the second loads it, which makes "same model" a fact
# rather than an assumption, and leaves the transform as the only difference
# between the two runs. strict=True on purpose: a structural difference between
# the arms must fail loudly here, not be absorbed.
_state = os.environ.get("GM_STATE")
if _state:
    if os.path.exists(_state):
        model.load_state_dict(torch.load(_state, weights_only=True), strict=True)
    else:
        torch.save(model.state_dict(), _state)

# Paper 5.6: full-graph capture for serving frameworks. vLLM requires model
# code to be capturable via `torch.compile(fullgraph=True)`, and SGLang's
# piecewise CUDA graph relies on the same capture; a single graph break fails
# the requirement. `backend="eager"` isolates Dynamo's capture from backend
# compilation, so this measures graph-break elimination and nothing else. It
# needs no GPU and is deterministic: the off arm must fail and the on arm must
# succeed.
#
# Restricted to models whose breaks live on the forward path. For the handful
# whose breaks are on the generation path, `fullgraph=True` over `generate()`
# fails in BOTH arms on the Python-level decode loop, which says nothing about
# GraphMend, so those are reported as skipped rather than as a false result.
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
        out = compiled.generate(**inputs, max_new_tokens=4, do_sample=False,
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

print("GMRESULT " + json.dumps({
    "key": key, "graphs": len(graphs), "breaks": max(0, len(graphs) - 1),
    "out_hash": out_hash, "in_shape": in_shape, "error": None,
}))
