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
from __future__ import annotations
from jaclang.lib.jaclib import __jac_log_flush_hook__, __jac_se_region_open__, __jac_trap_guard__
import hashlib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
if os.environ.get('PAPER_EVAL_DIR'):
    sys.path.insert(0, os.environ['PAPER_EVAL_DIR'])
from paper_eval.registry import MODELS
key = os.environ['GM_MODEL']
spec = MODELS[key]
torch.manual_seed(0)
model, inputs = spec['build']()
model.eval()
_state = os.environ.get('GM_STATE')
if _state:
    if os.path.exists(_state):
        model.load_state_dict(torch.load(_state, weights_only=True), strict=True)
    else:
        torch.save(model.state_dict(), _state)
if os.environ.get('GM_CHECK') == 'fullgraph':
    if spec.get('call') == 'generate':
        print('GMRESULT ' + json.dumps({'key': key, 'fullgraph': 'skipped', 'why': 'breaks are on the generation path; fullgraph over "\n                   "generate() fails in both arms on the decode loop', 'error': None}))
        raise SystemExit(0)
    torch._dynamo.reset()
    _fg = torch.compile(model, backend='eager', fullgraph=True, dynamic=False)
    if hasattr(_fg, 'register_forward_pre_hook'):
        _fg.register_forward_pre_hook(__jac_se_region_open__)
        _fg.register_forward_hook(__jac_log_flush_hook__, always_call=True)
    _fg = __jac_trap_guard__(_fg)
    _ok, _why = (True, None)
    try:
        with torch.no_grad():
            _fg(**inputs)
    except Exception as _exc:
        _ok = False
        _why = f'{type(_exc).__name__}: {str(_exc).strip().splitlines()[0][:160]}'
    print('GMRESULT ' + json.dumps({'key': key, 'fullgraph': 'ok' if _ok else 'failed', 'why': _why, 'error': None}))
    raise SystemExit(0)
graphs = []

def backend(gm: Any, example_inputs: Any) -> object:
    graphs.append(gm)
    return gm.forward
torch._dynamo.reset()
compiled = torch.compile(model, backend=backend, dynamic=False)
if hasattr(compiled, 'register_forward_pre_hook'):
    compiled.register_forward_pre_hook(__jac_se_region_open__)
    compiled.register_forward_hook(__jac_log_flush_hook__, always_call=True)
compiled = __jac_trap_guard__(compiled)
with torch.no_grad():
    if spec.get('call') == 'generate':
        out = compiled.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    else:
        out = compiled(**inputs)
t = out if isinstance(out, torch.Tensor) else None
for attr in ('logits', 'last_hidden_state'):
    if hasattr(out, attr):
        t = getattr(out, attr)
        break
if t is None and isinstance(out, (tuple, list)):
    t = out[0]
out_hash = hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16] if t is not None else None
in_shape = None
for v in inputs.values():
    if hasattr(v, 'shape'):
        in_shape = list(v.shape)
        break
compared = 'tokens' if spec.get('call') == 'generate' else 'logits'
print('GMRESULT ' + json.dumps({'key': key, 'graphs': len(graphs), 'breaks': max(0, len(graphs) - 1), 'out_hash': out_hash, 'in_shape': in_shape, 'compared': compared, 'error': None}))
import ast as _ast
import os as _os
_dump = _os.environ.get('GM_DUMP_DIR')
if _dump:
    from jaclang.runtime.runtime import JacRuntime as _Jac
    _prog = _Jac.get_program()
    _hub = getattr(getattr(_prog, 'mod', None), 'hub', {}) or {}
    _marks = ('__gm_cond_', '__jac_log_emit__', '__jac_tensor_eq_assert__')
    _os.makedirs(_dump, exist_ok=True)
    _written = 0
    _with_ast = 0
    for _path, _mod in _hub.items():
        _gen = getattr(_mod, 'gen', None)
        _pyast = getattr(_gen, 'py_ast', None) if _gen is not None else None
        if not _pyast:
            continue
        _with_ast += 1
        try:
            _src = _ast.unparse(_pyast[0])
        except Exception:
            continue
        _hits = {_k: _src.count(_k) for _k in _marks if _k in _src}
        if not _hits:
            continue
        _base = _os.path.basename(str(_path))
        if _base.endswith('.py'):
            _base = _base[:-3]
        with open(_os.path.join(_dump, _base + '.graphmend.py'), 'w') as _fh:
            _fh.write(_src)
        if _os.path.exists(str(_path)):
            with open(str(_path)) as _orig:
                _txt = _orig.read()
            with open(_os.path.join(_dump, _base + '.original.py'), 'w') as _fh:
                _fh.write(_txt)
        _written += 1
        print('GMDUMP file=' + _base + ' rules=' + str(_hits))
    print('GMDUMP_SUMMARY hub=' + str(len(_hub)) + ' with_ast=' + str(_with_ast) + ' written=' + str(_written) + ' dir=' + _dump)