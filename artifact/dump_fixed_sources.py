
# ---------------------------------------------------------------------------
# Dump every module GraphMend transformed in THIS run.
#
# entry.py is the right host for it: the claim path is live here, the entry
# program itself is Jac-compiled so the [Defer] hooks exist, and the registry
# scope has already bounded what gets claimed. Reading the module hub after the
# measurement means the sources written out are exactly the ones that produced
# the break counts printed above -- not what compiling a file in isolation
# would have produced, which misses the sites that matter.
#
# Set GM_DUMP_DIR to enable; unset, this block does nothing.
# ---------------------------------------------------------------------------
import ast as _ast
import os as _os

_dump = _os.environ.get("GM_DUMP_DIR")
if _dump:
    from jaclang.runtime.runtime import JacRuntime as _Jac

    _prog = _Jac.get_program()
    _hub = getattr(getattr(_prog, "mod", None), "hub", {}) or {}
    _marks = ("__gm_cond_", "__jac_log_emit__", "__jac_tensor_eq_assert__")
    _os.makedirs(_dump, exist_ok=True)
    _written = 0
    _with_ast = 0
    for _path, _mod in _hub.items():
        _gen = getattr(_mod, "gen", None)
        _pyast = getattr(_gen, "py_ast", None) if _gen is not None else None
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
        if _base.endswith(".py"):
            _base = _base[:-3]
        with open(_os.path.join(_dump, _base + ".graphmend.py"), "w") as _fh:
            _fh.write(_src)
        if _os.path.exists(str(_path)):
            with open(str(_path)) as _orig:
                _txt = _orig.read()
            with open(_os.path.join(_dump, _base + ".original.py"), "w") as _fh:
                _fh.write(_txt)
        _written += 1
        print("GMDUMP file=" + _base + " rules=" + str(_hits))
    print("GMDUMP_SUMMARY hub=" + str(len(_hub)) + " with_ast=" + str(_with_ast)
          + " written=" + str(_written) + " dir=" + _dump)
