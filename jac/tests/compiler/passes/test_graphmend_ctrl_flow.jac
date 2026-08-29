"""Predicated control flow: rewrite shapes and per-condition declines."""
import os;
import tempfile;
import from tests.support {
    graphmend_break_kinds as _break_kinds,
    graphmend_compile_path as _compile_path,
    graphmend_pysrc as _pysrc
}

"""A Python-source model with data-dependent return branches.

Written to a temp file at test time rather than checked in as a .py fixture:
the repository does not accept new Python files, and the point of the fixture
is only that GraphMend ingests .py source, which a generated file exercises
just as well."""
glob PY_WHERE_SRC = '''
import torch


@torch.compile()
def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        return x * (b * -1)
    else:
        return x * (b * -2)
''';

"""Compile the generated .py model above and return its Python lowering."""
def _py_fixture_src(gm: bool) -> str {
    with tempfile.TemporaryDirectory() as tmp {
        path = os.path.join(tmp, "py_where_return.py");
        with open(path, "w") as fh {
            fh.write(PY_WHERE_SRC);
        }
        return _compile_path(path, gm);
    }
}

"""Qwen-Audio-Chat's audio-fusion guard, from `QWenModel.forward`.

Kept verbatim in shape because it is the Table 2 row that separates the two
rules. [Where] must refuse it: the true branch dereferences `audio_info`, which
is None on exactly the runs the false branch serves, and it yields a Python list
of per-segment tensors against a `None` false operand. [Precond] carries it
instead, because that same dereference happens on every path through the branch
before any observable effect, which makes `audio_info is not None` a necessary
condition of reaching the end of the branch."""
glob PY_AUDIO_FUSION_SRC = '''
import torch
from typing import Dict


class M(torch.nn.Module):
    def forward(self, input_ids, past_key_values=None, audio_info=None):
        if past_key_values is None and torch.any(input_ids == self.start_id):
            bos_pos = torch.where(input_ids == self.start_id)
            eos_pos = torch.where(input_ids == self.start_id + 1)
            audio_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
            if isinstance(audio_info, Dict):
                audios = self.audio.encode(audio_info["input_audios"])
            else:
                audios = torch.concat([a["input_audios"] for a in audio_info])
        else:
            audios = None
        return audios
''';

"""A model whose torch import sits behind an availability guard."""
glob PY_GUARDED_IMPORT_SRC = '''
def _avail():
    return True


if _avail():
    import torch


def _compute_long(config, device, **rope_kwargs):
    if len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: `**rope_kwargs` should be unset")
    return torch.arange(0, 8).float().to(device)


ROPE_INIT_FUNCTIONS = {"long": _compute_long}


@torch.compile()
def update(self_obj, position_ids, device):
    seq_len = torch.max(position_ids) + 1
    if seq_len > 8:
        if not hasattr(self_obj, "long_inv_freq"):
            self_obj.long_inv_freq = self_obj.rope_init_fn(self_obj.config, device)
        self_obj.register_buffer("inv_freq", self_obj.long_inv_freq, persistent=False)
    else:
        self_obj.original_inv_freq = self_obj.original_inv_freq.to(device)
        self_obj.register_buffer("inv_freq", self_obj.original_inv_freq, persistent=False)
    return self_obj.inv_freq
''';

"""Write a .py model to a temp file and return its generated Python source."""
def _compile_path_of_py(src: str, name: str) -> str {
    with tempfile.TemporaryDirectory() as tmp {
        path = os.path.join(tmp, name);
        with open(path, "w") as fh {
            fh.write(src);
        }
        return _compile_path(path, True);
    }
}

"""The same, with the graphmend transforms off."""
def _compile_path_of_py_off(src: str, name: str) -> str {
    with tempfile.TemporaryDirectory() as tmp {
        path = os.path.join(tmp, name);
        with open(path, "w") as fh {
            fh.write(src);
        }
        return _compile_path(path, False);
    }
}

"""Import torch or skip the calling test."""
def _torch -> any {
    try {
        import torch;
        return torch;
    } except ImportError {
        testskip("torch not installed");
    }
}

"""Compile a fixture and exec its generated Python with @torch.compile inert.

The caller drives the raw function through a counting backend itself, exactly
as test_graphmend_where_integration does."""
def _exec_fixture(fixture: str, gm: bool) -> dict {
    torch = _torch();
    src = _pysrc(fixture, gm);
    ns: dict = {};
    orig_compile = torch.compile;
    def _shim(*a: any, **k: any) -> any {
        if a {
            return a[0];
        }
        def _wrap(fn: any) -> any {
            return fn;
        }
        return _wrap;
    }
    torch.compile = _shim;
    try {
        exec(compile(src, fixture, "exec"), ns);
    } finally {
        torch.compile = orig_compile;
    }
    return ns;
}

"""Count the contiguous graphs the backend receives for one call of `f`.

`grad` picks the gradient mode the call is traced under; the predication rule
specialises on it, so a row is only meaningful next to the mode it was taken
in."""
def _graph_count(fixture: str, gm: bool, args: tuple, grad: bool = False) -> int {
    torch = _torch();
    ns = _exec_fixture(fixture, gm);
    graphs: list = [];
    def _backend(g: any, inputs: any) -> any {
        graphs.append(g);
        return g.forward;
    }
    torch._dynamo.reset();
    fn = torch.compile(ns["f"], backend=_backend, dynamic=False);
    if grad {
        with torch.enable_grad() {
            fn(*args);
        }
    } else {
        with torch.no_grad() {
            fn(*args);
        }
    }
    return len(graphs);
}

test "an else-less body that only assigns becomes a predicated update" {
    src = _pysrc("where_update_plain.jac", True);
    # the false operand is the variable's value before the branch
    assert "__gm_cond_0 = torch.isnan(x).any()" in src;
    assert "out = torch.where(__gm_cond_0, x * 2, out)" in src;
    assert "if torch.isnan(x).any():" not in src;
    assert _break_kinds("where_update_plain.jac") == ["dyn_ctrl_fl"];
}

test "the fp16 overflow guard predicates, short circuit and all" {
    # The Florence-2 / BART encoder-layer clamp. Three things have to happen
    # together: the else-less body becomes a selection, the body-local
    # `clamp_value` hoists unconditionally (nothing reads it after the branch),
    # and the `or` over two 0-dim bool tensors becomes torch.logical_or --
    # hoisting the condition unchanged would leave that `or` as a graph break
    # of its own.
    src = _pysrc("where_predicated_update.jac", True);
    assert (
        "__gm_cond_0 = torch.logical_or(torch.isinf(hidden_states).any(), "
        "torch.isnan(hidden_states).any())" in src
    );
    assert "clamp_value = torch.finfo(hidden_states.dtype).max - 1000" in src;
    assert (
        "hidden_states = torch.where(__gm_cond_0, torch.clamp(hidden_states, "
        "min=-clamp_value, max=clamp_value), hidden_states)" in src
    );
    # the dtype test is not data dependent, so it stays a plain `if`: Dynamo
    # folds it at trace time and it is what keeps the FP32 path untouched.
    assert "if hidden_states.dtype == torch.float16:" in src;
    assert _break_kinds("where_predicated_update.jac") == ["dyn_ctrl_fl"];
}

test "the predicated fp16 guard collapses to one graph and keeps its values" {
    torch = _torch();
    finite = torch.tensor([[1.0, 3.0], [2.0, 4.0]]).half();
    overflow = torch.tensor([[65000.0, 1.0], [2.0, 3.0]]).half() * 10;
    before = _graph_count("where_predicated_update.jac", False, (finite, ));
    after = _graph_count("where_predicated_update.jac", True, (finite, ));
    assert before >= 2 , f"expected >=2 graphs without GraphMend, got {before}";
    assert after == 1 , f"expected 1 graph with GraphMend, got {after}";
    plain = _exec_fixture("where_predicated_update.jac", False)["f"];
    fixed = _exec_fixture("where_predicated_update.jac", True)["f"];
    for probe in [finite, overflow] {
        assert torch.equal(plain(probe), fixed(probe)) , "predication changed the value";
    }
}

test "a grad-reachable predicated update specialises on grad mode" {
    # The autograd hazard is that torch.where routes the untaken operand into
    # the backward pass. It exists only while gradients are recorded, so the
    # rewrite is emitted under a runtime grad-mode test instead of being
    # declined: the training path keeps the original branch verbatim.
    src = _pysrc("where_update_grad_reachable.jac", True);
    assert "if torch.is_grad_enabled():" in src;
    # grad on -> the original guard, untouched
    assert (
        "if hidden_states.dtype == torch.float16 and "
        "(torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):"
        in src
    );
    assert "hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)" in src;
    # grad off -> the predicated variant
    assert (
        "__gm_cond_0 = torch.logical_or(torch.isinf(hidden_states).any(), "
        "torch.isnan(hidden_states).any())" in src
    );
    assert (
        "hidden_states = torch.where(__gm_cond_0, torch.clamp(hidden_states, "
        "min=-clamp_value, max=clamp_value), hidden_states)" in src
    );
    assert _break_kinds("where_update_grad_reachable.jac") == ["dyn_ctrl_fl"];
}

test "grad-mode specialisation removes the break only on the inference path" {
    # Dynamo guards on grad mode rather than tracing it as data, so each mode
    # specialises to one variant. Inference must reach a single contiguous
    # graph; training must be left exactly as it was, breaks and all.
    torch = _torch();
    finite = torch.tensor([[1.0, 3.0], [2.0, 4.0]]).half();
    overflow = torch.tensor([[65000.0, 1.0], [2.0, 3.0]]).half() * 10;
    fixture = "where_update_grad_reachable.jac";
    assert _graph_count(fixture, True, (finite, ), False) == 1;
    before_grad = _graph_count(fixture, False, (finite, ), True);
    after_grad = _graph_count(fixture, True, (finite, ), True);
    assert before_grad >= 2 , f"expected >=2 graphs untransformed, got {before_grad}";
    assert after_grad == before_grad ,
        f"grad path must be untouched: {before_grad} -> {after_grad}";
    plain = _exec_fixture(fixture, False)["f"];
    fixed = _exec_fixture(fixture, True)["f"];
    for probe in [finite, overflow] {
        with torch.no_grad() {
            assert torch.equal(plain(probe), fixed(probe)) ,
                "predication changed the inference value";
        }
        with torch.enable_grad() {
            assert torch.equal(plain(probe), fixed(probe)) ,
                "predication changed the training value";
        }
    }
}

test "a negated tensor predicate becomes a tensor negation" {
    src = _pysrc("where_update_negated.jac", True);
    assert "__gm_cond_0 = torch.logical_not(torch.isnan(x).any())" in src;
    assert "out = torch.where(__gm_cond_0, x * 2, out)" in src;
}

test "a body local that is read after the branch is declined (soundness)" {
    src = _pysrc("where_update_late_local.jac", True);
    # `scale` has no value on the false path, so neither hoisting it nor
    # predicating it preserves what the untaken run used to see.
    assert "torch.where(" not in src;
    assert "if torch.isnan(x).any():" in src;
}

test "a None-defaulted incoming value is declined (soundness)" {
    # The false operand of the selection is the variable's own incoming value.
    # `mask` defaults to None and the enclosing guard proves it IS None on this
    # path, so torch.where would take None as an operand and raise. Found in
    # timm's Gemma4 ViT forward, which the eval imports.
    src = _pysrc("where_update_none_default.jac", True);
    assert "torch.where(" not in src;
    assert "if sentinel.any():" in src;
}

test "an unprovable call in a predicated body is declined (soundness)" {
    src = _pysrc("where_update_impure_body.jac", True);
    # torch.where evaluates the true operand on every run, so `audit` would
    # print on runs the original never took.
    assert "torch.where(" not in src;
    assert "out = audit(x * 2)" in src;
}

test "an augmented assignment in the body is declined" {
    src = _pysrc("where_update_aug.jac", True);
    assert "torch.where(" not in src;
    assert "out += x" in src;
}

test "a short circuit with a non-tensor operand is declined (soundness)" {
    src = _pysrc("where_update_opaque_operand.jac", True);
    # `flag` is a Python bool; torch.logical_or would reject it, and so would
    # torch.where if the whole chain short circuited to it.
    assert "torch.where(" not in src;
    assert "torch.logical_or(" not in src;
    assert "if flag or torch.isnan(x).any():" in src;
}

test "an early return in the body is declined" {
    src = _pysrc("where_update_early_return.jac", True);
    # A predicated update has nowhere to put a statement whose whole effect is
    # to skip the rest of the function.
    assert "torch.where(" not in src;
    assert "if torch.isnan(x).any():" in src;
}

test "an attribute write in the body is declined (soundness)" {
    src = _pysrc("where_update_attr_target.jac", True);
    # The write is observable outside the region, so it is not a predicated
    # update of a local.
    assert "torch.where(" not in src;
    assert "holder.cache = x * 2" in src;
}

test "predicated updates are preserved when graphmend is disabled" {
    src = _pysrc("where_predicated_update.jac", False);
    assert "torch.where(" not in src;
    assert "torch.logical_or(" not in src;
    assert "if hidden_states.dtype == torch.float16 and " in src;
}

test "return branches become torch.where" {
    src = _pysrc("where_return.jac", True);
    assert "torch.where(" in src;
    assert "return torch.where(" in src;
    assert "if " not in src;
    assert _break_kinds("where_return.jac") == ["dyn_ctrl_fl"];
}

test "assignment branches become torch.where" {
    src = _pysrc("where_assign.jac", True);
    assert "z = torch.where(" in src;
    assert "if " not in src;
}

test "py source: return branches become torch.where" {
    # The .py front-end (PyastBuildPass) produces UniIR whose nodes unparse
    # differently from .jac-origin nodes, so the rewrite must operate
    # structurally; this guards `jac run model.py` under the default auto mode.
    src = _py_fixture_src(True);
    assert "return torch.where(" in src;
    assert "if " not in src;
    src = _py_fixture_src(False);
    assert "torch.where(" not in src;
    assert "if " in src;
}

test "py source: an audio-fusion guard is merged by neither arm" {
    # Qwen-Audio-Chat's DC pair, measured by the paper harness. Predicating it
    # would run `self.audio.encode(audio_info[...])` on the text-only runs the
    # false branch exists to serve, where `audio_info` is None, and would then
    # ask torch.where to select between a list of tensors and None. Both arms
    # survive and no condition is hoisted into a temporary.
    src = _compile_path_of_py(PY_AUDIO_FUSION_SRC, "audio_fusion.py");
    assert "__gm_cond" not in src;
    assert "audios = None" in src;
    assert "torch.where(" in src;
    assert "torch.where(input_ids == self.start_id)" in src;
}

test "py source: an audio-fusion guard gains its Python-level precondition" {
    # Every path through the true branch dereferences `audio_info` before it
    # writes anything an observer can see: the dict arm subscripts it, the list
    # arm iterates it. So `audio_info is not None` holds on every run that
    # reaches the end of the branch, and leading the guard with it is
    # observationally inert while letting Dynamo decide the guard in Python
    # instead of tracing the `torch.any` behind it.
    src = _compile_path_of_py(PY_AUDIO_FUSION_SRC, "audio_fusion.py");
    assert "if audio_info is not None and past_key_values is None and torch.any("
        in src;
    # The rewrite adds a conjunct and changes nothing else: both arms stay.
    assert "audios = None" in src;
    assert "audios = self.audio.encode(audio_info['input_audios'])" in src;
}

test "the audio-fusion guard is untouched when graphmend is disabled" {
    src = _compile_path_of_py_off(PY_AUDIO_FUSION_SRC, "audio_fusion_off.py");
    assert "if audio_info is not None" not in src;
    assert "if past_key_values is None and torch.any(" in src;
}

test "an unconditionally dereferenced optional param leads the guard" {
    src = _pysrc("precond_audio_fusion.jac", True);
    assert "if info is not None and past is None and torch.any(x > 0):" in src;
    # Only the condition changes: neither arm is merged or hoisted.
    assert "__gm_cond" not in src;
    assert "fused = None" in src;
    assert "fused = info['values']" in src;
    assert _break_kinds("precond_audio_fusion.jac") == ["dyn_ctrl_fl"];
}

test "the precondition is preserved when graphmend is disabled" {
    src = _pysrc("precond_audio_fusion.jac", False);
    assert "if info is not None" not in src;
    assert "if past is None and torch.any(x > 0):" in src;
}

test "the precondition removes a real break and keeps the result" {
    # The claim the row rests on, measured rather than read off the source: with
    # the optional argument left at its default the guard folds in Python, the
    # two halves fuse into one graph, and the tensor that comes back is the same
    # one the untransformed function returns.
    torch = _torch();
    args = (-torch.ones(4), );
    assert _graph_count("precond_integration.jac", False, args) == 2;
    assert _graph_count("precond_integration.jac", True, args) == 1;
    off = _exec_fixture("precond_integration.jac", False)["f"](*args);
    on = _exec_fixture("precond_integration.jac", True)["f"](*args);
    assert torch.equal(off, on);
}

test "an attribute dereference establishes the precondition" {
    src = _pysrc("precond_attr_deref.jac", True);
    assert "if info is not None and torch.any(x > 0):" in src;
}

test "a dereference on only one inner arm is declined (soundness)" {
    # The other arm finishes the branch with `info` still None, so the test is
    # not a necessary condition of reaching the end of it.
    src = _pysrc("precond_partial_deref.jac", True);
    assert "if info is not None" not in src;
    assert "if torch.any(x > 0):" in src;
}

test "an observable write before the dereference is declined (soundness)" {
    src = _pysrc("precond_effect_first.jac", True);
    assert "if info is not None" not in src;
    assert "holder.cache = 1" in src;
}

test "an opaque call before the dereference is declined (soundness)" {
    src = _pysrc("precond_call_first.jac", True);
    assert "if info is not None" not in src;
    assert "staged = prep(x)" in src;
}

test "a rebound parameter is declined (soundness)" {
    # The dereference proves the post-rebind value non-None, which is not the
    # value the prepended test would read.
    src = _pysrc("precond_rebound_param.jac", True);
    assert "if info is not None" not in src;
    assert "if torch.any(x > 0):" in src;
}

test "a branch that can return early is declined (soundness)" {
    src = _pysrc("precond_early_exit.jac", True);
    assert "if info is not None" not in src;
    assert "if flag is not None" not in src;
}

test "a parameter without a None default is declined" {
    src = _pysrc("precond_required_param.jac", True);
    assert "if info is not None" not in src;
    assert "if torch.any(x > 0):" in src;
}

test "a guard whose own condition may have effects is declined (soundness)" {
    # Skipping the condition would skip the call inside it.
    src = _pysrc("precond_opaque_guard.jac", True);
    assert "if info is not None" not in src;
    assert "if torch.any(probe(x) > 0):" in src;
}

test "a guard inside a try is declined (soundness)" {
    # The raise the proof leans on is caught here, so it is live control flow.
    src = _pysrc("precond_guard_in_try.jac", True);
    assert "if info is not None" not in src;
    assert "if torch.any(x > 0):" in src;
}

test "a region [Where] already rewrote is declined (soundness)" {
    # [Where] lifts `info['a']` out of the arm that guarded it, which would make
    # a conditional dereference look unconditional. The proof has to read the
    # program as written, so a region carrying a synthesized temporary is out.
    src = _pysrc("precond_after_inner_rewrite.jac", True);
    assert "__gm_true_0 = info['a']" in src;
    assert "if info is not None" not in src;
    assert "if torch.any(x > 0):" in src;
}

test "a truth-value read is not a dereference (soundness)" {
    # bool(None) is False, not an error, so nothing is established.
    src = _pysrc("precond_mention_only.jac", True);
    assert "if info is not None" not in src;
}

test "common call branches merge differing arg via torch.where" {
    src = _pysrc("where_common_call.jac", True);
    assert "torch.where(" in src;
    # the differing arg is merged inside the single preserved call
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "persistent=False" in src;
    assert "if " not in src;
}

test "common call merges multiple differing args, preserves kwargs" {
    src = _pysrc("where_multi_arg.jac", True);
    # both differing positional args merged independently; kwarg kept
    assert "combine(torch.where(" in src;
    assert src.count("torch.where(") == 2;
    assert "mode='x'" in src;
    assert "if " not in src;
}

test "a scalar-materializing predicate is declined (soundness)" {
    # `.item()` yields a Python bool, and `torch.where` rejects a non-tensor
    # condition, so predicating this branch would replace a working graph
    # break with a TypeError. Detection still tags both shapes -- the direct
    # one and the one reached only through the use-def trace.
    src = _pysrc("scalar_predicate.jac", True);
    assert "torch.where(" not in src;
    assert "if x.item() > 5:" in src;
    assert "if hits > 5:" in src;
    # A bare alias is still a use-def edge, and the sibling `.sum()` licenses
    # detection on its own, so only the chain can spot the scalar.
    assert "if x.sum() > 0 and same > 5:" in src;
    assert _break_kinds("scalar_predicate.jac")
        == ["dyn_ctrl_fl", "dyn_ctrl_fl", "dyn_ctrl_fl"];
}

test "a scalar past the trace bound declines rather than slipping through" {
    # The two queries share the bound but not a safe answer when it runs out:
    # detection must fail closed (do not rewrite), the scalar query must
    # decline. Only that polarity keeps the out-of-reach `.item()` from
    # reaching torch.where as a Python bool.
    src = _pysrc("scalar_past_trace_bound.jac", True);
    assert "torch.where(" not in src;
    assert "if x.sum() > 0 and a11 > 5:" in src;
    assert _break_kinds("scalar_past_trace_bound.jac") == ["dyn_ctrl_fl"];
}

test "a predicate with more than one definition is not traced" {
    # The trace follows one defining assignment; with several, the reaching
    # one is not lexically decidable, so both queries answer safely and the
    # branch stays.
    src = _pysrc("multi_def_predicate.jac", True);
    assert "torch.where(" not in src;
    assert "if flag:" in src;
    assert _break_kinds("multi_def_predicate.jac") == [];
}

test "call-bearing branch values are declined" {
    src = _pysrc("cond_assign.jac", True);
    # torch.where would evaluate both calls, and torch.cond branches need
    # callees proven pure and capturable, a proof the analysis does not yet
    # attempt; the conditional stays.
    assert "torch.cond(" not in src;
    assert "torch.where(" not in src;
    assert "if x.sum() > 0:" in src;
    assert "dyn_ctrl_fl" in _break_kinds("cond_assign.jac");
}

test "different callees in branches are left untouched" {
    src = _pysrc("different_callees.jac", True);
    assert "torch.where(" not in src;
    assert "if x.sum() > 0:" in src;
    assert "foo(p)" in src;
    assert "bar(q)" in src;
}

test "static hasattr control flow is not tagged or rewritten" {
    # `if hasattr(self_obj, "flag")` has no tensor-data dependence; Dynamo
    # traces it without a break, so GraphMend must leave it alone. The tag
    # table is checked too: this shape WOULD be rewritten if mis-tagged.
    src = _pysrc("hasattr_static.jac", True);
    assert "torch.where(" not in src;
    assert "if hasattr(self_obj, 'flag')" in src;
    assert _break_kinds("hasattr_static.jac") == [];
}

test "multi-statement branches with shared call hoist and merge" {
    src = _pysrc("where_multi_stmt.jac", True);
    # data-dependent branch becomes a predicate assignment...
    assert "__gm_cond_0 = seq_len > omp" in src;
    assert "if seq_len > omp:" not in src;
    # ...with setups hoisted and the shared call emitted once via torch.where
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "self_obj.original_inv_freq = self_obj.original_inv_freq.to(device)" in src;
    # static hasattr guards are preserved (not rewritten)
    assert "if hasattr(self_obj.config" in src;
    assert "if not hasattr(self_obj, 'long_inv_freq')" in src;
    # `seq_len > omp` names no dynamic attr inline: only the use-def trace
    # (seq_len derives from torch.max) can tag it, and it must tag ONLY it.
    assert _break_kinds("where_multi_stmt.jac") == ["dyn_ctrl_fl"];
}

test "non-idempotent observable setup write is declined (soundness)" {
    src = _pysrc("where_unsafe_setup.jac", True);
    # `self_obj.calls = self_obj.calls + 1` is an observable, non-idempotent
    # write; hoisting it onto both paths would change state, so the legality
    # check declines and the break is left intact (no predication).
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
    assert "self_obj.calls = self_obj.calls + 1" in src;
}

test "setups writing the same bare local in both branches are declined" {
    src = _pysrc("where_local_clobber.jac", True);
    # Hoisting both `scale` writes leaves the else-branch value in place for
    # the merged call, so the true path would read the wrong operand.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a setup writing a module-scope name is declined" {
    # Function-scope confinement cannot see module-level readers, so a
    # target colliding with a module name declines.
    src = _pysrc("where_global_setup.jac", True);
    assert "torch.where(" not in src;
    assert "if x.sum() > 0:" in src;
}

test "a setup writing a local live outside the region is declined" {
    src = _pysrc("where_local_escape.jac", True);
    # `scale` is read after the branch, so hoisting the true-branch write
    # changes what the else path returns.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "differing common-call arguments containing calls are declined" {
    src = _pysrc("merge_arg_calls.jac", True);
    # torch.where evaluates both operands; merging call-bearing arguments
    # would run the untaken branch's call. Same rule as call_vs_literal,
    # applied to the common-call merge path.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a shadowed torch root in the initializer is declined" {
    src = _pysrc("guard_shadowed_torch_root.jac", True);
    # `import fake as torch` rebinds the root; the allowlisted leaf name
    # proves nothing without a genuine `import torch` binding.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a shadowed pure builtin in the initializer is declined" {
    src = _pysrc("guard_shadowed_len.jac", True);
    # `len` is on the pure-builtin list by spelling, but the module rebinds
    # it to an effectful function; the binding is the proof, not the name.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a device move whose consumers all normalize first is licensed" {
    src = _pysrc("where_device_move_shared.jac", True);
    # The other function's first mention of the attribute is the same
    # normalizing move, so no consumer can observe the hoisted placement.
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "if seq_len > 8:" not in src;
}

test "a normalization that does not dominate its reads is declined" {
    src = _pysrc("where_conditional_normalizer.jac", True);
    # `peek` normalizes only under a flag; the read outside that branch can
    # observe the hoisted placement. Textual order is not dominance.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a device move with a raw reader elsewhere is declined" {
    src = _pysrc("where_raw_consumer.jac", True);
    # `peek` reads the attribute without normalizing, so it can observe the
    # placement change on originally-untaken executions.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a device move whose target the region never consumes is declined" {
    src = _pysrc("where_unused_device_move.jac", True);
    # The move license exists to put a value consumed by the merged call on
    # the right device; an unconsumed target is a bare attribute mutation.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a twice-called helper is a diamond, not a cycle, and stays licensed" {
    src = _pysrc("guard_diamond_init.jac", True);
    # The sub-helper is proven once per call site (floor and ceil bounds);
    # only a helper still on the recursion stack declines. The initializer
    # also exercises floor/ceil and a tensor method on a local call's result.
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "if seq_len > 8:" not in src;
}

test "py source: a guard-imported torch root still proves" {
    # Model code imports torch behind an availability guard, which leaves the
    # name without a symbol binding; the proof falls back to the module's
    # plain import statement, and module-level rebinding still declines.
    src = _compile_path_of_py(PY_GUARDED_IMPORT_SRC, "py_guarded_import.py");
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "if seq_len > 8:" not in src;
}

test "a recursive initializer helper is declined" {
    src = _pysrc("guard_recursive_init.jac", True);
    # Each helper body is clean, but the call graph has a cycle; termination
    # of the speculated call is unprovable, so the hoist declines.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "an initializer helper that writes an attribute is declined" {
    src = _pysrc("guard_helper_write.jac", True);
    # The helper neither raises nor calls anything impure, but it assigns
    # `config.cache`; purity covers writes too, so the hoist declines.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "an initializer helper that deletes an attribute is declined" {
    src = _pysrc("guard_helper_delete.jac", True);
    # A delete is a write the assignment walk cannot see, so it needs its own
    # check; without one the helper reads as pure and the hoist proceeds.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "config read proven by a constructor invariant licenses the hoist" {
    src = _pysrc("guard_config_get.jac", True);
    # `config.rope_scaling.get("factor")` is on no purity list; the fact pass
    # proves it from RopeConfig's constructor, which raises unless the
    # attribute is a dict. The receiver type comes from the call-site
    # argument's annotation, not from the callee's declared parameter.
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "if seq_len > 8:" not in src;
}

test "a constructor invariant one import away is materialized and proves" {
    src = _pysrc("where_config_import.jac", True);
    # RopeConfig lives in a sibling module: the query follows the import and
    # compiles that module's AST into the hub on demand, then reads the
    # invariant from it.
    assert "register_buffer('inv_freq', torch.where(" in src;
    assert "if seq_len > 8:" not in src;
}

test "config read without the constructor invariant is declined" {
    src = _pysrc("guard_config_get_unproven.jac", True);
    # LooseConfig never validates rope_scaling, so `.get` stays opaque: the
    # same shape could be a session object doing I/O. Must decline.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "existence-guarded init calling a tensor-named method on an opaque receiver is declined" {
    src = _pysrc("guard_opaque_method_init.jac", True);
    # `config.store.clone()` matches the pure-tensor-method name list, but the
    # receiver is not a pure call, so nothing proves the method is a no-op.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "existence-guarded init calling an effectful torch API is declined" {
    src = _pysrc("guard_torch_effect_init.jac", True);
    # `torch.save` is rooted at torch but writes a file; purity is decided by
    # the checked pure-op list, not by the root module.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "hasattr guard whose key is not the memoized attribute is declined" {
    src = _pysrc("guard_key_mismatch.jac", True);
    # The `hasattr` header does not close over `counter`/`long_inv_freq`, so the
    # body is not idempotent: matching the guard *shape* alone would hoist a
    # non-idempotent write onto the untaken path. Must decline.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
    assert "self_obj.counter = self_obj.counter + 1" in src;
}

test "hasattr guard body with extra observable effects is declined" {
    src = _pysrc("guard_body_effects.jac", True);
    # Key matches the memoized write, but the body also mutates `self_obj.log`;
    # that effect would fire on runs the original never took. Must decline.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
    assert "self_obj.log.append('init')" in src;
}

test "hasattr-initialized attribute read outside the region is declined" {
    src = _pysrc("guard_attr_escapes.jac", True);
    # Hoisting creates `long_inv_freq` earlier than the original would; another
    # reader in the module can observe that, so the state change is not
    # confined and the hoist is declined.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "a call paired with a literal branch value is declined" {
    src = _pysrc("call_vs_literal.jac", True);
    # torch.where would speculate the call onto the untaken path; torch.cond
    # cannot return the float. Neither is sound, so the branch stays.
    assert "torch.where(" not in src;
    assert "torch.cond(" not in src;
    assert "if x.sum() > 0:" in src;
    assert "__gm_true_0 = side_effecting" not in src;
}

test "provably non-tensor branch values are declined" {
    src = _pysrc("where_nontensor_branch.jac", True);
    # torch.where cannot select a string or a container: rewriting would trade
    # a working conditional for a runtime TypeError, so both shapes decline.
    assert "torch.where(" not in src;
    assert "torch.cond(" not in src;
    assert "y = 'big'" in src;
    assert "return [x, 1]" in src;
    assert "dyn_ctrl_fl" in _break_kinds("where_nontensor_branch.jac");
}

test "scalar literal branch values are declined" {
    src = _pysrc("where_scalar_branch.jac", True);
    # torch.where returns a 0-dim tensor, so predicating an int, a float or a
    # bool branch changes the variable's type for every non-tensor consumer.
    assert "torch.where(" not in src;
    assert "torch.cond(" not in src;
    assert "n = 1" in src;
    assert "ok = True" in src;
    assert "return -1" in src;
    assert "dyn_ctrl_fl" in _break_kinds("where_scalar_branch.jac");
}

test "a scalar inside a tensor expression still predicates" {
    # The decline is about the branch VALUE being a scalar, not about a scalar
    # appearing anywhere in it: `x * (b * -1)` is a tensor.
    src = _pysrc("where_return.jac", True);
    assert "torch.where(" in src;
}

test "existence-guarded init with an impure initializer is declined" {
    src = _pysrc("guard_impure_init.jac", True);
    # The attribute callee cannot be pinned to one table entry, so every
    # signature-compatible candidate must be pure; `_compute_dirty` prints, so
    # speculating the init could emit output on runs the original never made.
    assert "torch.where(" not in src;
    assert "if seq_len > 8:" in src;
}

test "nested guard in a data-dependent branch composes trap + predication" {
    src = _pysrc("nested_guard_in_branch.jac", True);
    # the data-dependent outer branch is predicated...
    assert "__gm_cond_0 = x.sum() > 0" in src;
    assert "out.copy_(torch.where(__gm_cond_0," in src;
    # ...and the inner validation guard is lowered to a graph-native assert,
    # re-gated on the branch predicate so it only fires when the branch is taken.
    assert "torch._assert_async(" in src;
    # the gate is plain logical_or: trap lowering only fires on a receiver
    # proven to be a tensor, so the check needs no coercion.
    assert ("torch.logical_or(torch.logical_not(__gm_cond_0), (x == y).all())" in src);
    # no raw control flow / raise survives -- one contiguous graph.
    assert "raise ValueError" not in src;
    assert "if x.sum()" not in src;
    assert "if not (x == y)" not in src;
}

test "nested-compound break preserved when graphmend disabled" {
    src = _pysrc("nested_guard_in_branch.jac", False);
    assert "torch.where(" not in src;
    assert "torch._assert_async(" not in src;
    assert "raise ValueError('x and y must match')" in src;
    assert "if not (x == y).all():" in src;
}

test "control flow preserved when graphmend disabled" {
    src = _pysrc("where_return.jac", False);
    assert "torch.where(" not in src;
    assert "if " in src;
}
