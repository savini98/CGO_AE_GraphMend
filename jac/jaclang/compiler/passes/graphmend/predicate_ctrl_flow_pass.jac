import jaclang.compiler.frontend.unitree as uni;
import from jaclang.compiler.passes { UniPass }
import type from jaclang.compiler.driver.program { JacProgram }

glob _PURE_BUILTINS: set[str] = {
         'len',
         'float',
         'int',
         'bool',
         'str',
         'max',
         'min',
         'abs',
         'sum',
         'list',
         'tuple',
         'range',
         'getattr',
         'hasattr'
     },
     _PURE_TENSOR_METHODS: set[str] = {
         'all',
         'any',
         'to',
         'float',
         'double',
         'clone',
         'detach'
     },
     _TENSOR_BOOL_REDUCERS: set[str] = {'all', 'any'},
     _PURE_TORCH_FNS: set[str] = {
         'arange',
         'as_tensor',
         'cat',
         'ceil',
         'clamp',
         'cos',
         'einsum',
         'exp',
         'eye',
         'finfo',
         'floor',
         'full',
         'isinf',
         'isnan',
         'linspace',
         'log',
         'log2',
         'log10',
         'maximum',
         'minimum',
         'ones',
         'outer',
         'pow',
         'reshape',
         'rsqrt',
         'sin',
         'sqrt',
         'stack',
         'tensor',
         'where',
         'zeros'
     },
     _PRECOND_BUILTINS: set[str] = {'isinstance', 'issubclass', 'callable'},
     _PRECOND_TORCH_FNS: set[str] = {
         'all',
         'any',
         'argmax',
         'argmin',
         'count_nonzero',
         'eq',
         'equal',
         'ge',
         'gt',
         'isclose',
         'le',
         'logical_and',
         'logical_not',
         'logical_or',
         'lt',
         'mean',
         'ne',
         'numel',
         'prod',
         'sum'
     },
     _SCAN_UNSAFE: int = 0,
     _SCAN_CLEAN: int = 1,
     _SCAN_FATAL: int = 2;

obj PredicateCtrlFlowPass(UniPass) {
    has prog: JacProgram,
        _enabled: bool = False,
        _autograd_reachable: bool = False,
        _counter: int = 0,
        _mat_seen: set[str] = set();

    static has REQUIRES: tuple = ('graph-breaks', 'lowered-guards'),
        PROVIDES: tuple = ('predicated-branches', ),
        HOST_PASS: bool = True;

    def before_pass -> None;
    def exit_if_stmt(nd: uni.IfStmt) -> None;
    def _where_family(nd: uni.IfStmt) -> None;
    def _strengthen_guard(nd: uni.IfStmt) -> None;
    def _optional_params(fn: uni.Ability) -> list[str];
    def _never_rebound(fn: uni.Ability, name: str) -> bool;
    def _names_bound(targets: Sequence[any], name: str) -> bool;
    def _in_try(nd: uni.IfStmt, fn: uni.Ability) -> bool;
    def _cond_effect_free(cond: uni.Expr) -> bool;
    def _prepend_conjunct(nd: uni.IfStmt, name: str) -> None;
    def _seq_scan(stmts: Sequence[any], name: str) -> int;
    def _stmt_scan(s: uni.UniNode, name: str) -> int;
    def _else_seq(nd: uni.IfStmt) -> (list[any] | None);
    def _expr_scan(e: (uni.UniNode | None), name: str) -> int;
    def _expr_seq_scan(items: list[any], name: str) -> int;
    def _bool_scan(e: uni.BoolExpr, name: str) -> int;
    def _compare_scan(e: uni.CompareExpr, name: str) -> int;
    def _ternary_scan(e: uni.IfElseExpr, name: str) -> int;
    def _trailer_scan(e: uni.AtomTrailer, name: str) -> int;
    def _slice_scan(e: uni.IndexSlice, name: str) -> int;
    def _call_scan(c: uni.FuncCall, name: str) -> int;
    def _compr_scan(compr: list[any], parts: list[any], name: str) -> int;
    def _scan_call_pure(c: uni.FuncCall) -> bool;
    def _is_named(e: (uni.UniNode | None), name: str) -> bool;
    def _local_store_targets(targets: Sequence[any]) -> bool;
    def _rewrite_assignment(
        nd: uni.IfStmt, lhs: uni.Name, true_e: uni.Expr, false_e: uni.Expr
    ) -> None;

    def _rewrite_return(nd: uni.IfStmt, true_e: uni.Expr, false_e: uni.Expr) -> None;
    def _rewrite_predicated_update(nd: uni.IfStmt, grad_gated: bool) -> None;
    def _specialize_on_grad(
        nd: uni.IfStmt, static_cond: (uni.Expr | None), new_stmts: list
    ) -> None;

    def _torch_bound(nd: uni.IfStmt) -> bool;
    def _unwrap_expr(e: uni.Expr) -> uni.Expr;
    def _bool_join(e: uni.Expr) -> (str | None);
    def _lower_tensor_bool(
        at: uni.UniNode, e: uni.Expr, merged: bool
    ) -> (uni.Expr | None);

    def _static_and_prefix(
        cond: uni.Expr
    ) -> tuple[(uni.Expr | None), (uni.Expr | None)];

    def _tensor_bool_atom(e: uni.Expr) -> bool;
    def _update_targets(nd: uni.IfStmt) -> (list | None);
    def _merged_call_stmt(
        nd: uni.IfStmt,
        ca: uni.FuncCall,
        a_pos: list,
        merged: list,
        diff: list
    ) -> uni.ExprStmt;

    def _buffer_preregistered(nd: uni.IfStmt, bufname: str) -> bool;
    def _pure_value(v: uni.Expr) -> bool;
    def _bound_before(nd: uni.IfStmt, name: str) -> bool;
    def _incoming_selectable(nd: uni.IfStmt, name: str) -> bool;
    def _param_default_selectable(fn: uni.Ability, name: str) -> bool;
    def _none_tested(c: (uni.Expr | None), name: str) -> bool;
    def _merge_common_call(
        nd: uni.IfStmt,
        ca: uni.Expr,
        cb: uni.Expr,
        true_setups: Sequence[uni.CodeBlockStmt],
        false_setups: Sequence[uni.CodeBlockStmt]
    ) -> None;

    def _has_call(e: uni.UniNode) -> bool;
    def _has_none(a: uni.Expr, b: uni.Expr) -> bool;
    def _provably_non_tensor(e: uni.Expr) -> bool;
    def _scalar_literal(e: uni.Expr) -> bool;
    def _non_tensor_container(e: uni.Expr) -> bool;
    def _same_lhs(a: uni.Assignment, b: uni.Assignment) -> (uni.Name | None);
    def _assign(target: uni.Name, value: uni.Expr) -> uni.Assignment;
    def _hoist(
        at: uni.UniNode, cond: uni.Expr, t: uni.Expr, f: uni.Expr
    ) -> tuple[list, uni.Name, uni.Name, uni.Name];

    def _hoist_cond(at: uni.UniNode, cond: uni.Expr) -> tuple[list, uni.Name];
    def _is_assert_async(stmt: uni.UniNode) -> bool;
    def _is_eq_assert(stmt: uni.UniNode) -> bool;
    def _gate_asserts(setups: list, n: int, taken_when_true: bool) -> list;
    def _setups_safe(setups: Sequence[uni.CodeBlockStmt], region: uni.IfStmt) -> bool;
    def _effect_neutral(s: uni.UniNode, region: uni.IfStmt) -> bool;
    def _is_existence_guard(s: uni.UniNode, region: uni.IfStmt) -> bool;
    def _init_value_safe(v: (uni.Expr | uni.YieldExpr | None)) -> bool;
    def _call_speculatable(c: uni.FuncCall) -> bool;
    def _signature_accepts(fn: uni.Ability, c: uni.FuncCall) -> bool;
    def _module_fn(mod: uni.Module, name: str) -> (uni.Ability | None);
    def _dispatch_table_fns(mod: uni.Module) -> list[uni.Ability];
    def _impure_writes(fn: uni.Ability, mod: (uni.Module | None)) -> bool;
    def _module_scope_names(mod: uni.Module) -> set[str];
    def _body_pure_nonraising(
        fn: uni.Ability,
        c: uni.FuncCall,
        seen: (set[int] | None) = None,
        ctx: (dict[str, list[uni.Archetype]] | None) = None
    ) -> bool;

    def _callsite_ctx(
        fn: uni.Ability, c: uni.FuncCall
    ) -> dict[str, list[uni.Archetype]];

    def _arg_classes(
        arg: uni.Expr, c: uni.FuncCall, selfcls: list[uni.Archetype]
    ) -> list[uni.Archetype];

    def _self_classes(c: uni.FuncCall) -> list[uni.Archetype];
    def _ctx_dict_get(call: uni.FuncCall, ctx: dict[str, list[uni.Archetype]]) -> bool;
    def _witnessed_dict_get(call: uni.FuncCall, fn: uni.Ability) -> bool;
    def _mapping_witnessed(mod: uni.Module, key: str) -> bool;
    def _hasattr_key(c: uni.Expr) -> (tuple[str, str] | None);
    def _writes_attr(s: uni.Assignment, recv: str, key: str) -> bool;
    def _attr_confined(region: uni.IfStmt, key: str) -> bool;
    def _within(nd: uni.UniNode, anc: uni.UniNode) -> bool;
    def _varkw_name(fn: uni.Ability) -> (str | None);
    def _passes_extra_kwargs(fn: uni.Ability, c: uni.FuncCall) -> bool;
    def _guarded_by_varkw(r: uni.RaiseStmt, varkw: str, fn: uni.Ability) -> bool;
    def _mentions_name(e: (uni.Expr | None), nm: str) -> bool;
    def _pure_callee(t: (uni.Expr | None)) -> bool;
    def _pure_root(base: uni.Name) -> bool;
    def _pure_receiver(recv: (uni.Expr | None)) -> bool;
    def _assignment_neutral(s: uni.Assignment, region: uni.IfStmt) -> bool;
    def _is_device_move(s: uni.Assignment) -> bool;
    def _move_consumed(s: uni.Assignment, region: uni.IfStmt) -> bool;
    def _consumers_harmless(s: uni.Assignment, region: uni.IfStmt) -> bool;
    def _normalizes_first(ab: uni.Ability, tgt: str) -> bool;
    def _stmt_spot(nd: uni.UniNode) -> (tuple[int, int] | None);
    def _spot_chain(nd: uni.UniNode) -> dict[int, int];
    def _all_local_targets(targets: Sequence[uni.Expr], region: uni.IfStmt) -> bool;
    def _setup_writes(setups: Sequence[uni.CodeBlockStmt]) -> set[str];
}
