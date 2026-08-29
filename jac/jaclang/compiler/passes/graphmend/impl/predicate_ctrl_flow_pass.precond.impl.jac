import jaclang.compiler.frontend.unitree as uni;
import from jaclang.compiler.frontend.constant { Tokens as Tok }
import from jaclang.compiler.passes.graphmend.ast_edit {
    gen_name,
    gen_null,
    gen_token
}
import from jaclang.compiler.passes.graphmend.scope_facts { param_names }
import from jaclang.compiler.passes.graphmend.torch_attr_table { symbol_is_builtin }

impl PredicateCtrlFlowPass._strengthen_guard(nd: uni.IfStmt) -> None {
    fn = nd.find_parent_of_type(uni.Ability);
    if (fn is None) or (not (nd.body or [])) {
        return;
    }
    if self._in_try(nd, fn) or (not self._cond_effect_free(nd.condition)) {
        return;
    }
    for n in nd.get_all_sub_nodes(uni.Name) {
        if n.value.startswith('__gm_') {
            return;
        }
    }
    for name in self._optional_params(fn) {
        if self._mentions_name(nd.condition, name)
            or (not self._never_rebound(fn, name))
            or (self._seq_scan(nd.body, name) != _SCAN_FATAL) {
            continue;
        }
        self._prepend_conjunct(nd, name);
        return;
    }
}

impl PredicateCtrlFlowPass._cond_effect_free(cond: uni.Expr) -> bool {
    return self._expr_scan(cond, '') == _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._optional_params(fn: uni.Ability) -> list[str] {
    sig = fn.signature;
    if not isinstance(sig, uni.FuncSignature) {
        return [];
    }
    out: list[str] = [];
    for prm in (sig.get_parameters() or []) {
        d = prm.value;
        if (prm?.name is None) or prm.is_vararg or prm.is_kwargs or (d is None) {
            continue;
        }
        if isinstance(self._unwrap_expr(d), uni.Null) {
            out.append(prm.name.value);
        }
    }
    return out;
}

impl PredicateCtrlFlowPass._never_rebound(fn: uni.Ability, name: str) -> bool {
    if name not in param_names(fn) {
        return False;
    }
    if fn.get_all_sub_nodes(uni.MatchStmt) {
        return False;
    }
    for a in fn.get_all_sub_nodes(uni.Assignment) {
        if self._names_bound(a.target or [], name) {
            return False;
        }
    }
    for b in fn.get_all_sub_nodes(uni.BinaryExpr) {
        op: any = b.op;
        if isinstance(op, uni.Token)
            and (op.name == Tok.WALRUS_EQ)
            and self._names_bound([b.left], name) {
            return False;
        }
    }
    for f in fn.get_all_sub_nodes(uni.InForStmt) {
        if self._names_bound([f.target], name) {
            return False;
        }
    }
    for ic in fn.get_all_sub_nodes(uni.InnerCompr) {
        if self._names_bound([ic.target], name) {
            return False;
        }
    }
    for w in fn.get_all_sub_nodes(uni.ExprAsItem) {
        if (w.alias is not None) and self._names_bound([w.alias], name) {
            return False;
        }
    }
    for ex in fn.get_all_sub_nodes(uni.Except) {
        if (ex.name is not None) and (ex.name.value == name) {
            return False;
        }
    }
    for d in fn.get_all_sub_nodes(uni.DeleteStmt) {
        if self._mentions_name(d.target, name) {
            return False;
        }
    }
    return True;
}

impl PredicateCtrlFlowPass._names_bound(targets: Sequence[any], name: str) -> bool {
    for t in (targets or []) {
        cur: any = self._unwrap_expr(t) if isinstance(t, uni.Expr) else t;
        if isinstance(cur, uni.UnaryExpr) {
            cur = self._unwrap_expr(cur.operand);
        }
        if isinstance(cur, (uni.TupleVal, uni.ListVal)) {
            if self._names_bound(list(cur.values or []), name) {
                return True;
            }
            continue;
        }
        if isinstance(cur, uni.Name) and (cur.value == name) {
            return True;
        }
    }
    return False;
}

impl PredicateCtrlFlowPass._in_try(nd: uni.IfStmt, fn: uni.Ability) -> bool {
    cur: (uni.UniNode | None) = nd;
    while (cur is not None) and (cur is not fn) {
        if isinstance(cur, uni.TryStmt) {
            return True;
        }
        cur = cur.parent;
    }
    return False;
}

impl PredicateCtrlFlowPass._prepend_conjunct(nd: uni.IfStmt, name: str) -> None {
    ref = gen_name(nd, name, False);
    nul = gen_null(nd);
    isn = gen_token(nd, Tok.KW_ISN.value, 'is not');
    isnone = uni.CompareExpr(
        left=ref, rights=[nul], ops=[isn], kid=[ref, isn, nul]
    );
    lead = uni.CfgExpr(expr=isnone, kid=[isnone]);
    cond = nd.condition;
    inner = self._unwrap_expr(cond);
    vals: list[any] = [];
    join: any = None;
    if (self._bool_join(inner) == 'and') and isinstance(inner, uni.BoolExpr) {
        vals = [lead] + list(inner.values or []);
        join = inner.op;
    } else {
        kept: any = (
            cond if isinstance(cond, uni.CfgExpr) else uni.CfgExpr(
                expr=cond, kid=[cond]
            )
        );
        vals = [lead, kept];
        join = gen_token(nd, Tok.KW_AND.value, 'and');
    }
    merged = uni.BoolExpr(op=join, values=vals, kid=[join] + vals);
    holder: any = nd;
    setattr(holder, 'condition', merged);
    kids: list[any] = [merged] + list(nd.body or []);
    if nd.else_body is not None {
        kids.append(nd.else_body);
    }
    nd.set_kids(kids);
    self._counter += 1;
}

impl PredicateCtrlFlowPass._seq_scan(stmts: Sequence[any], name: str) -> int {
    for s in (stmts or []) {
        r = self._stmt_scan(s, name);
        if r != _SCAN_CLEAN {
            return r;
        }
    }
    return _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._else_seq(nd: uni.IfStmt) -> (list[any] | None) {
    eb = nd.else_body;
    if eb is None {
        return [];
    }
    if isinstance(eb, uni.ElseStmt) {
        return list(eb.body or []);
    }
    if isinstance(eb, uni.IfStmt) {
        return [eb];
    }
    return None;
}

impl PredicateCtrlFlowPass._stmt_scan(s: uni.UniNode, name: str) -> int {
    if isinstance(s, uni.Assignment) {
        v = s.value;
        if (s.aug_op is not None) or (not isinstance(v, uni.Expr)) {
            return _SCAN_UNSAFE;
        }
        r = self._expr_scan(v, name);
        if r != _SCAN_CLEAN {
            return r;
        }
        return (
            _SCAN_CLEAN
            if self._local_store_targets(s.target or [])
            else _SCAN_UNSAFE
        );
    }
    if isinstance(s, uni.ExprStmt) {
        return self._expr_scan(s.expr, name);
    }
    if isinstance(s, uni.AssertStmt) {
        r = self._expr_scan(s.condition, name);
        if r != _SCAN_CLEAN {
            return r;
        }
        if s.error_msg is None {
            return _SCAN_CLEAN;
        }
        return (
            _SCAN_CLEAN
            if (self._expr_scan(s.error_msg, name) == _SCAN_CLEAN)
            else _SCAN_UNSAFE
        );
    }
    if isinstance(s, uni.IfStmt) {
        rc = self._expr_scan(s.condition, name);
        if rc != _SCAN_CLEAN {
            return rc;
        }
        alt = self._else_seq(s);
        if alt is None {
            return _SCAN_UNSAFE;
        }
        rt = self._seq_scan(s.body, name);
        rf = self._seq_scan(alt, name);
        if (rt == _SCAN_UNSAFE) or (rf == _SCAN_UNSAFE) {
            return _SCAN_UNSAFE;
        }
        return (
            _SCAN_FATAL
            if ((rt == _SCAN_FATAL) and (rf == _SCAN_FATAL))
            else _SCAN_CLEAN
        );
    }
    if isinstance(s, uni.InForStmt) {
        rc = self._expr_scan(s.collection, name);
        if rc == _SCAN_FATAL {
            return _SCAN_FATAL;
        }
        if (rc == _SCAN_CLEAN) and self._is_named(s.collection, name) {
            return _SCAN_FATAL;
        }
        return _SCAN_UNSAFE;
    }
    return _SCAN_UNSAFE;
}

impl PredicateCtrlFlowPass._expr_scan(e: (uni.UniNode | None), name: str) -> int {
    if e is None {
        return _SCAN_CLEAN;
    }
    if isinstance(e, uni.CfgExpr) {
        return self._expr_scan(e.expr, name);
    }
    if isinstance(e, uni.AtomUnit) {
        return (
            self._expr_scan(e.value, name)
            if isinstance(e.value, uni.Expr)
            else _SCAN_UNSAFE
        );
    }
    if isinstance(e, uni.KWPair) {
        return self._expr_scan(e.value, name);
    }
    if isinstance(e, uni.Name) or isinstance(e, (uni.Literal, uni.MultiString)) {
        return _SCAN_CLEAN;
    }
    if isinstance(e, uni.AtomTrailer) {
        return self._trailer_scan(e, name);
    }
    if isinstance(e, uni.FuncCall) {
        return self._call_scan(e, name);
    }
    if isinstance(e, uni.BinaryExpr) {
        return self._expr_seq_scan([e.left, e.right], name);
    }
    if isinstance(e, uni.CompareExpr) {
        return self._compare_scan(e, name);
    }
    if isinstance(e, uni.BoolExpr) {
        return self._bool_scan(e, name);
    }
    if isinstance(e, uni.UnaryExpr) {
        return self._expr_scan(e.operand, name);
    }
    if isinstance(e, uni.IfElseExpr) {
        return self._ternary_scan(e, name);
    }
    if isinstance(e, (uni.ListVal, uni.SetVal, uni.TupleVal)) {
        return self._expr_seq_scan(list(e.values or []), name);
    }
    if isinstance(e, uni.DictVal) {
        items: list[any] = [];
        for kv in (e.kv_pairs or []) {
            items.append(kv.key);
            items.append(kv.value);
        }
        return self._expr_seq_scan(items, name);
    }
    if isinstance(e, uni.DictCompr) {
        return self._compr_scan(
            list(e.compr or []), [e.kv_pair.key, e.kv_pair.value], name
        );
    }
    if isinstance(e, uni.ListCompr) {
        return self._compr_scan(list(e.compr or []), [e.out_expr], name);
    }
    return _SCAN_UNSAFE;
}

impl PredicateCtrlFlowPass._expr_seq_scan(items: list[any], name: str) -> int {
    for it in items {
        r = self._expr_scan(it, name);
        if r != _SCAN_CLEAN {
            return r;
        }
    }
    return _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._bool_scan(e: uni.BoolExpr, name: str) -> int {
    vals = list(e.values or []);
    if not vals {
        return _SCAN_UNSAFE;
    }
    first = self._expr_scan(vals[0], name);
    if first != _SCAN_CLEAN {
        return first;
    }
    for v in vals[1:] {
        if self._expr_scan(v, name) == _SCAN_UNSAFE {
            return _SCAN_UNSAFE;
        }
    }
    return _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._compare_scan(e: uni.CompareExpr, name: str) -> int {
    rights = list(e.rights or []);
    if len(rights) == 1 {
        return self._expr_seq_scan([e.left, rights[0]], name);
    }
    r = self._expr_scan(e.left, name);
    if r != _SCAN_CLEAN {
        return r;
    }
    for x in rights {
        if self._expr_scan(x, name) == _SCAN_UNSAFE {
            return _SCAN_UNSAFE;
        }
    }
    return _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._ternary_scan(e: uni.IfElseExpr, name: str) -> int {
    rc = self._expr_scan(e.condition, name);
    if rc != _SCAN_CLEAN {
        return rc;
    }
    rt = self._expr_scan(e.value, name);
    rf = self._expr_scan(e.else_value, name);
    if (rt == _SCAN_UNSAFE) or (rf == _SCAN_UNSAFE) {
        return _SCAN_UNSAFE;
    }
    return (
        _SCAN_FATAL
        if ((rt == _SCAN_FATAL) and (rf == _SCAN_FATAL))
        else _SCAN_CLEAN
    );
}

impl PredicateCtrlFlowPass._trailer_scan(e: uni.AtomTrailer, name: str) -> int {
    r = self._expr_scan(e.target, name);
    if r != _SCAN_CLEAN {
        return r;
    }
    right = e.right;
    if isinstance(right, uni.IndexSlice) {
        ri = self._slice_scan(right, name);
        if ri != _SCAN_CLEAN {
            return ri;
        }
    } elif not isinstance(right, uni.Name) {
        return _SCAN_UNSAFE;
    }
    if e.is_null_ok {
        return _SCAN_CLEAN;
    }
    return _SCAN_FATAL if self._is_named(e.target, name) else _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._slice_scan(e: uni.IndexSlice, name: str) -> int {
    items: list[any] = [];
    for sl in (e.slices or []) {
        items.append(sl.start);
        items.append(sl.stop);
        items.append(sl.step);
    }
    return self._expr_seq_scan(items, name);
}

impl PredicateCtrlFlowPass._call_scan(c: uni.FuncCall, name: str) -> int {
    r = self._expr_scan(c.target, name);
    if r != _SCAN_CLEAN {
        return r;
    }
    for p in (c.params or []) {
        rp = self._expr_scan(p, name);
        if rp != _SCAN_CLEAN {
            return rp;
        }
    }
    return _SCAN_CLEAN if self._scan_call_pure(c) else _SCAN_UNSAFE;
}

impl PredicateCtrlFlowPass._compr_scan(
    compr: list[any], parts: list[any], name: str
) -> int {
    if not compr {
        return _SCAN_UNSAFE;
    }
    head: any = compr[0];
    r = self._expr_scan(head.collection, name);
    if r != _SCAN_CLEAN {
        return r;
    }
    if self._is_named(head.collection, name) {
        return _SCAN_FATAL;
    }
    rest: list[any] = list(parts);
    for ic in compr {
        holder: any = ic;
        rest.append(holder.collection);
        for cnd in (holder.conditional or []) {
            rest.append(cnd);
        }
    }
    for x in rest {
        if self._expr_scan(x, name) != _SCAN_CLEAN {
            return _SCAN_UNSAFE;
        }
    }
    return _SCAN_CLEAN;
}

impl PredicateCtrlFlowPass._scan_call_pure(c: uni.FuncCall) -> bool {
    t = c.target;
    if self._pure_callee(t) {
        return True;
    }
    if isinstance(t, uni.Name) {
        return (t.value in _PRECOND_BUILTINS)
            and ((t.sym is None) or symbol_is_builtin(t.sym));
    }
    if isinstance(t, uni.AtomTrailer) and isinstance(t.right, uni.Name) {
        base = t.target;
        if isinstance(base, uni.Name)
            and (base.value == 'torch')
            and (t.right.value in _PRECOND_TORCH_FNS)
            and self._pure_root(base) {
            return True;
        }
        return (t.right.value in _PURE_TENSOR_METHODS) and (not (c.params or []));
    }
    return False;
}

impl PredicateCtrlFlowPass._is_named(e: (uni.UniNode | None), name: str) -> bool {
    if (e is None) or (not isinstance(e, uni.Expr)) {
        return False;
    }
    inner = self._unwrap_expr(e);
    return isinstance(inner, uni.Name) and (inner.value == name);
}

impl PredicateCtrlFlowPass._local_store_targets(targets: Sequence[any]) -> bool {
    names: list[uni.Name] = [];
    for t in (targets or []) {
        items: list[any] = (
            list(t.values or [])
            if isinstance(t, (uni.TupleVal, uni.ListVal))
            else [t]
        );
        for it in items {
            if not isinstance(it, uni.Name) {
                return False;
            }
            names.append(it);
        }
    }
    if not names {
        return False;
    }
    mod = names[0].find_parent_of_type(uni.Module);
    if mod is None {
        return False;
    }
    outside = self._module_scope_names(mod);
    for n in names {
        if n.value in outside {
            return False;
        }
    }
    return True;
}
