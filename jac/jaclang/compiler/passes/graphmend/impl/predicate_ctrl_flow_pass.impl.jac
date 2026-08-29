import from jaclang.compiler.tools.unparse_pass { unparse_node }
import jaclang.compiler.frontend.unitree as uni;
import from jaclang.compiler.frontend.constant { GraphBreakKind, Tokens as Tok }
import from jaclang.compiler.passes.graphmend.ast_edit {
    facts_for,
    gen_name,
    replace_stmt,
    split_params,
    split_params_keyed,
    torch_call
}
import from jaclang.compiler.passes.graphmend.scope_facts {
    module_reaches_autograd,
    region_in_inference_ctx
}
import from jaclang.compiler.passes.graphmend.torch_attr_table {
    cond_is_data_dependent,
    cond_materializes_scalar,
    is_torch_attr_target
}

impl PredicateCtrlFlowPass.before_pass -> None {
    self._enabled = (self.prog.options.graphmend is True)
        and "where" not in (self.prog.options.graphmend_disable or []);
    if not self._enabled {
        self.prune_signal = True;
        return;
    }
    self._autograd_reachable = module_reaches_autograd(self.ir_in);
}

impl PredicateCtrlFlowPass.exit_if_stmt(nd: uni.IfStmt) -> None {
    if not self._enabled {
        return;
    }
    if facts_for(nd).breaks.get(nd) != GraphBreakKind.DYN_CTRL_FL {
        return;
    }
    before = self._counter;
    self._where_family(nd);
    if self._counter == before {
        self._strengthen_guard(nd);
    }
}

impl PredicateCtrlFlowPass._where_family(nd: uni.IfStmt) -> None {
    grad_gated = self._autograd_reachable and (not region_in_inference_ctx(nd));
    eb = nd.else_body;
    if eb is None {
        self._rewrite_predicated_update(nd, grad_gated);
        return;
    }
    if grad_gated {
        return;
    }
    if not isinstance(eb, uni.ElseStmt) {
        return;
    }
    tb = nd.body;
    fb = eb.body;
    if (not tb) or (not fb) {
        return;
    }
    if cond_materializes_scalar(nd.condition) {
        return;
    }
    if (len(tb) == 1) and (len(fb) == 1) {
        a0 = tb[0];
        b0 = fb[0];
        if isinstance(a0, uni.Assignment) and isinstance(b0, uni.Assignment) {
            lhs = self._same_lhs(a0, b0);
            av = a0.value;
            bv = b0.value;
            if (lhs is not None)
                and isinstance(av, uni.Expr)
                and isinstance(bv, uni.Expr)
                and (not self._has_none(av, bv)) {
                self._rewrite_assignment(nd, lhs, av, bv);
            }
        } elif isinstance(a0, uni.ReturnStmt) and isinstance(b0, uni.ReturnStmt) {
            if (a0.expr is not None)
                and (b0.expr is not None)
                and (not self._has_none(a0.expr, b0.expr)) {
                self._rewrite_return(nd, a0.expr, b0.expr);
            }
        } elif isinstance(a0, uni.ExprStmt) and isinstance(b0, uni.ExprStmt) {
            self._merge_common_call(nd, a0.expr, b0.expr, [], []);
        }
        return;
    }
    a_last = tb[-1];
    b_last = fb[-1];
    if isinstance(a_last, uni.ExprStmt) and isinstance(b_last, uni.ExprStmt) {
        (ts, fs) = (tb[:-1], fb[:-1]);
        if (not (self._setup_writes(ts) & self._setup_writes(fs)))
            and self._setups_safe(ts, nd)
            and self._setups_safe(fs, nd) {
            self._merge_common_call(nd, a_last.expr, b_last.expr, ts, fs);
        }
    }
}

impl PredicateCtrlFlowPass._rewrite_assignment(
    nd: uni.IfStmt, lhs: uni.Name, true_e: uni.Expr, false_e: uni.Expr
) -> None {
    if self._has_call(true_e) or self._has_call(false_e) {
        return;
    }
    if self._provably_non_tensor(true_e) or self._provably_non_tensor(false_e) {
        return;
    }
    (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
    where_call = torch_call(nd, 'where', [c_ref, t_ref, f_ref]);
    replace_stmt(nd, stmts + [self._assign(lhs, where_call)]);
}

impl PredicateCtrlFlowPass._rewrite_return(
    nd: uni.IfStmt, true_e: uni.Expr, false_e: uni.Expr
) -> None {
    if self._has_call(true_e) or self._has_call(false_e) {
        return;
    }
    if self._provably_non_tensor(true_e) or self._provably_non_tensor(false_e) {
        return;
    }
    (stmts, c_ref, t_ref, f_ref) = self._hoist(nd, nd.condition, true_e, false_e);
    where_call = torch_call(nd, 'where', [c_ref, t_ref, f_ref]);
    ret = uni.ReturnStmt(expr=where_call, kid=[where_call]);
    replace_stmt(nd, stmts + [ret]);
}

impl PredicateCtrlFlowPass._rewrite_predicated_update(
    nd: uni.IfStmt, grad_gated: bool
) -> None {
    updates = self._update_targets(nd);
    if updates is None {
        return;
    }
    if not self._torch_bound(nd) {
        return;
    }
    (static_cond, tensor_cond) = self._static_and_prefix(nd.condition);
    if tensor_cond is None {
        return;
    }
    lowered = self._lower_tensor_bool(nd, tensor_cond, False);
    if lowered is None {
        return;
    }
    n = self._counter;
    self._counter += 1;
    cond_tmp = gen_name(nd, f"__gm_cond_{n}", True);
    new_stmts: list = [self._assign(cond_tmp, lowered)];
    for (stmt, tgt, value) in updates {
        if tgt is None {
            new_stmts.append(stmt);
            continue;
        }
        cref = gen_name(nd, f"__gm_cond_{n}", False);
        prev = gen_name(tgt, tgt.value, False);
        sel = torch_call(nd, 'where', [cref, value, prev]);
        new_stmts.append(self._assign(gen_name(tgt, tgt.value, True), sel));
    }
    if not grad_gated {
        if static_cond is None {
            replace_stmt(nd, new_stmts);
            return;
        }
        holder: any = nd;
        setattr(holder, 'condition', static_cond);
        setattr(holder, 'body', new_stmts);
        nd.set_kids([static_cond] + new_stmts);
        return;
    }
    self._specialize_on_grad(nd, static_cond, new_stmts);
}

impl PredicateCtrlFlowPass._specialize_on_grad(
    nd: uni.IfStmt, static_cond: (uni.Expr | None), new_stmts: list
) -> None {
    kept_cond = nd.condition;
    kept_body = list(nd.body or []);
    kept = uni.IfStmt(
        condition=kept_cond, body=kept_body, kid=[kept_cond] + kept_body
    );
    taken: list = new_stmts;
    if static_cond is not None {
        inner = uni.IfStmt(
            condition=static_cond, body=new_stmts, kid=[static_cond] + new_stmts
        );
        taken = [inner];
    }
    otherwise = uni.ElseStmt(body=taken, kid=taken);
    grad_cond = torch_call(nd, 'is_grad_enabled', []);
    holder: any = nd;
    setattr(holder, 'condition', grad_cond);
    setattr(holder, 'body', [kept]);
    setattr(holder, 'else_body', otherwise);
    nd.set_kids([grad_cond, kept, otherwise]);
}

impl PredicateCtrlFlowPass._unwrap_expr(e: uni.Expr) -> uni.Expr {
    cur: uni.Expr = e;
    while True {
        if isinstance(cur, uni.CfgExpr) {
            cur = cur.expr;
            continue;
        }
        if isinstance(cur, uni.AtomUnit) and isinstance(cur.value, uni.Expr) {
            cur = cur.value;
            continue;
        }
        return cur;
    }
}

impl PredicateCtrlFlowPass._bool_join(e: uni.Expr) -> (str | None) {
    if (not isinstance(e, uni.BoolExpr)) or (e.op is None) {
        return None;
    }
    if e.op.name == Tok.KW_AND {
        return 'and';
    }
    if e.op.name == Tok.KW_OR {
        return 'or';
    }
    return None;
}

impl PredicateCtrlFlowPass._lower_tensor_bool(
    at: uni.UniNode, e: uni.Expr, merged: bool
) -> (uni.Expr | None) {
    term = self._unwrap_expr(e);
    join = self._bool_join(term);
    if (join is not None) and isinstance(term, uni.BoolExpr) {
        fname = 'logical_and' if (join == 'and') else 'logical_or';
        parts: list = [];
        for v in term.values {
            lv = self._lower_tensor_bool(at, v, True);
            if lv is None {
                return None;
            }
            parts.append(lv);
        }
        acc = parts[0];
        for p in parts[1:] {
            acc = torch_call(at, fname, [acc, p]);
        }
        return acc;
    }
    if isinstance(term, uni.UnaryExpr)
        and (term.op is not None)
        and (term.op.name == Tok.NOT) {
        inner = self._lower_tensor_bool(at, term.operand, merged);
        if inner is None {
            return None;
        }
        return torch_call(at, 'logical_not', [inner]);
    }
    if not self._tensor_bool_atom(term) {
        return None;
    }
    if merged and (not self._pure_value(term)) {
        return None;
    }
    return term;
}

impl PredicateCtrlFlowPass._merge_common_call(
    nd: uni.IfStmt,
    ca: uni.Expr,
    cb: uni.Expr,
    true_setups: Sequence[uni.CodeBlockStmt],
    false_setups: Sequence[uni.CodeBlockStmt]
) -> None {
    if (not isinstance(ca, uni.FuncCall)) or (not isinstance(cb, uni.FuncCall)) {
        return;
    }
    if (
        unparse_node(ca.target).replace(' ', '')
        != unparse_node(cb.target).replace(' ', '')
    ) {
        return;
    }
    (a_pos, a_kw) = split_params_keyed(ca.params);
    (b_pos, b_kw) = split_params_keyed(cb.params);
    if len(a_pos) != len(b_pos) {
        return;
    }
    if set(a_kw.keys()) != set(b_kw.keys()) {
        return;
    }
    for k in a_kw {
        if unparse_node(a_kw[k].value) != unparse_node(b_kw[k].value) {
            return;
        }
    }
    diff = [];
    for i in range(len(a_pos)) {
        if unparse_node(a_pos[i]) != unparse_node(b_pos[i]) {
            diff.append(i);
        }
    }
    if not diff {
        return;
    }
    for i in diff {
        if self._has_call(a_pos[i]) or self._has_call(b_pos[i]) {
            return;
        }
        if self._provably_non_tensor(a_pos[i]) or self._provably_non_tensor(b_pos[i]) {
            return;
        }
    }
    for s in (list(true_setups) + list(false_setups)) {
        if self._is_eq_assert(s) {
            return;
        }
    }
    n = self._counter;
    self._counter += 1;
    cond_tmp = gen_name(nd, f"__gm_cond_{n}", True);
    cond_assign = self._assign(cond_tmp, nd.condition);
    merged: list = [];
    for i in range(len(a_pos)) {
        if i in diff {
            cref = gen_name(nd, f"__gm_cond_{n}", False);
            merged.append(torch_call(nd, 'where', [cref, a_pos[i], b_pos[i]]));
        } else {
            merged.append(a_pos[i]);
        }
    }
    for k in a_kw {
        merged.append(a_kw[k]);
    }
    call_stmt = self._merged_call_stmt(nd, ca, a_pos, merged, diff);
    gated_true = self._gate_asserts(list(true_setups), n, True);
    gated_false = self._gate_asserts(list(false_setups), n, False);
    new_nodes = [cond_assign] + gated_true + gated_false + [call_stmt];
    replace_stmt(nd, new_nodes);
}

impl PredicateCtrlFlowPass._has_call(e: uni.UniNode) -> bool {
    return isinstance(e, uni.FuncCall) or bool(e.get_all_sub_nodes(uni.FuncCall));
}

impl PredicateCtrlFlowPass._has_none(a: uni.Expr, b: uni.Expr) -> bool {
    return isinstance(a, uni.Null) or isinstance(b, uni.Null);
}

impl PredicateCtrlFlowPass._provably_non_tensor(e: uni.Expr) -> bool {
    return self._scalar_literal(e) or self._non_tensor_container(e);
}

impl PredicateCtrlFlowPass._scalar_literal(e: uni.Expr) -> bool {
    if isinstance(e, uni.AtomUnit) {
        inner = e.value;
        return isinstance(inner, uni.Expr) and self._scalar_literal(inner);
    }
    if isinstance(e, uni.UnaryExpr) {
        return (e.op is not None)
            and (e.op.name in [Tok.PLUS, Tok.MINUS, Tok.BW_NOT, Tok.NOT])
            and self._scalar_literal(e.operand);
    }
    return isinstance(e, (uni.Int, uni.Float, uni.Bool));
}

impl PredicateCtrlFlowPass._non_tensor_container(e: uni.Expr) -> bool {
    if isinstance(e, uni.AtomUnit) {
        inner = e.value;
        return isinstance(inner, uni.Expr) and self._non_tensor_container(inner);
    }
    if isinstance(
        e,
        (
            uni.String,
            uni.MultiString,
            uni.FString,
            uni.ListVal,
            uni.SetVal,
            uni.TupleVal,
            uni.DictVal,
            uni.ListCompr,
            uni.DictCompr
        )
    ) {
        return True;
    }
    if isinstance(e, uni.BinaryExpr) {
        return self._non_tensor_container(e.left)
            or self._non_tensor_container(e.right);
    }
    return False;
}

impl PredicateCtrlFlowPass._same_lhs(
    a: uni.Assignment, b: uni.Assignment
) -> (uni.Name | None) {
    if (len(a.target) != 1) or (len(b.target) != 1) {
        return None;
    }
    ta = a.target[0];
    tb = b.target[0];
    if (
        isinstance(ta, uni.Name) and isinstance(tb, uni.Name) and (ta.value == tb.value)
    ) {
        return gen_name(ta, ta.value, True);
    }
    return None;
}

impl PredicateCtrlFlowPass._assign(
    target: uni.Name, value: uni.Expr
) -> uni.Assignment {
    return uni.Assignment(
        target=[target], value=value, type_tag=None, mutable=True, kid=[target, value]
    );
}

impl PredicateCtrlFlowPass._hoist(
    at: uni.UniNode, cond: uni.Expr, t: uni.Expr, f: uni.Expr
) -> tuple[list, uni.Name, uni.Name, uni.Name] {
    n = self._counter;
    (stmts, cond_ref) = self._hoist_cond(at, cond);
    true_tmp = gen_name(at, f"__gm_true_{n}", True);
    false_tmp = gen_name(at, f"__gm_false_{n}", True);
    stmts = stmts + [self._assign(true_tmp, t), self._assign(false_tmp, f)];
    true_ref = gen_name(at, f"__gm_true_{n}", False);
    false_ref = gen_name(at, f"__gm_false_{n}", False);
    return (stmts, cond_ref, true_ref, false_ref);
}

impl PredicateCtrlFlowPass._hoist_cond(
    at: uni.UniNode, cond: uni.Expr
) -> tuple[list, uni.Name] {
    n = self._counter;
    self._counter += 1;
    cond_tmp = gen_name(at, f"__gm_cond_{n}", True);
    cond_ref = gen_name(at, f"__gm_cond_{n}", False);
    return ([self._assign(cond_tmp, cond)], cond_ref);
}

impl PredicateCtrlFlowPass._is_assert_async(stmt: uni.UniNode) -> bool {
    if not isinstance(stmt, uni.ExprStmt) {
        return False;
    }
    call = stmt.expr;
    if not isinstance(call, uni.FuncCall) {
        return False;
    }
    return is_torch_attr_target(call.target, '_assert_async');
}

impl PredicateCtrlFlowPass._is_eq_assert(stmt: uni.UniNode) -> bool {
    if not isinstance(stmt, uni.ExprStmt) {
        return False;
    }
    call = stmt.expr;
    return (
        isinstance(call, uni.FuncCall)
        and isinstance(call.target, uni.Name)
        and (call.target.value == '__jac_tensor_eq_assert__')
    );
}

impl PredicateCtrlFlowPass._gate_asserts(
    setups: list, n: int, taken_when_true: bool
) -> list {
    out = [];
    for s in setups {
        if (not self._is_assert_async(s)) or (not isinstance(s, uni.ExprStmt)) {
            out.append(s);
            continue;
        }
        call = s.expr;
        if not isinstance(call, uni.FuncCall) {
            out.append(s);
            continue;
        }
        pos = split_params(call.params)[0];
        if not pos {
            out.append(s);
            continue;
        }
        check = pos[0];
        cref = gen_name(s, f"__gm_cond_{n}", False);
        gate: uni.Expr = (
            torch_call(s, 'logical_not', [cref]) if taken_when_true else cref
        );
        new_check = torch_call(s, 'logical_or', [gate, check]);
        new_params = [new_check] + list(call.params[1:]);
        new_call = uni.FuncCall(
            target=call.target,
            params=new_params,
            genai_call=None,
            kid=[call.target] + new_params
        );
        out.append(uni.ExprStmt(expr=new_call, in_fstring=False, kid=[new_call]));
    }
    return out;
}

impl PredicateCtrlFlowPass._merged_call_stmt(
    nd: uni.IfStmt,
    ca: uni.FuncCall,
    a_pos: list,
    merged: list,
    diff: list
) -> uni.ExprStmt {
    tgt = ca.target;
    if (diff == [1]) and (len(a_pos) >= 2) and isinstance(tgt, uni.AtomTrailer) {
        rt = tgt.right;
        first = a_pos[0];
        if isinstance(rt, uni.Name) and isinstance(first, uni.String) {
            if rt.value == 'register_buffer' {
                bufname = first.value.strip('"').strip("'");
                if bufname and bufname.isidentifier() {
                    if self._buffer_preregistered(nd, bufname) {
                        recv = tgt.target;
                        buf_nm = gen_name(nd, bufname, False);
                        buf_ref = uni.AtomTrailer(
                            target=recv,
                            right=buf_nm,
                            is_attr=True,
                            is_null_ok=False,
                            kid=[recv, buf_nm]
                        );
                        copy_nm = gen_name(nd, 'copy_', False);
                        copy_ref = uni.AtomTrailer(
                            target=buf_ref,
                            right=copy_nm,
                            is_attr=True,
                            is_null_ok=False,
                            kid=[buf_ref, copy_nm]
                        );
                        val = merged[1];
                        copy_call = uni.FuncCall(
                            target=copy_ref,
                            params=[val],
                            genai_call=None,
                            kid=[copy_ref, val]
                        );
                        return uni.ExprStmt(
                            expr=copy_call, in_fstring=False, kid=[copy_call]
                        );
                    }
                }
            }
        }
    }
    new_call = uni.FuncCall(
        target=ca.target, params=merged, genai_call=None, kid=[ca.target] + merged
    );
    return uni.ExprStmt(expr=new_call, in_fstring=False, kid=[new_call]);
}

impl PredicateCtrlFlowPass._buffer_preregistered(
    nd: uni.IfStmt, bufname: str
) -> bool {
    owner = nd.find_parent_of_type(uni.Module);
    if owner is None {
        return False;
    }
    site_ids = [];
    for c in nd.get_all_sub_nodes(uni.FuncCall) {
        site_ids.append(id(c));
    }
    for call in owner.get_all_sub_nodes(uni.FuncCall) {
        if id(call) in site_ids {
            continue;
        }
        ctgt = call.target;
        if not isinstance(ctgt, uni.AtomTrailer) {
            continue;
        }
        cright = ctgt.right;
        if not isinstance(cright, uni.Name) {
            continue;
        }
        if cright.value != 'register_buffer' {
            continue;
        }
        (pos, kw) = split_params_keyed(call.params);
        if len(pos) < 1 {
            continue;
        }
        p0 = pos[0];
        if not isinstance(p0, uni.String) {
            continue;
        }
        if p0.value.strip('"').strip("'") == bufname {
            return True;
        }
    }
    return False;
}
