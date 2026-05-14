static void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
        wgsl_tc_type_decl_const_or_let(tc, n);
        return;
    case WGSL_NODE_DECL_VAR:
        wgsl_tc_type_decl_var(tc, n);
        return;
    case WGSL_NODE_DECL_CONST_ASSERT:
        if (n->child_count >= 1) {
            wgsl_tc_type_expr(tc, n->children[0]);
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, n->children[0], "const_assert condition must be a bool");
            }
        }
        return;
    case WGSL_NODE_STMT_RETURN: {
        /* §9.4.10 — bare `return` only legal in a void function;
         * `return expr;` only legal in a non-void function whose
         * return type accepts the expr's type. */
        if (tc->continuing_depth > 0) {
            /* §9.4.9 — return is not allowed inside a continuing block. */
            wgsl_tc_error(tc, n,
                "'return' is not allowed inside a continuing block");
        }
        if (n->child_count == 0) {
            if (tc->current_fn_return) {
                wgsl_tc_error(tc, n,
                    "'return;' requires an expression in a function "
                    "with return type %s",
                    wgsl_type_kind_name((WGSLTypeKind)
                        tc->current_fn_return->kind));
            }
        } else {
            wgsl_tc_type_expr(tc, n->children[0]);
            WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (!tc->current_fn_return) {
                wgsl_tc_error(tc, n->children[0],
                    "'return' with expression in a void function");
            } else if (rt && wgsl_tc_init_type_mismatch(rt, tc->current_fn_return)) {
                wgsl_tc_error(tc, n->children[0],
                    "return value of type %s does not match function "
                    "return type %s",
                    wgsl_type_kind_name((WGSLTypeKind)rt->kind),
                    wgsl_type_kind_name((WGSLTypeKind)
                        tc->current_fn_return->kind));
            }
        }
        return;
    }
    case WGSL_NODE_STMT_FN_CALL:
        if (n->child_count >= 1) {
            WGSLNode *expr = n->children[0];
            wgsl_tc_type_expr(tc, expr);
            if (expr && expr->kind != WGSL_NODE_EXPR_CALL) {
                wgsl_tc_error(tc, n,
                    "expression statement must be a function call");
            }
            /* §9.5 — call statement discards the result, so the callee
             * must not carry `@must_use`. */
            WGSLNode *call = expr;
            if (call && call->kind == WGSL_NODE_EXPR_CALL &&
                call->child_count >= 1)
            {
                WGSLNode *callee = call->children[0];
                WGSLSymbol *csym = NULL;
                if (callee && callee->kind == WGSL_NODE_EXPR_IDENT) {
                    csym = wgsl_node_resolved_symbol(callee);
                } else if (callee &&
                           callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
                           callee->child_count >= 1)
                {
                    csym = wgsl_node_resolved_symbol(callee->children[0]);
                }
                int constructor =
                    csym && (csym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                             csym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
                             csym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                             csym->kind == WGSL_SYM_TYPE_ALIAS ||
                             csym->kind == WGSL_SYM_STRUCT);
                int must_use = csym &&
                    ((csym->flags & WGSL_SYM_FLAG_MUST_USE) ||
                     constructor ||
                     (csym->kind == WGSL_SYM_PREDECLARED_FN &&
                      wgsl_tc_builtin_must_use(csym->name, csym->name_len)));
                if (must_use) {
                    wgsl_tc_error(tc, n,
                        "call to '@must_use' function '%.*s' must not "
                        "discard its result",
                        (int)csym->name_len, csym->name);
                }
            }
        }
        return;
    case WGSL_NODE_STMT_PHONY_ASSIGN:
        if (n->child_count >= 1) {
            wgsl_tc_type_expr(tc, n->children[0]);
            /* §9.2.2 — RHS must be constructible / pointer / texture /
             * sampler.  Same shape as the function-arg gate. */
            WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (rt) {
                int ok = (rt->kind == WGSL_TYPE_PTR) ||
                         (rt->kind == WGSL_TYPE_TEXTURE) ||
                         (rt->kind == WGSL_TYPE_SAMPLER) ||
                         wgsl_tc_is_constructible_deep(tc, rt);
                if (!ok) {
                    wgsl_tc_error(tc, n->children[0],
                        "right-hand side of '_ = …' must be a "
                        "constructible, pointer, texture, or sampler "
                        "value (got %s)",
                        wgsl_type_kind_name((WGSLTypeKind)rt->kind));
                }
            }
        }
        if (n->child_count >= 2) wgsl_tc_type_expr(tc, n->children[1]);
        return;
    case WGSL_NODE_STMT_ASSIGN:
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT: {
        for (uint32_t i = 0; i < n->child_count; i++) wgsl_tc_type_expr(tc, n->children[i]);
        if (n->child_count > 0 && n->children[0]) {
            WGSLNode *lhs = n->children[0];
            if (!wgsl_typecheck_is_ref(tc, lhs)) {
                wgsl_tc_error(tc, lhs, "cannot assign to a value; left-hand side must be a reference");
            } else {
                WGSLAddressSpace as = WGSL_AS_NONE;
                WGSLAccessMode   am = WGSL_ACCESS_NONE;
                wgsl_tc_get_ref_space_and_mode(tc, lhs, &as, &am);
                if (am == WGSL_ACCESS_READ) {
                    wgsl_tc_error(tc, lhs, "cannot assign to read-only reference");
                }
            }
            /* §9.2.1 RHS type vs LHS store type. */
            if (n->kind == WGSL_NODE_STMT_ASSIGN && n->child_count >= 2) {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[1]);
                if (lt && rt && wgsl_tc_init_type_mismatch(rt, lt)) {
                    wgsl_tc_error(tc, n->children[1],
                        "value of type %s cannot be converted to the "
                        "assignment target's store type %s",
                        wgsl_type_kind_name((WGSLTypeKind)rt->kind),
                        wgsl_type_kind_name((WGSLTypeKind)lt->kind));
                }
            }
            if (n->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN &&
                n->child_count >= 2)
            {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                WGSLTypeInfo *rt = wgsl_tc_store_type_of(tc, n->children[1]);
                if (lt && rt) validate_compound_assignment(tc, n, lt, rt);
            }
            /* §9.3 — increment / decrement operand must be a ref to a
             * concrete integer scalar (i32 / u32).  The ref check above
             * already covers the let / param cases. */
            if ((n->kind == WGSL_NODE_STMT_INCREMENT ||
                 n->kind == WGSL_NODE_STMT_DECREMENT))
            {
                WGSLTypeInfo *lt = wgsl_tc_store_type_of(tc, lhs);
                int ok_int_scalar =
                    lt && (lt == tc->types->t_i32 || lt == tc->types->t_u32);
                if (lt && !ok_int_scalar) {
                    wgsl_tc_error(tc, lhs,
                        "'%s' requires a concrete integer scalar (i32 "
                        "or u32) reference; got %s",
                        n->kind == WGSL_NODE_STMT_INCREMENT ? "++" : "--",
                        wgsl_type_kind_name((WGSLTypeKind)lt->kind));
                }
            }
        }
        return;
    }
    case WGSL_NODE_STMT_BREAK:
        /* §9.4.6 — `break` inside a continuing block must target an
         * inner loop / switch; otherwise it would exit the loop's
         * continuing block, which is forbidden. */
        if (tc->continuing_depth > 0 &&
            tc->inner_break_target_depth == 0)
        {
            wgsl_tc_error(tc, n,
                "'break' is not allowed to exit the enclosing loop's "
                "continuing block");
        }
        return;
    case WGSL_NODE_STMT_BREAK_IF:
        for (uint32_t i = 0; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
        if (n->child_count >= 1) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, n->children[0],
                    "'break if' condition must evaluate to bool");
            }
        }
        return;
    case WGSL_NODE_STMT_CONTINUE:
        /* §9.4.8 / §9.7 — `continue` inside a continuing block must
         * target an inner loop; otherwise it would re-enter the
         * enclosing loop's continuing block, which is forbidden. */
        if (tc->continuing_depth > 0 &&
            tc->inner_continue_target_depth == 0)
        {
            wgsl_tc_error(tc, n,
                "'continue' is not allowed inside a continuing block");
        }
        return;
    case WGSL_NODE_STMT_CONTINUING: {
        /* Bump continuing_depth so any lexically-immediate STMT_RETURN
         * encountered during recursion is flagged (return is forbidden
         * regardless of inner loop nesting since there's no inner fn
         * boundary).  Reset both inner-target depths so a bare `break`
         * / `continue` directly in this body (not inside an inner
         * loop / switch) can be flagged.  §9.4.6 / §9.4.8 / §9.4.9. */
        tc->continuing_depth += 1;
        int saved_break = tc->inner_break_target_depth;
        int saved_cont  = tc->inner_continue_target_depth;
        tc->inner_break_target_depth    = 0;
        tc->inner_continue_target_depth = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        tc->inner_break_target_depth    = saved_break;
        tc->inner_continue_target_depth = saved_cont;
        tc->continuing_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_SWITCH: {
        /* §9.4.2 — selector + case structural checks. */
        /* §9.4.6 — switch is also a `break` target, so bump the inner
         * break-target depth for the duration of the body walk. */
        tc->inner_break_target_depth += 1;
        WGSLTypeInfo *sel_t = NULL;
        if (n->child_count >= 1 && n->children[0]) {
            wgsl_tc_type_expr(tc, n->children[0]);
            sel_t = wgsl_tc_store_type_of(tc, n->children[0]);
            int ok_sel = sel_t &&
                (sel_t == tc->types->t_i32 || sel_t == tc->types->t_u32 ||
                 sel_t == tc->types->t_abstract_int);
            if (sel_t && !ok_sel) {
                wgsl_tc_error(tc, n->children[0],
                    "switch selector must be a concrete integer scalar "
                    "(i32 or u32); got %s",
                    wgsl_type_kind_name((WGSLTypeKind)sel_t->kind));
            }
        }
        int default_count = 0;
        /* Track case-selector const-values seen, to flag duplicates.
         * Stack-cap is fine for realistic shaders. */
        int64_t seen[256];
        int seen_count = 0;
        WGSLTypeInfo *case_common_t = NULL;
        for (uint32_t k = 1; k < n->child_count; k++) {
            WGSLNode *cc = n->children[k];
            if (!cc || cc->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
            uint32_t sel_cnt   = (uint32_t)(cc->payload[0] & 0xFFFFFFFFu);
            uint32_t is_def_aln = (uint32_t)((cc->payload[0] >> 32) & 1u);
            uint32_t has_def_in_list = (uint32_t)((cc->payload[0] >> 33) & 1u);
            if (is_def_aln) {
                default_count += 1;
                walk_stmt(tc, cc);
                continue;
            }
            /* `case X, default, Y:` — the default keyword inside the
             * selector list counts toward the switch's default tally
             * (per spec §9.4.2 / §18 grammar). */
            if (has_def_in_list) {
                default_count += 1;
            }
            /* Each selector child first, then the body. */
            for (uint32_t s = 0; s < sel_cnt && s < cc->child_count; s++) {
                WGSLNode *se = cc->children[s];
                if (!se) continue;
                wgsl_tc_type_expr(tc, se);
                WGSLTypeInfo *st = wgsl_tc_store_type_of(tc, se);
                /* Selector ↔ case type compatibility. */
                int case_compatible = sel_t && st &&
                    !wgsl_tc_init_type_mismatch(st, sel_t);
                if (sel_t == tc->types->t_abstract_int &&
                    (st == tc->types->t_i32 ||
                     st == tc->types->t_u32 ||
                     st == tc->types->t_abstract_int))
                {
                    case_compatible = 1;
                }
                if ((sel_t == tc->types->t_i32 ||
                     sel_t == tc->types->t_u32) &&
                    st == tc->types->t_abstract_int)
                {
                    case_compatible = 1;
                }
                if (sel_t && st && !case_compatible) {
                    wgsl_tc_error(tc, se,
                        "case selector type %s does not match switch "
                        "selector type %s",
                        wgsl_type_kind_name((WGSLTypeKind)st->kind),
                        wgsl_type_kind_name((WGSLTypeKind)sel_t->kind));
                }
                if (st == tc->types->t_i32 || st == tc->types->t_u32) {
                    if (!case_common_t) {
                        case_common_t = st;
                    } else if (case_common_t != st) {
                        wgsl_tc_error(tc, se,
                            "case selector type %s does not match earlier "
                            "case selector type %s",
                            wgsl_type_kind_name((WGSLTypeKind)st->kind),
                            wgsl_type_kind_name(
                                (WGSLTypeKind)case_common_t->kind));
                    }
                }
                /* §9.4.2: each case selector must be a const-expression.
                 * Run consteval, then either fold the value into the
                 * duplicate-check set or surface the spec error.  In
                 * either branch, drop the consteval-internal diagnostic
                 * noise so we present a single clean message. */
                if (tc->cev && tc->diag) {
                    size_t sc = tc->diag->count;
                    int    se_e = tc->diag->error_count;
                    int    cv = tc->cev->had_error;
                    WGSLValue v = {0};
                    int eval_ok = wgsl_consteval_expr(tc->cev, se, &v) &&
                                  v.kind == WGSL_VAL_INT;
                    /* Always roll the consteval bag back; we'll emit
                     * the right diagnostic at this layer instead. */
                    tc->diag->count       = sc;
                    tc->diag->error_count = se_e;
                    tc->cev->had_error    = cv;
                    if (eval_ok) {
                        int dup = 0;
                        for (int x = 0; x < seen_count; x++) {
                            if (seen[x] == v.u.i) { dup = 1; break; }
                        }
                        if (dup) {
                            wgsl_tc_error(tc, se,
                                "duplicate case selector value %lld",
                                (long long)v.u.i);
                        } else if (seen_count <
                                   (int)(sizeof seen / sizeof seen[0]))
                        {
                            seen[seen_count++] = v.u.i;
                        }
                    } else {
                        wgsl_tc_error(tc, se,
                            "case selector must be a const-expression");
                    }
                }
            }
            /* Walk the body. */
            if (sel_cnt < cc->child_count) {
                walk_stmt(tc, cc->children[sel_cnt]);
            }
        }
        if (default_count == 0) {
            wgsl_tc_error(tc, n,
                "switch statement must have exactly one default clause");
        } else if (default_count > 1) {
            wgsl_tc_error(tc, n,
                "switch statement must not have more than one default "
                "clause (found %d)", default_count);
        }
        tc->inner_break_target_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE: {
        /* §9.4.6 / §9.4.8 — these are both `break` and `continue`
         * targets, so bump both inner-target depths.  Switch is a
         * `break` target only (handled in STMT_SWITCH above). */
        tc->inner_break_target_depth    += 1;
        tc->inner_continue_target_depth += 1;
        check_continue_past_decl(tc, n);
        /* §9.4.3 / §9.4.4 — `while` cond and `for` cond must be bool.
         * For `while`, child[0] is the cond.  For `for`, the cond is
         * whichever child is an EXPR node (init / update are stmts). */
        WGSLNode *cond_check = NULL;
        const char *cond_label = NULL;
        if (n->kind == WGSL_NODE_STMT_WHILE && n->child_count >= 1) {
            cond_check = n->children[0];
            cond_label = "while";
        } else if (n->kind == WGSL_NODE_STMT_FOR) {
            for (uint32_t i = 0; i < n->child_count; i++) {
                WGSLNode *c = n->children[i];
                if (c && c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                       c->kind <= WGSL_NODE_EXPR_INDIRECTION)
                {
                    cond_check = c;
                    cond_label = "for";
                    break;
                }
            }
            uint32_t flags = (uint32_t)n->payload[0];
            if (flags & 4u) {
                uint32_t update_index = 0;
                if (flags & 1u) update_index += 1;
                if (flags & 2u) update_index += 1;
                if (update_index < n->child_count) {
                    WGSLNode *upd = n->children[update_index];
                    if (upd &&
                        (upd->kind == WGSL_NODE_DECL_VAR ||
                         upd->kind == WGSL_NODE_DECL_LET ||
                         upd->kind == WGSL_NODE_DECL_CONST))
                    {
                        wgsl_tc_error(tc, upd,
                            "for update must not be a declaration");
                    }
                }
            }
        }
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        if (cond_check) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, cond_check);
            if (t && t != tc->types->t_bool) {
                wgsl_tc_error(tc, cond_check,
                    "'%s' condition must evaluate to bool", cond_label);
            }
        }
        tc->inner_break_target_depth    -= 1;
        tc->inner_continue_target_depth -= 1;
        return;
    }
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_COMPOUND:
        /* Mixed scope: some children are exprs (e.g. if-cond), others are
         * statements (the block body).  Walk blindly — for each child,
         * try expr-typing first by kind, else recurse as stmt. */
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                wgsl_tc_type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        /* §9.4.1 — `if` / `else if` condition (child[0]) must be bool. */
        if ((n->kind == WGSL_NODE_STMT_IF ||
             n->kind == WGSL_NODE_STMT_ELSE_IF) &&
            n->child_count >= 1)
        {
            WGSLNode *c = n->children[0];
            if (c && c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                   c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, c);
                if (t && t != tc->types->t_bool) {
                    wgsl_tc_error(tc, c,
                        "'if' condition must evaluate to bool");
                }
            }
        }
        return;
    default:
        return;
    }
}

