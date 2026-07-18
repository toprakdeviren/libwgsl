/**
 * @file pointers_block.c — uniformity analysis.
 */
#include "internal/uniformity_priv.h"

/* §15.2.4 — parameter desugaring for uniformity.
 *
 * A function parameter is uniform inside its callee when every reachable
 * call site passes a uniform actual.  Entry-point parameters are seeded
 * as non-uniform unless their builtin is spec-uniform.  This covers the
 * same pointer actual desugaring as §15.2.4 and also lets ordinary
 * helper parameters avoid spurious declaration-time diagnostics.
 *
 * Start optimistic (all non-entry parameters uniform) and monotonically
 * mark params non-uniform as call sites prove it.  Iteration propagates
 * through forwarding calls such as `fn f(p) { g(p); }`. */
/* PtrParamCtx: uniformity_priv.h */
void scan_pointer_param_calls(PtrParamCtx *pcx, WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_CALL) {
        WGSLSymbol *callee = call_user_fn_sym(n);
        if (callee) {
            uint32_t argc = n->child_count > 0 ? n->child_count - 1 : 0;
            for (uint32_t a = 0; a < argc; a++) {
                WGSLSymbol *param =
                    uniform_fn_param_symbol(pcx->ucx->v, callee, a);
                if (!uniform_is_param(param)) continue;
                int idx = wgsl_val_sym_index(pcx->ucx->v, param);
                if (idx < 0 || (size_t)idx >= pcx->ucx->nsyms) continue;
                if (!is_uniform_expr(pcx->ucx, n->children[1 + a]) &&
                    !pcx->nonuniform[idx])
                {
                    pcx->nonuniform[idx] = 1;
                    *pcx->changed = 1;
                }
            }
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_pointer_param_calls(pcx, n->children[i]);
    }
}

uint8_t *compute_param_nonuniform(WGSLValidator *v) {
    if (!v || !v->res) return NULL;
    size_t n = v->res->all_decl_count;
    uint8_t *nonuniform = (uint8_t *)calloc(n, sizeof *nonuniform);
    if (!nonuniform) return NULL;

    UniformCtx seed = { v, nonuniform, n, 0, NULL, NULL, NULL };
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!s || s->kind != WGSL_SYM_FUNCTION || !s->ast) continue;
        if (!(s->flags & (WGSL_SYM_FLAG_VERTEX |
                          WGSL_SYM_FLAG_FRAGMENT |
                          WGSL_SYM_FLAG_COMPUTE)))
            continue;
        uint32_t P  = wgsl_fn_param_count(s->ast);
        for (uint32_t a = 0; a < P; a++) {
            WGSLSymbol *param = uniform_fn_param_symbol(v, s, a);
            if (!param || uniform_param_builtin_is_uniform(&seed, param))
                continue;
            int idx = wgsl_val_sym_index(v, param);
            if (idx >= 0 && (size_t)idx < n) nonuniform[idx] = 1;
        }
    }

    int changed = 1;
    for (size_t iter = 0; changed && iter <= n + 4; iter++) {
        changed = 0;
        UniformCtx ucx = { v, nonuniform, n, 0, NULL, NULL, NULL };
        PtrParamCtx pcx = { &ucx, nonuniform, &changed };
        for (size_t i = 0; i < n; i++) {
            WGSLSymbol *s = v->res->all_decls[i];
            if (!s || s->kind != WGSL_SYM_FUNCTION ||
                !s->ast || s->ast->child_count == 0)
                continue;
            WGSLNode *body = s->ast->children[s->ast->child_count - 1];
            scan_pointer_param_calls(&pcx, body);
        }
    }
    return nonuniform;
}

void check_uniformity_block(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return;
    WGSLValidator *v = cx->v;

    /* §15.2.9 annotation — record per-node divergent-CF status so
     * tooling (IDE, LSP) can surface it.  Only meaningful for STMT-
     * and EXPR-kind nodes; markers like ATTRIBUTE / DECL_STRUCT_MEMBER
     * inherit it too without harm. */
    if (is_divergent) n->flags |= (uint16_t)  WGSL_FLAG_DIVERGENT_CF;
    else              n->flags &= (uint16_t) ~WGSL_FLAG_DIVERGENT_CF;

    if (n->kind == WGSL_NODE_DECL_VAR ||
        n->kind == WGSL_NODE_DECL_LET ||
        n->kind == WGSL_NODE_DECL_CONST)
    {
        WGSLSymbol *sym = wgsl_val_find_symbol_for_ast(v, n);
        WGSLNode *init = uniform_decl_init_expr(n);
        int u = init ? analyze_uniform_expr(cx, init, is_divergent, fn_reqs)
                     : !(is_divergent & UD_WORKGROUP);
        if (sym) uniform_state_set(cx, sym, u);
        return;
    }

    if (n->kind == WGSL_NODE_STMT_ASSIGN ||
        n->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN)
    {
        WGSLNode *lhs = n->child_count > 0 ? n->children[0] : NULL;
        WGSLNode *rhs = n->child_count > 1 ? n->children[1] : NULL;
        int lhs_u = lhs ? analyze_uniform_expr(cx, lhs, is_divergent, fn_reqs)
                        : !(is_divergent & UD_WORKGROUP);
        int rhs_u = rhs ? analyze_uniform_expr(cx, rhs, is_divergent, fn_reqs)
                        : !(is_divergent & UD_WORKGROUP);
        WGSLSymbol *root = uniform_assignment_root(lhs);
        if (root) {
            int prev_u = decl_init_is_uniform(cx, root);
            int full = uniform_lhs_is_full_reference(lhs);
            int new_u = rhs_u;
            if (n->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN) {
                new_u = new_u && lhs_u;
            }
            if (!full) new_u = new_u && lhs_u && prev_u;
            uniform_state_set(cx, root, new_u);
        }
        return;
    }

    if (n->kind == WGSL_NODE_STMT_INCREMENT ||
        n->kind == WGSL_NODE_STMT_DECREMENT)
    {
        WGSLNode *lhs = n->child_count > 0 ? n->children[0] : NULL;
        int lhs_u = lhs ? analyze_uniform_expr(cx, lhs, is_divergent, fn_reqs)
                        : !(is_divergent & UD_WORKGROUP);
        WGSLSymbol *root = uniform_assignment_root(lhs);
        if (root) {
            int prev_u = decl_init_is_uniform(cx, root);
            int full = uniform_lhs_is_full_reference(lhs);
            uniform_state_set(cx, root, full ? lhs_u : (lhs_u && prev_u));
        }
        return;
    }

    if (n->kind == WGSL_NODE_STMT_COMPOUND) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        UniformBehavior b = WGSL_BEHAVIOR_NEXT;
        int cur_divergent = is_divergent;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
            check_uniformity_block(cx, c, cur_divergent, fn_reqs);
            UniformBehavior sb = uniform_beh_stmt(c);
            int c_is_loop =
                c->kind == WGSL_NODE_STMT_LOOP ||
                c->kind == WGSL_NODE_STMT_FOR ||
                c->kind == WGSL_NODE_STMT_WHILE;
            if (uniform_loop_makes_following_divergent(
                    cx, c, cur_divergent, fn_reqs) ||
                (!c_is_loop && uniform_stmt_has_partial_interrupt(
                    cx, c, cur_divergent,
                    (UniformBehavior)(
                        WGSL_BEHAVIOR_BREAK |
                        WGSL_BEHAVIOR_CONTINUE),
                    fn_reqs)))
            {
                cur_divergent = uniform_normalize_divergence(
                    cur_divergent | UD_ALL);
            }
            b = (UniformBehavior)((b & ~WGSL_BEHAVIOR_NEXT) | sb);
        }
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }

    /* §15.2.6 — divergence-introducing statements. */
    if (n->kind == WGSL_NODE_STMT_IF ||
        n->kind == WGSL_NODE_STMT_ELSE_IF)
    {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        if (n->child_count < 2) {
            if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
            return;
        }
        int cond_unif = analyze_uniform_expr(cx, n->children[0],
                                             is_divergent, fn_reqs);
        int branch_div = uniform_diverge_for_condition(
            is_divergent, n->children[0], cond_unif);
        uint8_t *base_known = NULL, *base_uniform = NULL;
        uint8_t *acc_known = NULL, *acc_uniform = NULL;
        if (uniform_state_copy(cx, &base_known, &base_uniform) &&
            uniform_state_alloc_zero(cx, &acc_known, &acc_uniform))
        {
            int saw_else = 0;
            for (uint32_t i = 1; i < n->child_count; i++) {
                WGSLNode *branch = n->children[i];
                if (!branch) continue;
                if (branch->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
                uniform_state_restore(cx, base_known, base_uniform);
                check_uniformity_block(cx, branch, branch_div, fn_reqs);
                if (uniform_beh_stmt(branch) & WGSL_BEHAVIOR_NEXT) {
                    uniform_state_join_current_into(cx, acc_known, acc_uniform);
                }
            }
            if (!saw_else) {
                uniform_state_join_into(
                    cx->nsyms, acc_known, acc_uniform,
                    base_known, base_uniform);
            }
            uniform_state_restore(cx, acc_known, acc_uniform);
            free(base_known);
            free(base_uniform);
            free(acc_known);
            free(acc_uniform);
            if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
            return;
        }
        free(base_known);
        free(base_uniform);
        free(acc_known);
        free(acc_uniform);
        for (uint32_t i = 1; i < n->child_count; i++) {
            check_uniformity_block(cx, n->children[i], branch_div, fn_reqs);
        }
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_ELSE) {
        for (uint32_t i = 0; i < n->child_count; i++) {
            check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
        }
        return;
    }
    if (n->kind == WGSL_NODE_STMT_WHILE) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        WGSLNode *cond = n->child_count >= 1 ? n->children[0] : NULL;
        WGSLNode *body = n->child_count >= 2 ? n->children[1] : NULL;
        int force_nonuniform_after = body &&
            uniform_stmt_has_partial_interrupt(
                cx, body, is_divergent,
                WGSL_BEHAVIOR_CONTINUE, fn_reqs);
        uint8_t *entry_known = NULL, *entry_uniform = NULL;
        uint8_t *head_known = NULL, *head_uniform = NULL;
        uint8_t *next_known = NULL, *next_uniform = NULL;
        if (uniform_state_copy(cx, &entry_known, &entry_uniform) &&
            uniform_state_copy(cx, &head_known, &head_uniform) &&
            uniform_state_copy(cx, &next_known, &next_uniform))
        {
            for (size_t iter = 0; iter <= cx->nsyms + 4; iter++) {
                uniform_state_clear_arrays(cx, next_known, next_uniform);
                uniform_state_join_into(
                    cx->nsyms, next_known, next_uniform,
                    head_known, head_uniform);
                uniform_state_restore(cx, head_known, head_uniform);
                int cond_u = cond
                    ? analyze_uniform_expr(cx, cond, is_divergent, fn_reqs)
                    : !(is_divergent & UD_WORKGROUP);
                int cond_div = cond
                    ? uniform_diverge_for_condition(is_divergent, cond, cond_u)
                    : is_divergent;
                int loop_divergent = cond_div;
                if (body && uniform_stmt_has_partial_interrupt(
                        cx, body, cond_div,
                        (UniformBehavior)(
                            WGSL_BEHAVIOR_BREAK |
                            WGSL_BEHAVIOR_CONTINUE |
                            WGSL_BEHAVIOR_RETURN),
                        fn_reqs))
                {
                    loop_divergent = uniform_normalize_divergence(
                        loop_divergent | UD_ALL);
                }
                if (body) {
                    check_uniformity_block(cx, body, loop_divergent, fn_reqs);
                }
                uniform_state_join_current_into(cx, next_known, next_uniform);
                if (memcmp(head_known, next_known, cx->nsyms) == 0 &&
                    memcmp(head_uniform, next_uniform, cx->nsyms) == 0)
                    break;
                memcpy(head_known, next_known, cx->nsyms);
                memcpy(head_uniform, next_uniform, cx->nsyms);
            }
            uint8_t *exit_known = NULL, *exit_uniform = NULL;
            int saw_exit = 0;
            if (uniform_state_alloc_zero(cx, &exit_known, &exit_uniform)) {
                uniform_state_restore(cx, entry_known, entry_uniform);
                for (uint32_t i = 0; i < n->child_count; i++) {
                    uniform_collect_break_exits(
                        cx, n->children[i], is_divergent, fn_reqs,
                        exit_known, exit_uniform, &saw_exit);
                }
                if (cond && saw_exit) {
                    uniform_state_join_into(
                        cx->nsyms, exit_known, exit_uniform,
                        entry_known, entry_uniform);
                }
            }
            if (saw_exit) {
                uniform_state_restore(cx, exit_known, exit_uniform);
            } else {
                uniform_state_restore(cx, head_known, head_uniform);
            }
            if (force_nonuniform_after) {
                uniform_state_mark_known_nonuniform(cx);
            }
            free(exit_known);
            free(exit_uniform);
        }
        free(entry_known);
        free(entry_uniform);
        free(head_known);
        free(head_uniform);
        free(next_known);
        free(next_uniform);
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_FOR) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        WGSLNode *init = uniform_for_init(n);
        WGSLNode *cond = uniform_for_cond(n);
        WGSLNode *update = uniform_for_update(n);
        WGSLNode *body = uniform_for_body(n);
        if (init) check_uniformity_block(cx, init, is_divergent, fn_reqs);
        int force_nonuniform_after = body &&
            uniform_stmt_has_partial_interrupt(
                cx, body, is_divergent,
                WGSL_BEHAVIOR_CONTINUE, fn_reqs);

        uint8_t *entry_known = NULL, *entry_uniform = NULL;
        uint8_t *head_known = NULL, *head_uniform = NULL;
        uint8_t *next_known = NULL, *next_uniform = NULL;
        if (uniform_state_copy(cx, &entry_known, &entry_uniform) &&
            uniform_state_copy(cx, &head_known, &head_uniform) &&
            uniform_state_copy(cx, &next_known, &next_uniform))
        {
            for (size_t iter = 0; iter <= cx->nsyms + 4; iter++) {
                uniform_state_clear_arrays(cx, next_known, next_uniform);
                if (cond) {
                    uniform_state_join_into(
                        cx->nsyms, next_known, next_uniform,
                        head_known, head_uniform);
                }
                uniform_state_restore(cx, head_known, head_uniform);
                int cond_u = cond
                    ? analyze_uniform_expr(cx, cond, is_divergent, fn_reqs)
                    : !(is_divergent & UD_WORKGROUP);
                int cond_div = cond
                    ? uniform_diverge_for_condition(is_divergent, cond, cond_u)
                    : is_divergent;
                int loop_divergent = cond_div;
                if (body && uniform_stmt_has_partial_interrupt(
                        cx, body, cond_div,
                        (UniformBehavior)(
                            WGSL_BEHAVIOR_BREAK |
                            WGSL_BEHAVIOR_CONTINUE |
                            WGSL_BEHAVIOR_RETURN),
                        fn_reqs))
                {
                    loop_divergent = uniform_normalize_divergence(
                        loop_divergent | UD_ALL);
                }
                if (body) {
                    check_uniformity_block(cx, body, loop_divergent, fn_reqs);
                }
                if (update) {
                    check_uniformity_block(cx, update, loop_divergent, fn_reqs);
                }
                uniform_state_join_current_into(cx, next_known, next_uniform);
                if (memcmp(head_known, next_known, cx->nsyms) == 0 &&
                    memcmp(head_uniform, next_uniform, cx->nsyms) == 0)
                    break;
                memcpy(head_known, next_known, cx->nsyms);
                memcpy(head_uniform, next_uniform, cx->nsyms);
            }
            uint8_t *exit_known = NULL, *exit_uniform = NULL;
            int saw_exit = 0;
            if (uniform_state_alloc_zero(cx, &exit_known, &exit_uniform)) {
                uniform_state_restore(cx, entry_known, entry_uniform);
                for (uint32_t i = 0; i < n->child_count; i++) {
                    uniform_collect_break_exits(
                        cx, n->children[i], is_divergent, fn_reqs,
                        exit_known, exit_uniform, &saw_exit);
                }
                if (cond && saw_exit) {
                    uniform_state_join_into(
                        cx->nsyms, exit_known, exit_uniform,
                        entry_known, entry_uniform);
                }
            }
            if (saw_exit) {
                uniform_state_restore(cx, exit_known, exit_uniform);
            } else {
                uniform_state_restore(cx, head_known, head_uniform);
            }
            if (force_nonuniform_after) {
                uniform_state_mark_known_nonuniform(cx);
            }
            free(exit_known);
            free(exit_uniform);
        }
        free(entry_known);
        free(entry_uniform);
        free(head_known);
        free(head_uniform);
        free(next_known);
        free(next_uniform);
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_SWITCH) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        int sel_unif = (n->child_count >= 1)
            ? analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs)
            : !(is_divergent & UD_WORKGROUP);
        int sel_div = (n->child_count >= 1)
            ? uniform_diverge_for_condition(is_divergent, n->children[0], sel_unif)
            : is_divergent;
        uint8_t *base_known = NULL, *base_uniform = NULL;
        uint8_t *acc_known = NULL, *acc_uniform = NULL;
        uint8_t *exit_known = NULL, *exit_uniform = NULL;
        if (uniform_state_copy(cx, &base_known, &base_uniform) &&
            uniform_state_alloc_zero(cx, &acc_known, &acc_uniform) &&
            uniform_state_alloc_zero(cx, &exit_known, &exit_uniform))
        {
            int has_default = 0;
            int saw_exit = 0;
            for (uint32_t i = 1; i < n->child_count; i++) {
                WGSLNode *clause = n->children[i];
                if (!clause || clause->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
                if (wgsl_case_is_default_alone(clause)) has_default = 1;
                uniform_state_restore(cx, base_known, base_uniform);
                check_uniformity_block(cx, clause, sel_div, fn_reqs);
                UniformBehavior cb = uniform_beh_stmt(clause);
                if (cb & (WGSL_BEHAVIOR_NEXT | WGSL_BEHAVIOR_BREAK)) {
                    uniform_state_join_current_into(cx, acc_known, acc_uniform);
                }
                uniform_state_restore(cx, base_known, base_uniform);
                if (cb & WGSL_BEHAVIOR_BREAK) {
                    uniform_collect_break_exits(
                        cx, clause, sel_div, fn_reqs,
                        exit_known, exit_uniform, &saw_exit);
                }
            }
            if (!has_default) {
                uniform_state_join_into(
                    cx->nsyms, acc_known, acc_uniform,
                    base_known, base_uniform);
            }
            if (saw_exit) {
                uniform_state_join_into(
                    cx->nsyms, acc_known, acc_uniform,
                    exit_known, exit_uniform);
            }
            uniform_state_restore(cx, acc_known, acc_uniform);
        }
        free(base_known);
        free(base_uniform);
        free(acc_known);
        free(acc_uniform);
        free(exit_known);
        free(exit_uniform);
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_LOOP) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        uint8_t *entry_known = NULL, *entry_uniform = NULL;
        uint8_t *head_known = NULL, *head_uniform = NULL;
        uint8_t *next_known = NULL, *next_uniform = NULL;
        int loop_divergent =
            is_divergent;
        if (uniform_loop_has_partial_interrupt(cx, n, is_divergent, fn_reqs)) {
            loop_divergent = uniform_normalize_divergence(loop_divergent | UD_ALL);
        }
        if (uniform_state_copy(cx, &entry_known, &entry_uniform) &&
            uniform_state_copy(cx, &head_known, &head_uniform) &&
            uniform_state_copy(cx, &next_known, &next_uniform))
        {
            for (size_t iter = 0; iter <= cx->nsyms + 4; iter++) {
                uniform_state_clear_arrays(cx, next_known, next_uniform);
                uniform_state_restore(cx, head_known, head_uniform);
                UniformBehavior b = WGSL_BEHAVIOR_NEXT;
                for (uint32_t i = 0; i < n->child_count; i++) {
                    WGSLNode *c = n->children[i];
                    if (!c) continue;
                    if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
                    check_uniformity_block(cx, c, loop_divergent, fn_reqs);
                    UniformBehavior sb = uniform_beh_stmt(c);
                    b = (UniformBehavior)((b & ~WGSL_BEHAVIOR_NEXT) | sb);
                }
                uniform_state_join_current_into(cx, next_known, next_uniform);
                if (memcmp(head_known, next_known, cx->nsyms) == 0 &&
                    memcmp(head_uniform, next_uniform, cx->nsyms) == 0)
                    break;
                memcpy(head_known, next_known, cx->nsyms);
                memcpy(head_uniform, next_uniform, cx->nsyms);
            }
            uint8_t *exit_known = NULL, *exit_uniform = NULL;
            int saw_exit = 0;
            if (uniform_state_alloc_zero(cx, &exit_known, &exit_uniform)) {
                uniform_state_restore(cx, entry_known, entry_uniform);
                for (uint32_t i = 0; i < n->child_count; i++) {
                    uniform_collect_break_exits(
                        cx, n->children[i], loop_divergent, fn_reqs,
                        exit_known, exit_uniform, &saw_exit);
                }
            }
            if (saw_exit) {
                uniform_state_restore(cx, exit_known, exit_uniform);
            } else {
                uniform_state_restore(cx, head_known, head_uniform);
            }
            free(exit_known);
            free(exit_uniform);
        }
        free(entry_known);
        free(entry_uniform);
        free(head_known);
        free(head_uniform);
        free(next_known);
        free(next_uniform);
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }
    if (n->kind == WGSL_NODE_STMT_CONTINUING) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        for (uint32_t i = 0; i < n->child_count; i++) {
            check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
        }
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }

    if (n->kind == WGSL_NODE_STMT_CASE_CLAUSE) {
        int diag_scope = push_node_diagnostic_attr_scope(cx->v, n);
        uint32_t sel_count = wgsl_case_selector_count(n);
        for (uint32_t i = 0; i < sel_count && i < n->child_count; i++) {
            analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs);
        }
        for (uint32_t i = sel_count; i < n->child_count; i++) {
            check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
        }
        if (diag_scope) wgsl_diag_pop_scope(cx->v->diag);
        return;
    }

    if (n->kind == WGSL_NODE_STMT_BREAK_IF) {
        if (n->child_count > 0) {
            analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs);
        }
        return;
    }

    if (n->kind == WGSL_NODE_STMT_FN_CALL)
    {
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (uniform_expr_node(n->children[i])) {
                analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs);
                if (n->children[i] &&
                    n->children[i]->kind == WGSL_NODE_EXPR_CALL)
                {
                    apply_user_call_pointer_effects(
                        cx, n->children[i], is_divergent, fn_reqs);
                }
            } else {
                check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
            }
        }
        return;
    }

    if (n->kind == WGSL_NODE_STMT_RETURN ||
        n->kind == WGSL_NODE_STMT_PHONY_ASSIGN)
    {
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (uniform_expr_node(n->children[i])) {
                analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs);
            } else {
                check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
            }
        }
        return;
    }

    if (n->kind == WGSL_NODE_EXPR_CALL) {
        analyze_uniform_expr(cx, n, is_divergent, fn_reqs);
        return;
    }

    /* Default: walk children with same divergence. */
    for (uint32_t i = 0; i < n->child_count; i++) {
        if (uniform_expr_node(n->children[i])) {
            analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs);
        } else {
            check_uniformity_block(cx, n->children[i], is_divergent, fn_reqs);
        }
    }
}

void validate_uniformity(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    /* §15.2.3 / §15.2.7 — pre-compute per-fn uniformity-CF requirement
     * bits (which fns transitively call something needing uniform CF).
     * Used by `check_uniformity_block` to flag user-fn calls in
     * divergent CF when the callee carries those requirements. */
    uint32_t *fn_reqs = compute_fn_uniformity_reqs(v);
    uint8_t *param_nonuniform = compute_param_nonuniform(v);
    uint8_t *fn_returns_uniform =
        compute_fn_returns_uniform(v, fn_reqs, param_nonuniform);
    size_t nsyms = v->res ? v->res->all_decl_count : 0;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (n && n->kind == WGSL_NODE_DECL_FUNCTION && n->child_count > 0) {
            /* §4.2: push a diagnostic scope for per-function
             * @diagnostic attributes.  Walk the fn's attribute range
             * for @diagnostic attrs and register their filters. */
            wgsl_diag_push_scope(v->diag);
            uint32_t FA = wgsl_fn_attr_count(n);
            set_diagnostic_attr_filters(v, n->children, FA);

            uint8_t *sym_uniform = nsyms ? (uint8_t *)calloc(nsyms, 1) : NULL;
            uint8_t *sym_known = nsyms ? (uint8_t *)calloc(nsyms, 1) : NULL;
            UniformCtx ucx = {
                v, param_nonuniform, nsyms, 0, sym_uniform, sym_known,
                fn_returns_uniform
            };
            check_uniformity_block(&ucx, n->children[n->child_count - 1], 0, fn_reqs);
            free(sym_uniform);
            free(sym_known);
            wgsl_diag_pop_scope(v->diag);
        }
    }
    free(fn_returns_uniform);
    free(param_nonuniform);
    free(fn_reqs);
}
