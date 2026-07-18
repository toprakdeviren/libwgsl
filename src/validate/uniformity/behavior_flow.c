/**
 * @file behavior_flow.c — statement behavior + loop interrupt analysis
 */
#include "internal/uniformity_priv.h"

UniformBehavior uniform_beh_stmt(WGSLNode *n) {
    if (!n) return WGSL_BEHAVIOR_NEXT;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_RETURN:   return WGSL_BEHAVIOR_RETURN;
    case WGSL_NODE_STMT_BREAK:    return WGSL_BEHAVIOR_BREAK;
    case WGSL_NODE_STMT_CONTINUE: return WGSL_BEHAVIOR_CONTINUE;
    case WGSL_NODE_STMT_DISCARD:  return WGSL_BEHAVIOR_NEXT;
    case WGSL_NODE_STMT_COMPOUND: return uniform_beh_compound(n);

    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF: {
        UniformBehavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            if (c->kind == WGSL_NODE_STMT_COMPOUND ||
                c->kind == WGSL_NODE_STMT_ELSE_IF ||
                c->kind == WGSL_NODE_STMT_ELSE)
            {
                result = (UniformBehavior)(result | uniform_beh_stmt(c));
            }
        }
        if (!saw_else) result = (UniformBehavior)(result | WGSL_BEHAVIOR_NEXT);
        return result ? result : WGSL_BEHAVIOR_NEXT;
    }
    case WGSL_NODE_STMT_ELSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return uniform_beh_stmt(c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }
    case WGSL_NODE_STMT_LOOP: {
        UniformBehavior body = WGSL_BEHAVIOR_NEXT;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if ((body & WGSL_BEHAVIOR_NEXT) == 0) break;
            UniformBehavior s = uniform_beh_stmt(c);
            body = (UniformBehavior)((body & ~WGSL_BEHAVIOR_NEXT) | s);
        }
        UniformBehavior r = 0;
        if (body & WGSL_BEHAVIOR_BREAK)  r |= WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        return r ? r : WGSL_BEHAVIOR_NEXT;
    }
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE: {
        UniformBehavior body = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                body = (UniformBehavior)(body | uniform_beh_stmt(c));
            }
        }
        UniformBehavior r = WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        return r;
    }
    case WGSL_NODE_STMT_SWITCH: {
        UniformBehavior result = 0;
        int has_default = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c || c->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
            if (wgsl_case_is_default_alone(c)) has_default = 1;
            result = (UniformBehavior)(result | uniform_beh_stmt(c));
        }
        if (!has_default) result = (UniformBehavior)(result | WGSL_BEHAVIOR_NEXT);
        if (result & WGSL_BEHAVIOR_BREAK) {
            result = (UniformBehavior)((result & ~WGSL_BEHAVIOR_BREAK) |
                                       WGSL_BEHAVIOR_NEXT);
        }
        return result ? result : WGSL_BEHAVIOR_NEXT;
    }
    case WGSL_NODE_STMT_CASE_CLAUSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return uniform_beh_stmt(c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }
    case WGSL_NODE_STMT_BREAK_IF: return WGSL_BEHAVIOR_BREAK | WGSL_BEHAVIOR_NEXT;
    case WGSL_NODE_STMT_CONTINUING:
        return uniform_beh_compound(n);
    default:
        return WGSL_BEHAVIOR_NEXT;
    }
}

int uniform_stmt_has_partial_interrupt(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    UniformBehavior mask, const uint32_t *fn_reqs)
{
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF: {
        if (n->child_count < 2) return 0;
        int cond_u = uniform_expr_quiet(
            cx, n->children[0], is_divergent, fn_reqs);
        int branch_div = uniform_diverge_for_condition(
            is_divergent, n->children[0], cond_u);
        UniformBehavior b = uniform_beh_stmt(n);
        if (!cond_u && (b & mask) && (b & WGSL_BEHAVIOR_NEXT)) {
            return 1;
        }
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (uniform_stmt_has_partial_interrupt(
                    cx, c, branch_div, mask, fn_reqs))
                return 1;
        }
        return 0;
    }
    case WGSL_NODE_STMT_SWITCH: {
        if (n->child_count < 1) return 0;
        int sel_u = uniform_expr_quiet(
            cx, n->children[0], is_divergent, fn_reqs);
        int sel_div = uniform_diverge_for_condition(
            is_divergent, n->children[0], sel_u);
        UniformBehavior b = uniform_beh_stmt(n);
        if (!sel_u && (b & mask) && (b & WGSL_BEHAVIOR_NEXT)) {
            return 1;
        }
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (uniform_stmt_has_partial_interrupt(
                    cx, c, sel_div, mask, fn_reqs))
                return 1;
        }
        return 0;
    }
    case WGSL_NODE_STMT_BREAK_IF: {
        if (!(mask & WGSL_BEHAVIOR_BREAK)) return 0;
        if (n->child_count == 0) return 0;
        return !uniform_expr_quiet(cx, n->children[0],
                                   is_divergent, fn_reqs);
    }
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE:
        return 0;
    default:
        break;
    }

    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *c = n->children[i];
        if (!c || uniform_expr_node(c)) continue;
        if (uniform_stmt_has_partial_interrupt(
                cx, c, is_divergent, mask, fn_reqs))
            return 1;
    }
    return 0;
}

int uniform_loop_has_partial_interrupt(
    UniformCtx *cx, WGSLNode *loop, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!loop) return 0;
    UniformBehavior mask = (UniformBehavior)(
        WGSL_BEHAVIOR_BREAK | WGSL_BEHAVIOR_CONTINUE | WGSL_BEHAVIOR_RETURN);
    for (uint32_t i = 0; i < loop->child_count; i++) {
        WGSLNode *c = loop->children[i];
        if (!c || uniform_expr_node(c)) continue;
        if (uniform_stmt_has_partial_interrupt(
                cx, c, is_divergent, mask, fn_reqs))
            return 1;
    }
    return 0;
}

int uniform_loop_makes_following_divergent(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return 0;
    if (n->kind != WGSL_NODE_STMT_FOR &&
        n->kind != WGSL_NODE_STMT_WHILE)
        return 0;

    WGSLNode *cond = NULL;
    WGSLNode *body = NULL;
    if (n->kind == WGSL_NODE_STMT_FOR) {
        cond = uniform_for_cond(n);
        body = uniform_for_body(n);
    } else {
        cond = n->child_count >= 1 ? n->children[0] : NULL;
        body = n->child_count >= 2 ? n->children[1] : NULL;
    }
    if (!body) return 0;

    int cond_u = cond
        ? uniform_expr_quiet(cx, cond, is_divergent, fn_reqs)
        : !(is_divergent & UD_WORKGROUP);
    int body_div = cond
        ? uniform_diverge_for_condition(is_divergent, cond, cond_u)
        : is_divergent;
    UniformBehavior body_b = uniform_beh_stmt(body);
    if (!cond_u && (body_b & WGSL_BEHAVIOR_RETURN)) return 1;
    return uniform_stmt_has_partial_interrupt(
        cx, body, body_div,
        WGSL_BEHAVIOR_RETURN, fn_reqs);
}

UniformBehavior uniform_collect_break_exits(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs,
    uint8_t *exit_known, uint8_t *exit_uniform, int *saw_exit)
{
    if (!n) return WGSL_BEHAVIOR_NEXT;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_BREAK:
        if (saw_exit) *saw_exit = 1;
        uniform_state_join_current_into(cx, exit_known, exit_uniform);
        return WGSL_BEHAVIOR_BREAK;

    case WGSL_NODE_STMT_COMPOUND:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_CASE_CLAUSE: {
        UniformBehavior b = WGSL_BEHAVIOR_NEXT;
        uint32_t start = 0;
        if (n->kind == WGSL_NODE_STMT_CASE_CLAUSE) {
            start = wgsl_node_name_span(n).offset;
        }
        for (uint32_t i = start; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
            UniformBehavior cb = uniform_collect_break_exits(
                cx, c, is_divergent, fn_reqs,
                exit_known, exit_uniform, saw_exit);
            b = (UniformBehavior)((b & ~WGSL_BEHAVIOR_NEXT) | cb);
        }
        return b;
    }

    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF: {
        if (n->child_count < 2) return WGSL_BEHAVIOR_NEXT;
        int cond_u = analyze_uniform_expr(cx, n->children[0],
                                          is_divergent, fn_reqs);
        int branch_div = uniform_diverge_for_condition(
            is_divergent, n->children[0], cond_u);
        uint8_t *base_known = NULL, *base_uniform = NULL;
        uint8_t *next_known = NULL, *next_uniform = NULL;
        if (!uniform_state_copy(cx, &base_known, &base_uniform) ||
            !uniform_state_alloc_zero(cx, &next_known, &next_uniform))
        {
            free(base_known);
            free(base_uniform);
            free(next_known);
            free(next_uniform);
            check_uniformity_block(cx, n, is_divergent, fn_reqs);
            return uniform_beh_stmt(n);
        }
        UniformBehavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *branch = n->children[i];
            if (!branch) continue;
            if (branch->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            uniform_state_restore(cx, base_known, base_uniform);
            UniformBehavior bb = uniform_collect_break_exits(
                cx, branch, branch_div, fn_reqs,
                exit_known, exit_uniform, saw_exit);
            result = (UniformBehavior)(result | bb);
            if (bb & WGSL_BEHAVIOR_NEXT) {
                uniform_state_join_current_into(cx, next_known, next_uniform);
            }
        }
        if (!saw_else) {
            uniform_state_join_into(
                cx->nsyms, next_known, next_uniform,
                base_known, base_uniform);
            result = (UniformBehavior)(result | WGSL_BEHAVIOR_NEXT);
        }
        if (result & WGSL_BEHAVIOR_NEXT) {
            uniform_state_restore(cx, next_known, next_uniform);
        } else {
            uniform_state_restore(cx, base_known, base_uniform);
        }
        free(base_known);
        free(base_uniform);
        free(next_known);
        free(next_uniform);
        return result ? result : WGSL_BEHAVIOR_NEXT;
    }

    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_SWITCH:
        check_uniformity_block(cx, n, is_divergent, fn_reqs);
        return WGSL_BEHAVIOR_NEXT;

    default:
        check_uniformity_block(cx, n, is_divergent, fn_reqs);
        return uniform_beh_stmt(n);
    }
}

int uniform_is_param(const WGSLSymbol *sym) {
    return sym && sym->kind == WGSL_SYM_PARAM;
}

int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    if (!uniform_is_param(sym)) return 0;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms || !cx->param_nonuniform) {
        return 0;
    }
    return cx->param_nonuniform[idx] == 0;
}

WGSLSymbol *uniform_fn_param_symbol(
    WGSLValidator *v, const WGSLSymbol *fn, uint32_t param_idx)
{
    if (!fn || !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION) return NULL;
    WGSLNode *decl = fn->ast;
    uint32_t FA = wgsl_fn_attr_count(decl);
    uint32_t P  = wgsl_fn_param_count(decl);
    if (param_idx >= P || FA + param_idx >= decl->child_count) return NULL;
    WGSLNode *param = decl->children[FA + param_idx];
    if (!param || param->kind != WGSL_NODE_DECL_PARAM) return NULL;
    return wgsl_val_find_symbol_for_ast(v, param);
}

