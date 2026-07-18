/**
 * @file state.c — uniformity state env + root/address helpers
 */
#include "internal/uniformity_priv.h"

int uniform_normalize_divergence(int divergence) {
    if (divergence & UD_SUBGROUP) divergence |= UD_WORKGROUP;
    return divergence;
}

int uniform_expr_is_subgroup_uniform(const WGSLNode *n) {
    return n && ((n->flags & WGSL_FLAG_SUBGROUP_UNIFORM) ||
                 !(n->flags & WGSL_FLAG_NONUNIFORM));
}

int uniform_diverge_for_condition(
    int parent_divergence, const WGSLNode *cond, int workgroup_uniform)
{
    int d = parent_divergence;
    if (!workgroup_uniform) d |= UD_WORKGROUP;
    if (!uniform_expr_is_subgroup_uniform(cond)) d |= UD_ALL;
    return uniform_normalize_divergence(d);
}

int ident_text_eq(
    WGSLValidator *v, const WGSLNode *n, const char *want, uint32_t want_len)
{
    if (!v || !v->src || !n || n->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = wgsl_node_name_span(n).offset;
    uint32_t len = wgsl_node_name_span(n).length;
    return len == want_len && memcmp(v->src->bytes + off, want, want_len) == 0;
}

int diagnostic_attr_filter(
    WGSLValidator *v, const WGSLNode *attr,
    const char **out_rule, uint32_t *out_rule_len, int *out_severity)
{
    if (out_rule) *out_rule = NULL;
    if (out_rule_len) *out_rule_len = 0;
    if (out_severity) *out_severity = -1;
    if (!v || !v->src || !attr || attr->kind != WGSL_NODE_ATTRIBUTE) return 0;
    if (attr->child_count < 3) return 0;
    if (!ident_text_eq(v, attr->children[0], "diagnostic", 10)) return 0;

    WGSLNode *sev_id = attr->children[1];
    if (!sev_id || sev_id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t soff = wgsl_node_name_span(sev_id).offset;
    uint32_t slen = wgsl_node_name_span(sev_id).length;
    const char *snm = v->src->bytes + soff;
    int sev = -1;
    if (slen == 3 && memcmp(snm, "off", 3) == 0)     sev = 0;
    if (slen == 4 && memcmp(snm, "info", 4) == 0)    sev = WGSL_DIAG_INFO;
    if (slen == 7 && memcmp(snm, "warning", 7) == 0) sev = WGSL_DIAG_WARNING;
    if (slen == 5 && memcmp(snm, "error", 5) == 0)   sev = WGSL_DIAG_ERROR;
    if (sev < 0) return 0;

    WGSLNode *rule_id = attr->children[2];
    if (!rule_id || rule_id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t roff = wgsl_node_name_span(rule_id).offset;
    uint32_t rlen = wgsl_node_name_span(rule_id).length;
    const char *rnm = v->src->bytes + roff;
    if (!((rlen == 21 && memcmp(rnm, "derivative_uniformity", 21) == 0) ||
          (rlen == 19 && memcmp(rnm, "subgroup_uniformity", 19) == 0)))
    {
        return 0;
    }

    if (out_rule) *out_rule = rnm;
    if (out_rule_len) *out_rule_len = rlen;
    if (out_severity) *out_severity = sev;
    return 1;
}

int set_diagnostic_attr_filters(
    WGSLValidator *v, WGSLNode *const *attrs, uint32_t attr_count)
{
    if (!v || !attrs || attr_count == 0) return 0;
    int applied = 0;
    for (uint32_t i = 0; i < attr_count; i++) {
        const char *rule = NULL;
        uint32_t rule_len = 0;
        int sev = -1;
        if (!diagnostic_attr_filter(v, attrs[i], &rule, &rule_len, &sev)) continue;
        wgsl_diag_set_filter(v->diag, rule, rule_len, sev);
        applied = 1;
    }
    return applied;
}

int push_diagnostic_attr_scope(
    WGSLValidator *v, WGSLNode *const *attrs, uint32_t attr_count)
{
    if (!v || !attrs || attr_count == 0) return 0;
    wgsl_diag_push_scope(v->diag);
    if (!set_diagnostic_attr_filters(v, attrs, attr_count)) {
        wgsl_diag_pop_scope(v->diag);
        return 0;
    }
    return 1;
}

int push_node_diagnostic_attr_scope(WGSLValidator *v, const WGSLNode *n) {
    if (!n || !(n->flags & WGSL_FLAG_HAS_ATTRS)) return 0;
    uint32_t attr_count = wgsl_stmt_attr_count(n);
    WGSLNode **attrs = wgsl_stmt_attrs(n);
    return push_diagnostic_attr_scope(v, attrs, attr_count);
}

int function_filter_severity(
    WGSLValidator *v, const WGSLNode *fn, const char *rule, uint32_t rule_len)
{
    if (!v || !v->diag || !rule) return -1;
    int sev = -1;
    int global_filters = v->diag->scope_depth > 0
        ? v->diag->scope_stack[0]
        : v->diag->filter_count;
    for (int i = 0; i < global_filters; i++) {
        if (v->diag->filters[i].rule &&
            strlen(v->diag->filters[i].rule) == rule_len &&
            memcmp(v->diag->filters[i].rule, rule, rule_len) == 0)
        {
            sev = (int)v->diag->filters[i].severity;
        }
    }
    if (fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
        uint32_t attr_count = wgsl_fn_attr_count(fn);
        for (uint32_t i = 0; i < attr_count && i < fn->child_count; i++) {
            const char *attr_rule = NULL;
            uint32_t attr_rule_len = 0;
            int attr_sev = -1;
            if (diagnostic_attr_filter(
                    v, fn->children[i], &attr_rule, &attr_rule_len, &attr_sev) &&
                attr_rule_len == rule_len &&
                memcmp(attr_rule, rule, rule_len) == 0)
            {
                sev = attr_sev;
            }
        }
    }
    return sev;
}

int uniform_state_get(UniformCtx *cx, WGSLSymbol *sym, int *out) {
    if (!cx || !sym || !cx->sym_known || !cx->sym_uniform || !out) return 0;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return 0;
    if (!cx->sym_known[idx]) return 0;
    *out = cx->sym_uniform[idx] != 0;
    return 1;
}

void uniform_state_set(UniformCtx *cx, WGSLSymbol *sym, int uniform) {
    if (!cx || !sym || !cx->sym_known || !cx->sym_uniform) return;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return;
    cx->sym_known[idx] = 1;
    cx->sym_uniform[idx] = uniform ? 1u : 0u;
}

int uniform_state_copy(UniformCtx *cx, uint8_t **known, uint8_t **uniform) {
    if (!cx || !known || !uniform || cx->nsyms == 0) return 0;
    *known = (uint8_t *)malloc(cx->nsyms);
    *uniform = (uint8_t *)malloc(cx->nsyms);
    if (!*known || !*uniform) {
        free(*known);
        free(*uniform);
        *known = NULL;
        *uniform = NULL;
        return 0;
    }
    memcpy(*known, cx->sym_known, cx->nsyms);
    memcpy(*uniform, cx->sym_uniform, cx->nsyms);
    return 1;
}

void uniform_state_restore(UniformCtx *cx, const uint8_t *known, const uint8_t *uniform) {
    if (!cx || !known || !uniform || cx->nsyms == 0) return;
    memcpy(cx->sym_known, known, cx->nsyms);
    memcpy(cx->sym_uniform, uniform, cx->nsyms);
}

void uniform_state_join_into(
    size_t n, uint8_t *acc_known, uint8_t *acc_uniform,
    const uint8_t *known, const uint8_t *uniform)
{
    for (size_t i = 0; i < n; i++) {
        if (!known[i]) continue;
        if (!acc_known[i]) {
            acc_known[i] = 1;
            acc_uniform[i] = uniform[i];
        } else if (!uniform[i]) {
            acc_uniform[i] = 0;
        }
    }
}

void uniform_state_join_current_into(
    UniformCtx *cx, uint8_t *acc_known, uint8_t *acc_uniform)
{
    if (!cx || !acc_known || !acc_uniform ||
        !cx->sym_known || !cx->sym_uniform)
        return;
    uniform_state_join_into(
        cx->nsyms, acc_known, acc_uniform,
        cx->sym_known, cx->sym_uniform);
}

void uniform_state_clear_arrays(
    UniformCtx *cx, uint8_t *known, uint8_t *uniform)
{
    if (!cx || !known || !uniform || cx->nsyms == 0) return;
    memset(known, 0, cx->nsyms);
    memset(uniform, 0, cx->nsyms);
}

void uniform_state_mark_known_nonuniform(UniformCtx *cx) {
    if (!cx || !cx->sym_known || !cx->sym_uniform) return;
    for (size_t i = 0; i < cx->nsyms; i++) {
        if (cx->sym_known[i]) cx->sym_uniform[i] = 0;
    }
}

int uniform_state_alloc_zero(
    UniformCtx *cx, uint8_t **known, uint8_t **uniform)
{
    if (!cx || !known || !uniform || cx->nsyms == 0) return 0;
    *known = (uint8_t *)calloc(cx->nsyms, 1);
    *uniform = (uint8_t *)calloc(cx->nsyms, 1);
    if (!*known || !*uniform) {
        free(*known);
        free(*uniform);
        *known = NULL;
        *uniform = NULL;
        return 0;
    }
    return 1;
}

WGSLNode *uniform_decl_init_expr(const WGSLNode *d) {
    if (!d) return NULL;
    if (d->kind == WGSL_NODE_DECL_VAR) {
        uint32_t A   = wgsl_varlike_attr_count(d);
        uint32_t TPL = wgsl_varlike_tpl_count(d);
        int has_type = wgsl_varlike_has_type(d);
        int has_init = wgsl_varlike_has_init(d);
        if (!has_init) return NULL;
        uint32_t init_idx = A + TPL + (has_type ? 1u : 0u);
        return init_idx < d->child_count ? d->children[init_idx] : NULL;
    }
    if (d->kind == WGSL_NODE_DECL_LET ||
        d->kind == WGSL_NODE_DECL_CONST)
    {
        int has_type = wgsl_letlike_has_type(d);
        uint32_t init_idx = has_type ? 1u : 0u;
        return init_idx < d->child_count ? d->children[init_idx] : NULL;
    }
    return NULL;
}

int decl_init_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    if (!sym || !sym->ast) return 1;
    int known = 0;
    if (uniform_state_get(cx, sym, &known)) return known;
    if (cx->init_depth >= 16) return 1;

    WGSLNode *init = uniform_decl_init_expr(sym->ast);
    if (!init) return 1;

    cx->init_depth += 1;
    int r = is_uniform_expr(cx, init);
    cx->init_depth -= 1;
    uniform_state_set(cx, sym, r);
    return r;
}

int uniform_expr_quiet(
    UniformCtx *cx, WGSLNode *expr, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !cx->v || !cx->v->diag || !cx->v->cev)
        return analyze_uniform_expr(cx, expr, is_divergent, fn_reqs);
    size_t saved_count = cx->v->diag->count;
    int saved_errs = cx->v->diag->error_count;
    int saved_cev = cx->v->cev->store.had_error;
    int r = analyze_uniform_expr(cx, expr, is_divergent, fn_reqs);
    cx->v->diag->count = saved_count;
    cx->v->diag->error_count = saved_errs;
    cx->v->cev->store.had_error = saved_cev;
    return r;
}

WGSLSymbol *uniform_assignment_root(WGSLNode *n) {
    if (!n) return NULL;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->kind == WGSL_SYM_VAR &&
            (s->as == WGSL_AS_FUNCTION || s->as == WGSL_AS_NONE))
            return s;
        return NULL;
    }
    case WGSL_NODE_EXPR_MEMBER:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count > 0 ? uniform_assignment_root(n->children[0]) : NULL;
    case WGSL_NODE_EXPR_INDIRECTION:
        return n->child_count > 0 ? uniform_memory_root(n->children[0], 0) : NULL;
    case WGSL_NODE_EXPR_ADDR_OF:
        return n->child_count > 0 ? uniform_assignment_root(n->children[0]) : NULL;
    default:
        return NULL;
    }
}

WGSLSymbol *uniform_memory_root(WGSLNode *n, int depth) {
    if (!n || depth > 32) return NULL;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (!s) return NULL;
        if (s->kind == WGSL_SYM_LET) {
            WGSLNode *init = uniform_decl_init_expr(s->ast);
            if (init) return uniform_memory_root(init, depth + 1);
        }
        if (s->kind == WGSL_SYM_VAR || s->kind == WGSL_SYM_PARAM)
            return s;
        return NULL;
    }
    case WGSL_NODE_EXPR_ADDR_OF:
    case WGSL_NODE_EXPR_INDIRECTION:
    case WGSL_NODE_EXPR_MEMBER:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count > 0
            ? uniform_memory_root(n->children[0], depth + 1)
            : NULL;
    default:
        return NULL;
    }
}

int uniform_root_contents_are_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_VAR) {
        if (uniform_module_var_is_readonly(sym)) return 1;
        if (sym->as == WGSL_AS_FUNCTION || sym->as == WGSL_AS_NONE)
            return decl_init_is_uniform(cx, sym);
        return 0;
    }
    if (sym->kind == WGSL_SYM_PARAM) {
        return uniform_param_builtin_is_uniform(cx, sym) ||
               parameter_is_uniform(cx, sym);
    }
    if (sym->kind == WGSL_SYM_LET) {
        return decl_init_is_uniform(cx, sym);
    }
    return 0;
}

int uniform_reference_address_is_uniform(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);

int uniform_pointer_address_is_uniform(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return 0;
    is_divergent = uniform_normalize_divergence(is_divergent);
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->kind == WGSL_SYM_LET) {
            WGSLNode *init = uniform_decl_init_expr(s->ast);
            if (init) {
                return uniform_pointer_address_is_uniform(
                    cx, init, is_divergent, fn_reqs);
            }
        }
        if (s && s->kind == WGSL_SYM_PARAM) {
            return !(is_divergent & UD_WORKGROUP);
        }
        return !(is_divergent & UD_WORKGROUP);
    }
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count > 0 &&
               uniform_pointer_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_ADDR_OF:
        return n->child_count > 0 &&
               uniform_reference_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_INDIRECTION:
        return n->child_count > 0 &&
               uniform_pointer_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count > 0 &&
               uniform_pointer_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_INDEX: {
        int base_u = n->child_count > 0 &&
            uniform_pointer_address_is_uniform(
                cx, n->children[0], is_divergent, fn_reqs);
        int idx_u = n->child_count > 1 &&
            analyze_uniform_expr(cx, n->children[1], is_divergent, fn_reqs);
        return base_u && idx_u && !(is_divergent & UD_WORKGROUP);
    }
    default:
        return analyze_uniform_expr(cx, n, is_divergent, fn_reqs);
    }
}

int uniform_reference_address_is_uniform(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT:
        return !(uniform_normalize_divergence(is_divergent) & UD_WORKGROUP);
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count > 0 &&
               uniform_reference_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count > 0 &&
               uniform_reference_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_INDEX: {
        int base_u = n->child_count > 0 &&
            uniform_reference_address_is_uniform(
                cx, n->children[0], is_divergent, fn_reqs);
        int idx_u = n->child_count > 1 &&
            analyze_uniform_expr(cx, n->children[1], is_divergent, fn_reqs);
        return base_u && idx_u &&
               !(uniform_normalize_divergence(is_divergent) & UD_WORKGROUP);
    }
    case WGSL_NODE_EXPR_INDIRECTION:
        return n->child_count > 0 &&
               uniform_pointer_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_ADDR_OF:
        return n->child_count > 0 &&
               uniform_reference_address_is_uniform(
                   cx, n->children[0], is_divergent, fn_reqs);
    default:
        return analyze_uniform_expr(cx, n, is_divergent, fn_reqs);
    }
}

int uniform_lhs_is_full_reference(WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        return s && s->kind == WGSL_SYM_VAR &&
               (s->as == WGSL_AS_FUNCTION || s->as == WGSL_AS_NONE);
    }
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_ADDR_OF:
        return n->child_count > 0 && uniform_lhs_is_full_reference(n->children[0]);
    case WGSL_NODE_EXPR_INDIRECTION:
        return n->child_count > 0 &&
               uniform_lhs_is_full_reference(n->children[0]);
    default:
        return 0;
    }
}

int uniform_expr_node(const WGSLNode *n) {
    return n && n->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
           n->kind <= WGSL_NODE_EXPR_INDIRECTION;
}

WGSLNode *uniform_for_init(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = wgsl_for_flags(n);
    uint32_t idx = 0;
    if (!(flags & 1u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

WGSLNode *uniform_for_cond(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = wgsl_for_flags(n);
    uint32_t idx = 0;
    if (flags & 1u) idx += 1;
    if (!(flags & 2u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

WGSLNode *uniform_for_update(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = wgsl_for_flags(n);
    uint32_t idx = 0;
    if (flags & 1u) idx += 1;
    if (flags & 2u) idx += 1;
    if (!(flags & 4u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

WGSLNode *uniform_for_body(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR || n->child_count == 0) return NULL;
    WGSLNode *body = n->children[n->child_count - 1];
    return body && body->kind == WGSL_NODE_STMT_COMPOUND ? body : NULL;
}

/* UniformBehavior: uniformity_priv.h */
UniformBehavior uniform_beh_stmt(WGSLNode *n);

UniformBehavior uniform_beh_compound(WGSLNode *n) {
    UniformBehavior b = WGSL_BEHAVIOR_NEXT;
    if (!n) return b;
    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *c = n->children[i];
        if (!c) continue;
        if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
        UniformBehavior s = uniform_beh_stmt(c);
        b = (UniformBehavior)((b & ~WGSL_BEHAVIOR_NEXT) | s);
    }
    return b;
}

