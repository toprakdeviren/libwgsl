/**
 * @file uniformity.c — §15.2 uniformity analysis.
 *
 * Conservative uniformity oracle:
 *   - var<uniform>                     → uniform
 *   - read-only resource roots         → uniform
 *   - fn-scope `var` / `let`           → tracked through init,
 *                                          assignments, branches, and
 *                                          loop / switch joins
 *   - non-read-only module vars        → non-uniform
 *   - fn parameters                    → non-uniform unless they are
 *                                          spec-listed uniform builtins
 *   - literals / const / override      → uniform
 *
 * Combined with per-fn `UR_DERIVATIVE` / `UR_SUBGROUP` requirement
 * bitsets propagated through the call graph, this is enough to drive
 * the §15.6.1 barrier / §15.6.2 derivative / §15.6.3-4 subgroup-and-quad
 * "must be called in uniform control flow" diagnostics.
 *
 * Co-extensive with the validator-facing WGSL §15.2 surface: pointer
 * desugaring, assignment lattice, expression lattice, and call-site
 * requirements.  §15.2.9 full-AST annotation remains a tooling breadth
 * improvement rather than an accept/reject gap.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/token.h"

/* Conservative uniformity oracle.  It tracks the leaves and simple
 * per-symbol value states the spec commonly cares about:
 *   - var<uniform>  reads      → uniform
 *   - non-read-only module var reads → non-uniform
 *   - ordinary function parameters   → non-uniform (potentially
 *                                      per-invocation at entry-point
 *                                      boundary)
 *   - const / override / let bound to const-exprs → uniform
 *   - literals                 → uniform
 *
 * This is enough to flag `if (frag_input > 0) { dpdx(...); }` etc.
 * It remains intentionally conservative where the spec allows "may
 * trigger" diagnostics. */
/* §15.2.5 helper — pulls a fn-scope `var` / `let` initializer expression
 * and runs the uniformity oracle on it.  Without this, every function-
 * scope `var` would be reported as non-uniform — false-flagging the
 * canonical reduction pattern
 *     `for (var s = 128u; s > 0u; s >>= 1u) { … workgroupBarrier(); }`
 * where `s` is a uniform loop counter.
 *
 * Post-init assignments are tracked through sequential statements,
 * partial/full assignment distinction, branches, and loop/switch joins. */
typedef struct {
    WGSLValidator *v;
    const uint8_t *param_nonuniform;
    size_t         nsyms;
    int            init_depth;
    uint8_t       *sym_uniform;
    uint8_t       *sym_known;
} UniformCtx;

static int is_uniform_expr(UniformCtx *cx, WGSLNode *n);
static int analyze_uniform_expr(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
static WGSLSymbol *call_user_fn_sym(WGSLNode *call);
static void check_uniformity_block(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);
static WGSLSymbol *uniform_memory_root(WGSLNode *n, int depth);
static int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym);
static int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym);
static int uniform_module_var_is_readonly(const WGSLSymbol *sym);
static int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym);

enum {
    UD_NONE      = 0,
    UD_WORKGROUP = 1u << 0,
    UD_SUBGROUP  = 1u << 1,
    UD_ALL       = UD_WORKGROUP | UD_SUBGROUP,
};

static int uniform_normalize_divergence(int divergence) {
    if (divergence & UD_SUBGROUP) divergence |= UD_WORKGROUP;
    return divergence;
}

static int uniform_expr_is_subgroup_uniform(const WGSLNode *n) {
    return n && ((n->flags & WGSL_FLAG_SUBGROUP_UNIFORM) ||
                 !(n->flags & WGSL_FLAG_NONUNIFORM));
}

static int uniform_diverge_for_condition(
    int parent_divergence, const WGSLNode *cond, int workgroup_uniform)
{
    int d = parent_divergence;
    if (!workgroup_uniform) d |= UD_WORKGROUP;
    if (!uniform_expr_is_subgroup_uniform(cond)) d |= UD_ALL;
    return uniform_normalize_divergence(d);
}

static int ident_text_eq(
    WGSLValidator *v, const WGSLNode *n, const char *want, uint32_t want_len)
{
    if (!v || !v->src || !n || n->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(n->payload[0] >> 32);
    return len == want_len && memcmp(v->src->bytes + off, want, want_len) == 0;
}

static int diagnostic_attr_filter(
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
    uint32_t soff = (uint32_t)(sev_id->payload[0] & 0xFFFFFFFFu);
    uint32_t slen = (uint32_t)(sev_id->payload[0] >> 32);
    const char *snm = v->src->bytes + soff;
    int sev = -1;
    if (slen == 3 && memcmp(snm, "off", 3) == 0)     sev = 0;
    if (slen == 4 && memcmp(snm, "info", 4) == 0)    sev = WGSL_DIAG_INFO;
    if (slen == 7 && memcmp(snm, "warning", 7) == 0) sev = WGSL_DIAG_WARNING;
    if (slen == 5 && memcmp(snm, "error", 5) == 0)   sev = WGSL_DIAG_ERROR;
    if (sev < 0) return 0;

    WGSLNode *rule_id = attr->children[2];
    if (!rule_id || rule_id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t roff = (uint32_t)(rule_id->payload[0] & 0xFFFFFFFFu);
    uint32_t rlen = (uint32_t)(rule_id->payload[0] >> 32);
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

static int set_diagnostic_attr_filters(
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

static int push_diagnostic_attr_scope(
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

static int push_node_diagnostic_attr_scope(WGSLValidator *v, const WGSLNode *n) {
    if (!n || !(n->flags & WGSL_FLAG_HAS_ATTRS)) return 0;
    uint32_t attr_count = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
    WGSLNode **attrs = (WGSLNode **)(uintptr_t)n->payload[2];
    return push_diagnostic_attr_scope(v, attrs, attr_count);
}

static int function_filter_severity(
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
        uint32_t attr_count = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
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

static int uniform_state_get(UniformCtx *cx, WGSLSymbol *sym, int *out) {
    if (!cx || !sym || !cx->sym_known || !cx->sym_uniform || !out) return 0;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return 0;
    if (!cx->sym_known[idx]) return 0;
    *out = cx->sym_uniform[idx] != 0;
    return 1;
}

static void uniform_state_set(UniformCtx *cx, WGSLSymbol *sym, int uniform) {
    if (!cx || !sym || !cx->sym_known || !cx->sym_uniform) return;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return;
    cx->sym_known[idx] = 1;
    cx->sym_uniform[idx] = uniform ? 1u : 0u;
}

static int uniform_state_copy(UniformCtx *cx, uint8_t **known, uint8_t **uniform) {
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

static void uniform_state_restore(UniformCtx *cx, const uint8_t *known, const uint8_t *uniform) {
    if (!cx || !known || !uniform || cx->nsyms == 0) return;
    memcpy(cx->sym_known, known, cx->nsyms);
    memcpy(cx->sym_uniform, uniform, cx->nsyms);
}

static void uniform_state_join_into(
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

static void uniform_state_join_current_into(
    UniformCtx *cx, uint8_t *acc_known, uint8_t *acc_uniform)
{
    if (!cx || !acc_known || !acc_uniform ||
        !cx->sym_known || !cx->sym_uniform)
        return;
    uniform_state_join_into(
        cx->nsyms, acc_known, acc_uniform,
        cx->sym_known, cx->sym_uniform);
}

static void uniform_state_clear_arrays(
    UniformCtx *cx, uint8_t *known, uint8_t *uniform)
{
    if (!cx || !known || !uniform || cx->nsyms == 0) return;
    memset(known, 0, cx->nsyms);
    memset(uniform, 0, cx->nsyms);
}

static void uniform_state_mark_known_nonuniform(UniformCtx *cx) {
    if (!cx || !cx->sym_known || !cx->sym_uniform) return;
    for (size_t i = 0; i < cx->nsyms; i++) {
        if (cx->sym_known[i]) cx->sym_uniform[i] = 0;
    }
}

static int uniform_state_alloc_zero(
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

static WGSLNode *uniform_decl_init_expr(const WGSLNode *d) {
    if (!d) return NULL;
    if (d->kind == WGSL_NODE_DECL_VAR) {
        uint32_t A   = (uint32_t)(d->payload[1] & 0xFFFFFFFFu);
        uint32_t TPL = (uint32_t)(d->payload[1] >> 32);
        int has_type = (int)(d->payload[2] & 1u);
        int has_init = (int)(d->payload[2] >> 32);
        if (!has_init) return NULL;
        uint32_t init_idx = A + TPL + (has_type ? 1u : 0u);
        return init_idx < d->child_count ? d->children[init_idx] : NULL;
    }
    if (d->kind == WGSL_NODE_DECL_LET ||
        d->kind == WGSL_NODE_DECL_CONST)
    {
        int has_type = (int)(d->payload[1] & 1u);
        uint32_t init_idx = has_type ? 1u : 0u;
        return init_idx < d->child_count ? d->children[init_idx] : NULL;
    }
    return NULL;
}

static int decl_init_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
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

static int uniform_expr_quiet(
    UniformCtx *cx, WGSLNode *expr, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !cx->v || !cx->v->diag || !cx->v->cev)
        return analyze_uniform_expr(cx, expr, is_divergent, fn_reqs);
    size_t saved_count = cx->v->diag->count;
    int saved_errs = cx->v->diag->error_count;
    int saved_cev = cx->v->cev->had_error;
    int r = analyze_uniform_expr(cx, expr, is_divergent, fn_reqs);
    cx->v->diag->count = saved_count;
    cx->v->diag->error_count = saved_errs;
    cx->v->cev->had_error = saved_cev;
    return r;
}

static WGSLSymbol *uniform_assignment_root(WGSLNode *n) {
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

static WGSLSymbol *uniform_memory_root(WGSLNode *n, int depth) {
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

static int uniform_root_contents_are_uniform(UniformCtx *cx, WGSLSymbol *sym) {
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

static int uniform_reference_address_is_uniform(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs);

static int uniform_pointer_address_is_uniform(
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

static int uniform_reference_address_is_uniform(
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

static int uniform_lhs_is_full_reference(WGSLNode *n) {
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

static int uniform_expr_node(const WGSLNode *n) {
    return n && n->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
           n->kind <= WGSL_NODE_EXPR_INDIRECTION;
}

static WGSLNode *uniform_for_init(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = (uint32_t)n->payload[0];
    uint32_t idx = 0;
    if (!(flags & 1u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

static WGSLNode *uniform_for_cond(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = (uint32_t)n->payload[0];
    uint32_t idx = 0;
    if (flags & 1u) idx += 1;
    if (!(flags & 2u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

static WGSLNode *uniform_for_update(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR) return NULL;
    uint32_t flags = (uint32_t)n->payload[0];
    uint32_t idx = 0;
    if (flags & 1u) idx += 1;
    if (flags & 2u) idx += 1;
    if (!(flags & 4u)) return NULL;
    return idx < n->child_count ? n->children[idx] : NULL;
}

static WGSLNode *uniform_for_body(WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_STMT_FOR || n->child_count == 0) return NULL;
    WGSLNode *body = n->children[n->child_count - 1];
    return body && body->kind == WGSL_NODE_STMT_COMPOUND ? body : NULL;
}

typedef uint8_t UniformBehavior;

static UniformBehavior uniform_beh_stmt(WGSLNode *n);

static UniformBehavior uniform_beh_compound(WGSLNode *n) {
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

static UniformBehavior uniform_beh_stmt(WGSLNode *n) {
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
            if ((uint32_t)((c->payload[0] >> 32) & 1u)) has_default = 1;
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

static int uniform_stmt_has_partial_interrupt(
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

static int uniform_loop_has_partial_interrupt(
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

static int uniform_loop_makes_following_divergent(
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

static UniformBehavior uniform_collect_break_exits(
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
            start = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
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

static int uniform_is_param(const WGSLSymbol *sym) {
    return sym && sym->kind == WGSL_SYM_PARAM;
}

static int parameter_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    if (!uniform_is_param(sym)) return 0;
    int idx = wgsl_val_sym_index(cx->v, sym);
    if (idx < 0 || (size_t)idx >= cx->nsyms || !cx->param_nonuniform) {
        return 0;
    }
    return cx->param_nonuniform[idx] == 0;
}

static WGSLSymbol *uniform_fn_param_symbol(
    WGSLValidator *v, const WGSLSymbol *fn, uint32_t param_idx)
{
    if (!fn || !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION) return NULL;
    WGSLNode *decl = fn->ast;
    uint32_t FA = (uint32_t)(decl->payload[1] & 0xFFFFFFFFu);
    uint32_t P  = (uint32_t)(decl->payload[1] >> 32);
    if (param_idx >= P || FA + param_idx >= decl->child_count) return NULL;
    WGSLNode *param = decl->children[FA + param_idx];
    if (!param || param->kind != WGSL_NODE_DECL_PARAM) return NULL;
    return wgsl_val_find_symbol_for_ast(v, param);
}

/* Side-effect: every node visited gets WGSL_FLAG_NONUNIFORM set / cleared
 * to record the §15.2.9 "MAY_BE_NON_UNIFORM" tag.  IDE / tooling reads
 * this via the public `wgsl_node_is_non_uniform()` query.  Children are
 * recursively annotated even when an early non-uniform leaf would
 * suffice for the bool result — so callers don't see partial trees. */
static int is_uniform_expr(UniformCtx *cx, WGSLNode *n) {
    return analyze_uniform_expr(cx, n, 0, NULL);
}

/* Names of builtin functions that the spec requires to be invoked in
 * uniform control flow (§15.6.1 barriers, §15.6.2 derivatives + texture
 * sampling with implicit derivatives).  These map to the
 * `derivative_uniformity` filter rule. */
static int call_requires_uniform_cf(const char *nm, uint32_t len) {
    static const struct { const char *n; uint32_t l; } kNames[] = {
        /* §15.6.1 control barriers */
        { "workgroupBarrier",      16 },
        { "storageBarrier",        14 },
        { "textureBarrier",        14 },
        { "workgroupUniformLoad",  20 },
        /* §15.6.2 derivative-quad ops on invocation-specified values */
        { "dpdx",                   4 },
        { "dpdxCoarse",            10 },
        { "dpdxFine",               8 },
        { "dpdy",                   4 },
        { "dpdyCoarse",            10 },
        { "dpdyFine",               8 },
        { "fwidth",                 6 },
        { "fwidthCoarse",          12 },
        { "fwidthFine",            10 },
        /* §15.6.2 texture sampling with implicit derivatives */
        { "textureSample",         13 },
        { "textureSampleBias",     17 },
        { "textureSampleCompare",  20 },
    };
    for (size_t i = 0; i < sizeof kNames / sizeof kNames[0]; i++) {
        if (kNames[i].l == len && memcmp(kNames[i].n, nm, len) == 0) return 1;
    }
    return 0;
}

/* §15.6.3 subgroup operations + §15.6.4 quad operations must run in
 * subgroup-uniform / quad-uniform control flow.  All `subgroup*` and
 * `quad*` builtin names are subject to the rule; both map to the
 * `subgroup_uniformity` filter rule per §3.8.3.  Detection by prefix
 * (faster than enumerating ~28 names) and matches the actual builtin
 * naming convention in `kBuiltinTable`. */
static int call_requires_subgroup_uniform_cf(const char *nm, uint32_t len) {
    if (len >= 8 && memcmp(nm, "subgroup", 8) == 0) return 1;
    if (len >= 4 && memcmp(nm, "quad",     4) == 0) return 1;
    return 0;
}

/* Per-fn uniformity-requirement bits — populated by `compute_fn_uniformity_reqs`
 * before `validate_uniformity` runs.  Indexed by symbol position in
 * `v->res->all_decls`. */
#define UR_DERIVATIVE   0x1u   /* fn (or transitive callee) needs uniform CF for §15.6.2 */
#define UR_SUBGROUP     0x2u   /* same, for §15.6.3 / §15.6.4 */

/* Pre-scan: every fn's body for direct calls to uniform-CF-requiring
 * builtins.  Returns per-fn bitset; later propagated through the call
 * graph by `propagate_fn_uniformity_reqs`. */
typedef struct {
    WGSLValidator *v;
    uint32_t       bits;       /* output: collected for the fn currently being scanned */
} ReqScanCtx;

static void scan_fn_direct_reqs(ReqScanCtx *cx, WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_CALL && n->child_count > 0) {
        WGSLNode *fn = n->children[0];
        if (fn && fn->kind == WGSL_NODE_EXPR_IDENT) {
            const char *nm = cx->v->src->bytes +
                             (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
            uint32_t len = (uint32_t)(fn->payload[0] >> 32);
            if (call_requires_uniform_cf(nm, len))           cx->bits |= UR_DERIVATIVE;
            if (call_requires_subgroup_uniform_cf(nm, len))  cx->bits |= UR_SUBGROUP;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_fn_direct_reqs(cx, n->children[i]);
    }
}

/* Propagate transitively: if fn F calls fn G, F gets G's bits.  Run
 * to fixed point (cap iterations at decl_count to bound runtime in
 * pathological cycles). */
typedef struct {
    uint32_t *reqs;            /* same pointer used during the BFS */
    uint32_t  caller_bits;     /* bits to OR into each callee */
} PropCtx;

static void prop_edge_cb(WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx) {
    /* In our propagation, edges go caller→callee in the call graph.
     * We walk *from* each fn's body and collect its callees' bits
     * back into the caller — so swap the direction by reading the
     * callee's bits and OR'ing them into the caller's slot. */
    PropCtx *cx = (PropCtx *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0) return;
    cx->caller_bits |= cx->reqs[idx];
}

static uint32_t *compute_fn_uniformity_reqs(WGSLValidator *v) {
    if (!v->res) return NULL;
    size_t n = v->res->all_decl_count;
    uint32_t *reqs = (uint32_t *)calloc(n, sizeof *reqs);
    if (!reqs) return NULL;
    /* Phase 1: per-fn direct-call scan. */
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (s->kind != WGSL_SYM_FUNCTION || !s->ast) continue;
        if (s->ast->child_count == 0) continue;
        WGSLNode *body = s->ast->children[s->ast->child_count - 1];
        ReqScanCtx cx = { v, 0 };
        scan_fn_direct_reqs(&cx, body);
        if (function_filter_severity(
                v, s->ast, "derivative_uniformity", 21) == 0)
        {
            cx.bits &= ~UR_DERIVATIVE;
        }
        if (function_filter_severity(
                v, s->ast, "subgroup_uniformity", 19) == 0)
        {
            cx.bits &= ~UR_SUBGROUP;
        }
        reqs[i] = cx.bits;
    }
    /* Phase 2: fixpoint propagate via call graph (caller picks up
     * each callee's bits).  Cap iterations at n to bound the worst
     * case; in practice fixpoint converges in 1-2 sweeps. */
    int changed = 1;
    int iter = 0;
    while (changed && iter < (int)n + 4) {
        changed = 0;
        for (size_t i = 0; i < n; i++) {
            WGSLSymbol *s = v->res->all_decls[i];
            if (s->kind != WGSL_SYM_FUNCTION) continue;
            PropCtx pc = { reqs, reqs[i] };
            wgsl_val_enum_call_edges(v, s, prop_edge_cb, &pc);
            if (pc.caller_bits != reqs[i]) {
                reqs[i] = pc.caller_bits;
                changed = 1;
            }
        }
        iter += 1;
    }
    return reqs;
}

/* Look up a fn symbol's call-site name (EXPR_IDENT child) and return
 * the resolved symbol if it's a user-defined fn (not a builtin). */
static WGSLSymbol *call_user_fn_sym(WGSLNode *call) {
    if (!call || call->child_count == 0) return NULL;
    WGSLNode *fn = call->children[0];
    if (!fn || fn->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *s = wgsl_node_resolved_symbol(fn);
    if (!s || s->kind != WGSL_SYM_FUNCTION) return NULL;
    return s;
}

static int uniform_param_builtin_name(
    UniformCtx *cx, WGSLSymbol *sym, const char **out_name, uint32_t *out_len)
{
    if (out_name) *out_name = NULL;
    if (out_len) *out_len = 0;
    if (!cx || !sym || !sym->ast || sym->ast->kind != WGSL_NODE_DECL_PARAM)
        return 0;
    uint32_t attr_count = (uint32_t)(sym->ast->payload[1] & 0xFFFFFFFFu);
    WGSLNode *attr = wgsl_val_find_attr(cx->v, sym->ast, 0, attr_count, "builtin");
    if (!attr || attr->child_count < 2) return 0;
    WGSLNode *arg = attr->children[1];
    if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = (uint32_t)(arg->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(arg->payload[0] >> 32);
    if (out_name) *out_name = cx->v->src->bytes + off;
    if (out_len) *out_len = len;
    return 1;
}

static WGSLSymbol *uniform_param_owner_function(UniformCtx *cx, WGSLSymbol *param) {
    if (!cx || !cx->v || !cx->v->res || !param) return NULL;
    for (size_t i = 0; i < cx->v->res->all_decl_count; i++) {
        WGSLSymbol *fn = cx->v->res->all_decls[i];
        if (!fn || fn->kind != WGSL_SYM_FUNCTION ||
            !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION)
            continue;
        uint32_t P = (uint32_t)(fn->ast->payload[1] >> 32);
        for (uint32_t p = 0; p < P; p++) {
            if (uniform_fn_param_symbol(cx->v, fn, p) == param) return fn;
        }
    }
    return NULL;
}

static int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    const char *nm = NULL;
    uint32_t len = 0;
    if (!uniform_param_builtin_name(cx, sym, &nm, &len)) return 0;
    WGSLSymbol *fn = uniform_param_owner_function(cx, sym);
    int is_compute = fn && (fn->flags & WGSL_SYM_FLAG_COMPUTE);
    return (len == 12 && memcmp(nm, "workgroup_id", 12) == 0) ||
           (len == 14 && memcmp(nm, "num_workgroups", 14) == 0) ||
           (is_compute &&
            ((len == 13 && memcmp(nm, "subgroup_size", 13) == 0) ||
             (len == 13 && memcmp(nm, "num_subgroups", 13) == 0)));
}

static int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    const char *nm = NULL;
    uint32_t len = 0;
    if (!uniform_param_builtin_name(cx, sym, &nm, &len)) return 0;
    return uniform_param_builtin_is_uniform(cx, sym) ||
           (len == 13 && memcmp(nm, "subgroup_size", 13) == 0) ||
           (len == 13 && memcmp(nm, "num_subgroups", 13) == 0) ||
           (len == 11 && memcmp(nm, "subgroup_id", 11) == 0);
}

static int uniform_module_var_is_readonly(const WGSLSymbol *sym) {
    if (!sym || sym->kind != WGSL_SYM_VAR) return 0;
    if (sym->as == WGSL_AS_UNIFORM || sym->as == WGSL_AS_HANDLE)
        return 1;
    if (sym->as == WGSL_AS_STORAGE && sym->am == WGSL_ACCESS_READ)
        return 1;
    return 0;
}

static int call_name(UniformCtx *cx, WGSLNode *call, const char **name, uint32_t *len) {
    if (!call || call->kind != WGSL_NODE_EXPR_CALL ||
        call->child_count == 0 || !name || !len)
        return 0;
    WGSLNode *fn = call->children[0];
    if (!fn) return 0;
    if (fn->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && fn->child_count > 0) {
        fn = fn->children[0];
    }
    if (!fn || fn->kind != WGSL_NODE_EXPR_IDENT) return 0;
    WGSLSymbol *sym = wgsl_node_resolved_symbol(fn);
    if (sym && sym->name) {
        *name = sym->name;
        *len = sym->name_len;
        return 1;
    }
    if (!cx || !cx->v || !cx->v->src) return 0;
    *name = cx->v->src->bytes + (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
    *len = (uint32_t)(fn->payload[0] >> 32);
    return 1;
}

static int uniform_param_index(
    WGSLValidator *v, const WGSLSymbol *fn, const WGSLSymbol *param,
    uint32_t *out)
{
    if (!v || !fn || !param || !out ||
        !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION)
        return 0;
    uint32_t P = (uint32_t)(fn->ast->payload[1] >> 32);
    for (uint32_t i = 0; i < P; i++) {
        if (uniform_fn_param_symbol(v, fn, i) == param) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

static WGSLSymbol *uniform_lhs_pointer_param(WGSLNode *lhs) {
    if (!lhs) return NULL;
    switch ((WGSLNodeKind)lhs->kind) {
    case WGSL_NODE_EXPR_INDIRECTION:
        if (lhs->child_count > 0) {
            WGSLNode *p = lhs->children[0];
            if (p && p->kind == WGSL_NODE_EXPR_IDENT) {
                WGSLSymbol *sym = wgsl_node_resolved_symbol(p);
                if (sym && sym->kind == WGSL_SYM_PARAM) return sym;
            }
        }
        return NULL;
    case WGSL_NODE_EXPR_MEMBER:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_PAREN:
        return lhs->child_count > 0
            ? uniform_lhs_pointer_param(lhs->children[0])
            : NULL;
    default:
        return NULL;
    }
}

static void apply_user_call_pointer_effects(
    UniformCtx *cx, WGSLNode *call, int is_divergent,
    const uint32_t *fn_reqs);

static int uniform_pointer_param_lhs_address_is_uniform(
    UniformCtx *cx, WGSLNode *lhs, WGSLSymbol *param, WGSLNode *actual,
    int is_divergent, const uint32_t *fn_reqs)
{
    if (!lhs) return 0;
    switch ((WGSLNodeKind)lhs->kind) {
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_ADDR_OF:
        return lhs->child_count > 0 &&
               uniform_pointer_param_lhs_address_is_uniform(
                   cx, lhs->children[0], param, actual,
                   is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_MEMBER:
        return lhs->child_count > 0 &&
               uniform_pointer_param_lhs_address_is_uniform(
                   cx, lhs->children[0], param, actual,
                   is_divergent, fn_reqs);
    case WGSL_NODE_EXPR_INDEX: {
        int base_u = lhs->child_count > 0 &&
            uniform_pointer_param_lhs_address_is_uniform(
                cx, lhs->children[0], param, actual,
                is_divergent, fn_reqs);
        int idx_u = lhs->child_count > 1 &&
            analyze_uniform_expr(cx, lhs->children[1],
                                 is_divergent, fn_reqs);
        return base_u && idx_u;
    }
    case WGSL_NODE_EXPR_INDIRECTION: {
        if (lhs->child_count == 0) return 0;
        WGSLNode *ptr = lhs->children[0];
        if (ptr && ptr->kind == WGSL_NODE_EXPR_PAREN && ptr->child_count > 0) {
            ptr = ptr->children[0];
        }
        if (ptr && ptr->kind == WGSL_NODE_EXPR_IDENT &&
            wgsl_node_resolved_symbol(ptr) == param)
        {
            return uniform_reference_address_is_uniform(
                cx, actual, is_divergent, fn_reqs);
        }
        return uniform_pointer_address_is_uniform(
            cx, ptr, is_divergent, fn_reqs);
    }
    default:
        return analyze_uniform_expr(cx, lhs, is_divergent, fn_reqs);
    }
}

static void apply_user_call_effects_from_stmt(
    UniformCtx *cx, WGSLSymbol *callee, WGSLNode *call, WGSLNode *stmt,
    int is_divergent, const uint32_t *fn_reqs)
{
    if (!cx || !callee || !call || !stmt) return;
    is_divergent = uniform_normalize_divergence(is_divergent);
    if ((stmt->kind == WGSL_NODE_STMT_IF ||
         stmt->kind == WGSL_NODE_STMT_ELSE_IF) &&
        stmt->child_count >= 2)
    {
        int cond_u = analyze_uniform_expr(
            cx, stmt->children[0], is_divergent, fn_reqs);
        int branch_div = uniform_diverge_for_condition(
            is_divergent, stmt->children[0], cond_u);
        for (uint32_t i = 1; i < stmt->child_count; i++) {
            apply_user_call_effects_from_stmt(
                cx, callee, call, stmt->children[i], branch_div, fn_reqs);
        }
        return;
    }
    if (stmt->kind == WGSL_NODE_STMT_ASSIGN ||
        stmt->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN)
    {
        WGSLNode *lhs = stmt->child_count > 0 ? stmt->children[0] : NULL;
        WGSLNode *rhs = stmt->child_count > 1 ? stmt->children[1] : NULL;
        WGSLSymbol *param = uniform_lhs_pointer_param(lhs);
        uint32_t param_idx = 0;
        if (param && uniform_param_index(cx->v, callee, param, &param_idx) &&
            call->child_count > 1 + param_idx)
        {
            WGSLNode *actual = call->children[1 + param_idx];
            WGSLSymbol *actual_root =
                uniform_memory_root(actual, 0);
            if (actual_root && actual_root->kind == WGSL_SYM_VAR &&
                (actual_root->as == WGSL_AS_FUNCTION ||
                 actual_root->as == WGSL_AS_NONE))
            {
                int lhs_u = lhs ? uniform_pointer_param_lhs_address_is_uniform(
                    cx, lhs, param, actual, is_divergent, fn_reqs)
                    : !(is_divergent & UD_WORKGROUP);
                int rhs_u = rhs ? analyze_uniform_expr(
                    cx, rhs, is_divergent, fn_reqs)
                    : !(is_divergent & UD_WORKGROUP);
                uniform_state_set(
                    cx, actual_root,
                    lhs_u && rhs_u && !(is_divergent & UD_WORKGROUP));
            }
        }
    }
    if (stmt->kind == WGSL_NODE_EXPR_CALL) {
        apply_user_call_pointer_effects(cx, stmt, is_divergent, fn_reqs);
    }
    UniformBehavior b = WGSL_BEHAVIOR_NEXT;
    for (uint32_t i = 0; i < stmt->child_count; i++) {
        WGSLNode *c = stmt->children[i];
        if (!c) continue;
        if ((b & WGSL_BEHAVIOR_NEXT) == 0) break;
        if (!uniform_expr_node(c)) {
            apply_user_call_effects_from_stmt(
                cx, callee, call, c, is_divergent, fn_reqs);
            UniformBehavior cb = uniform_beh_stmt(c);
            b = (UniformBehavior)((b & ~WGSL_BEHAVIOR_NEXT) | cb);
        }
    }
}

static void apply_user_call_pointer_effects(
    UniformCtx *cx, WGSLNode *call, int is_divergent,
    const uint32_t *fn_reqs)
{
    WGSLSymbol *callee = call_user_fn_sym(call);
    if (!callee || !callee->ast || callee->ast->child_count == 0) return;
    WGSLNode *body = callee->ast->children[callee->ast->child_count - 1];
    if (!body) return;
    apply_user_call_effects_from_stmt(
        cx, callee, call, body, is_divergent, fn_reqs);
}

static int call_reads_readwrite_storage_texture(
    UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len)
{
    if (!cx || !cx->v || !cx->v->tc || !call) return 0;
    if (!(len == 11 && memcmp(nm, "textureLoad", 11) == 0)) return 0;
    if (call->child_count < 2) return 0;
    WGSLTypeInfo *tex = wgsl_typecheck_type_of(cx->v->tc, call->children[1]);
    if (tex && tex->kind == WGSL_TYPE_REF) tex = (WGSLTypeInfo *)tex->ref;
    return tex && tex->kind == WGSL_TYPE_TEXTURE &&
           tex->rows == WGSL_ACCESS_READ_WRITE;
}

static int call_result_may_be_nonuniform(
    UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len)
{
    if (call_requires_uniform_cf(nm, len)) return 1;
    if (call_requires_subgroup_uniform_cf(nm, len)) return 1;
    if (call_reads_readwrite_storage_texture(cx, call, nm, len)) return 1;
    return 0;
}

static int call_result_is_known_uniform(const char *nm, uint32_t len) {
    return len == 20 && memcmp(nm, "workgroupUniformLoad", 20) == 0;
}

static int call_result_is_subgroup_uniform(const char *nm, uint32_t len) {
    static const struct { const char *n; uint32_t l; } kNames[] = {
        { "subgroupAdd",            11 },
        { "subgroupMul",            11 },
        { "subgroupMax",            11 },
        { "subgroupMin",            11 },
        { "subgroupAll",            11 },
        { "subgroupAny",            11 },
        { "subgroupAnd",            11 },
        { "subgroupOr",             10 },
        { "subgroupXor",            11 },
        { "subgroupBallot",         14 },
        { "subgroupBroadcast",      17 },
        { "subgroupBroadcastFirst", 22 },
    };
    for (size_t i = 0; i < sizeof kNames / sizeof kNames[0]; i++) {
        if (kNames[i].l == len && memcmp(kNames[i].n, nm, len) == 0) return 1;
    }
    return 0;
}

static void uniform_collect_returns(
    UniformCtx *cx, WGSLNode *n, const uint32_t *fn_reqs,
    int *saw_return, int *all_uniform)
{
    if (!n || !saw_return || !all_uniform) return;
    if (n->kind == WGSL_NODE_STMT_RETURN) {
        *saw_return = 1;
        WGSLNode *expr = n->child_count > 0 ? n->children[0] : NULL;
        if (expr && !analyze_uniform_expr(cx, expr, UD_NONE, fn_reqs)) {
            *all_uniform = 0;
        }
        return;
    }
    if (n->kind == WGSL_NODE_DECL_FUNCTION) return;
    for (uint32_t i = 0; i < n->child_count; i++) {
        uniform_collect_returns(
            cx, n->children[i], fn_reqs, saw_return, all_uniform);
    }
}

static int user_function_result_is_uniform(
    UniformCtx *cx, WGSLSymbol *fn, const uint32_t *fn_reqs)
{
    if (!cx || !fn || !fn->ast || fn->ast->child_count == 0) return 1;
    if (cx->init_depth >= 16) return 0;
    WGSLNode *body = fn->ast->children[fn->ast->child_count - 1];
    if (!body) return 1;
    uint8_t *saved_known = NULL, *saved_uniform = NULL;
    int have_state = uniform_state_copy(cx, &saved_known, &saved_uniform);
    int saw_return = 0;
    int all_uniform = 1;
    cx->init_depth += 1;
    uniform_collect_returns(cx, body, fn_reqs, &saw_return, &all_uniform);
    cx->init_depth -= 1;
    if (have_state) {
        uniform_state_restore(cx, saved_known, saved_uniform);
        free(saved_known);
        free(saved_uniform);
    }
    return !saw_return || all_uniform;
}

static void check_call_requirements(
    UniformCtx *cx, WGSLNode *call, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !call || call->kind != WGSL_NODE_EXPR_CALL) return;
    WGSLValidator *v = cx->v;
    if (call->child_count == 0) return;
    WGSLNode *fn = call->children[0];
    if (!fn) return;
    if (fn->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && fn->child_count > 0) {
        fn = fn->children[0];
    }
    if (!fn || fn->kind != WGSL_NODE_EXPR_IDENT) return;

    const char *nm = v->src->bytes + (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(fn->payload[0] >> 32);
    WGSLSymbol *fsym = wgsl_node_resolved_symbol(fn);
    if (fsym && fsym->name) {
        nm = fsym->name;
        len = fsym->name_len;
    }

    is_divergent = uniform_normalize_divergence(is_divergent);
    if ((is_divergent & UD_WORKGROUP) && call_requires_uniform_cf(nm, len)) {
        wgsl_val_filterable(v, call, WGSL_DIAG_ERROR, "derivative_uniformity",
            "collective operation '%.*s' must be called in "
            "uniform control flow", (int)len, nm);
    }
    if ((is_divergent & UD_SUBGROUP) &&
        call_requires_subgroup_uniform_cf(nm, len))
    {
        wgsl_val_filterable(v, call, WGSL_DIAG_ERROR, "subgroup_uniformity",
            "subgroup / quad operation '%.*s' must be called in "
            "uniform control flow", (int)len, nm);
    }
    if (is_divergent && fn_reqs) {
        WGSLSymbol *callee = call_user_fn_sym(call);
        if (callee) {
            int idx = wgsl_val_sym_index(v, callee);
            if (idx >= 0) {
                uint32_t bits = fn_reqs[idx];
                if ((is_divergent & UD_WORKGROUP) && (bits & UR_DERIVATIVE)) {
                    int sev = function_filter_severity(
                        v, callee->ast, "derivative_uniformity", 21);
                    if (sev < 0 || sev == WGSL_DIAG_ERROR) {
                        wgsl_val_error(v, call,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses derivative-required ops)",
                            (int)callee->name_len, callee->name);
                    } else if (sev > 0) {
                        wgsl_diag_emit_at(v->diag, v->src,
                            (WGSLDiagSeverity)sev,
                            call->span_offset, call->span_length, NULL,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses derivative-required ops)",
                            (int)callee->name_len, callee->name);
                    }
                }
                if ((is_divergent & UD_SUBGROUP) && (bits & UR_SUBGROUP)) {
                    int sev = function_filter_severity(
                        v, callee->ast, "subgroup_uniformity", 19);
                    if (sev < 0 || sev == WGSL_DIAG_ERROR) {
                        wgsl_val_error(v, call,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses subgroup / quad ops)",
                            (int)callee->name_len, callee->name);
                    } else if (sev > 0) {
                        wgsl_diag_emit_at(v->diag, v->src,
                            (WGSLDiagSeverity)sev,
                            call->span_offset, call->span_length, NULL,
                            "call to '%.*s' must be in uniform "
                            "control flow (callee transitively "
                            "uses subgroup / quad ops)",
                            (int)callee->name_len, callee->name);
                    }
                }
            }
        }
    }
}

static void check_call_param_requirements(
    UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len,
    const int *arg_uniform, uint32_t argc, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!cx || !call || !nm) return;
    if (len == 20 && memcmp(nm, "workgroupUniformLoad", 20) == 0 &&
        call->child_count > 1 &&
        !uniform_pointer_address_is_uniform(
            cx, call->children[1], is_divergent, fn_reqs))
    {
        wgsl_val_filterable(cx->v, call, WGSL_DIAG_ERROR,
            "derivative_uniformity",
            "argument to 'workgroupUniformLoad' must have a uniform address");
    }

    int arg_idx = -1;
    if ((len == 17 && memcmp(nm, "subgroupShuffleUp", 17) == 0) ||
        (len == 19 && memcmp(nm, "subgroupShuffleDown", 19) == 0) ||
        (len == 18 && memcmp(nm, "subgroupShuffleXor", 18) == 0))
    {
        arg_idx = 1; /* delta / mask */
    }
    if (arg_idx >= 0 && (uint32_t)arg_idx < argc &&
        arg_uniform && !arg_uniform[arg_idx])
    {
        const char *what =
            (len == 18 && memcmp(nm, "subgroupShuffleXor", 18) == 0)
                ? "mask" : "delta";
        wgsl_val_filterable(cx->v, call, WGSL_DIAG_ERROR,
            "subgroup_uniformity",
            "argument '%s' to '%.*s' must be uniform",
            what, (int)len, nm);
    }
}

static int analyze_uniform_expr(
    UniformCtx *cx, WGSLNode *n, int is_divergent,
    const uint32_t *fn_reqs)
{
    if (!n) return 1;
    is_divergent = uniform_normalize_divergence(is_divergent);
    int u = (is_divergent & UD_WORKGROUP) ? 0 : 1;
    int sg = (is_divergent & UD_SUBGROUP) ? 0 : 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        break;

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        int src_u = 1;
        int src_sg = 1;
        if (sym) {
            if (sym->kind == WGSL_SYM_VAR) {
                if (uniform_module_var_is_readonly(sym)) {
                    src_u = 1;
                    src_sg = 1;
                } else if (sym->as == WGSL_AS_FUNCTION || sym->as == WGSL_AS_NONE) {
                    src_u = decl_init_is_uniform(cx, sym);
                    src_sg = src_u;
                } else {
                    src_u = 0;
                    src_sg = 0;
                }
            } else if (sym->kind == WGSL_SYM_LET) {
                WGSLNode *init = uniform_decl_init_expr(sym->ast);
                src_u = decl_init_is_uniform(cx, sym);
                src_sg = src_u || (init && uniform_expr_is_subgroup_uniform(init));
            } else if (sym->kind == WGSL_SYM_PARAM) {
                src_u = uniform_param_builtin_is_uniform(cx, sym) ||
                        parameter_is_uniform(cx, sym);
                src_sg = src_u ||
                         uniform_param_builtin_is_subgroup_uniform(cx, sym);
            } else if (sym->kind == WGSL_SYM_CONST ||
                       sym->kind == WGSL_SYM_OVERRIDE ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN ||
                       sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                       sym->kind == WGSL_SYM_PREDECLARED_FN)
            {
                src_u = 1;
                src_sg = 1;
            }
        }
        u = !(is_divergent & UD_WORKGROUP) && src_u;
        sg = !(is_divergent & UD_SUBGROUP) && src_sg;
        break;
    }

    case WGSL_NODE_EXPR_BINARY: {
        if (n->child_count < 2) break;
        WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
        int lhs_u = analyze_uniform_expr(cx, n->children[0],
                                         is_divergent, fn_reqs);
        int lhs_sg = uniform_expr_is_subgroup_uniform(n->children[0]);
        int rhs_div = is_divergent;
        if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
            rhs_div = uniform_diverge_for_condition(rhs_div,
                                                    n->children[0], lhs_u);
        }
        int rhs_u = analyze_uniform_expr(cx, n->children[1],
                                         rhs_div, fn_reqs);
        int rhs_sg = uniform_expr_is_subgroup_uniform(n->children[1]);
        u = lhs_u && rhs_u;
        sg = lhs_sg && rhs_sg && !(is_divergent & UD_SUBGROUP);
        break;
    }

    case WGSL_NODE_EXPR_CALL: {
        check_call_requirements(cx, n, is_divergent, fn_reqs);
        const char *nm = NULL;
        uint32_t len = 0;
        if (!call_name(cx, n, &nm, &len)) {
            nm = "";
            len = 0;
        }
        int args_u[16] = {0};
        uint32_t argc = n->child_count > 0 ? n->child_count - 1 : 0;
        if (argc > (uint32_t)(sizeof args_u / sizeof args_u[0])) {
            argc = (uint32_t)(sizeof args_u / sizeof args_u[0]);
        }
        int all_args_u = !(is_divergent & UD_WORKGROUP);
        int all_args_sg = !(is_divergent & UD_SUBGROUP);
        for (uint32_t i = 0; i < argc; i++) {
            args_u[i] = analyze_uniform_expr(cx, n->children[1 + i],
                                             is_divergent, fn_reqs);
            if (!args_u[i]) all_args_u = 0;
            if (!uniform_expr_is_subgroup_uniform(n->children[1 + i])) {
                all_args_sg = 0;
            }
        }
        check_call_param_requirements(
            cx, n, nm, len, args_u, argc, is_divergent, fn_reqs);
        int known_u = call_result_is_known_uniform(nm, len);
        if (known_u && len == 20 &&
            memcmp(nm, "workgroupUniformLoad", 20) == 0)
        {
            known_u = n->child_count > 1 &&
                uniform_pointer_address_is_uniform(
                    cx, n->children[1], is_divergent, fn_reqs);
        }
        u = known_u
            ? !(is_divergent & UD_WORKGROUP)
            : (all_args_u &&
               !call_result_may_be_nonuniform(cx, n, nm, len));
        WGSLSymbol *callee = call_user_fn_sym(n);
        if (callee && !user_function_result_is_uniform(cx, callee, fn_reqs)) {
            u = 0;
        }
        sg = u || (all_args_sg && call_result_is_subgroup_uniform(nm, len));
        break;
    }

    case WGSL_NODE_EXPR_INDIRECTION: {
        if (n->child_count > 0) {
            analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs);
            WGSLSymbol *root = uniform_memory_root(n->children[0], 0);
            int addr_u = uniform_pointer_address_is_uniform(
                cx, n->children[0], is_divergent, fn_reqs);
            u = !(is_divergent & UD_WORKGROUP) &&
                addr_u &&
                uniform_root_contents_are_uniform(cx, root);
            sg = u;
        }
        break;
    }

    case WGSL_NODE_EXPR_INDEX: {
        int obj_u = n->child_count > 0
            ? analyze_uniform_expr(cx, n->children[0], is_divergent, fn_reqs)
            : !(is_divergent & UD_WORKGROUP);
        int obj_sg = n->child_count > 0
            ? uniform_expr_is_subgroup_uniform(n->children[0])
            : !(is_divergent & UD_SUBGROUP);
        int idx_u = n->child_count > 1
            ? analyze_uniform_expr(cx, n->children[1], is_divergent, fn_reqs)
            : !(is_divergent & UD_WORKGROUP);
        int idx_sg = n->child_count > 1
            ? uniform_expr_is_subgroup_uniform(n->children[1])
            : !(is_divergent & UD_SUBGROUP);
        u = obj_u && idx_u;
        sg = obj_sg && idx_sg && !(is_divergent & UD_SUBGROUP);
        break;
    }

    default:
        u = !(is_divergent & UD_WORKGROUP);
        sg = !(is_divergent & UD_SUBGROUP);
        for (uint32_t i = 0; i < n->child_count; i++) {
            if (!analyze_uniform_expr(cx, n->children[i], is_divergent, fn_reqs)) {
                u = 0;
            }
            if (!uniform_expr_is_subgroup_uniform(n->children[i])) {
                sg = 0;
            }
        }
        break;
    }

    if (u) {
        n->flags &= (uint16_t)~WGSL_FLAG_NONUNIFORM;
        n->flags |= (uint16_t)WGSL_FLAG_SUBGROUP_UNIFORM;
    } else {
        n->flags |= (uint16_t)WGSL_FLAG_NONUNIFORM;
        if (sg) n->flags |= (uint16_t)WGSL_FLAG_SUBGROUP_UNIFORM;
        else    n->flags &= (uint16_t)~WGSL_FLAG_SUBGROUP_UNIFORM;
    }
    return u;
}

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
typedef struct {
    UniformCtx *ucx;
    uint8_t    *nonuniform;
    int        *changed;
} PtrParamCtx;

static void scan_pointer_param_calls(PtrParamCtx *pcx, WGSLNode *n) {
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

static uint8_t *compute_param_nonuniform(WGSLValidator *v) {
    if (!v || !v->res) return NULL;
    size_t n = v->res->all_decl_count;
    uint8_t *nonuniform = (uint8_t *)calloc(n, sizeof *nonuniform);
    if (!nonuniform) return NULL;

    UniformCtx seed = { v, nonuniform, n, 0, NULL, NULL };
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!s || s->kind != WGSL_SYM_FUNCTION || !s->ast) continue;
        if (!(s->flags & (WGSL_SYM_FLAG_VERTEX |
                          WGSL_SYM_FLAG_FRAGMENT |
                          WGSL_SYM_FLAG_COMPUTE)))
            continue;
        uint32_t P  = (uint32_t)(s->ast->payload[1] >> 32);
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
        UniformCtx ucx = { v, nonuniform, n, 0, NULL, NULL };
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

static void check_uniformity_block(
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
                if ((uint32_t)((clause->payload[0] >> 32) & 1u)) has_default = 1;
                uniform_state_restore(cx, base_known, base_uniform);
                check_uniformity_block(cx, clause, sel_div, fn_reqs);
                UniformBehavior cb = uniform_beh_stmt(clause);
                if (cb & (WGSL_BEHAVIOR_NEXT | WGSL_BEHAVIOR_BREAK)) {
                    uniform_state_join_current_into(cx, acc_known, acc_uniform);
                }
                uniform_state_restore(cx, base_known, base_uniform);
                uniform_collect_break_exits(
                    cx, clause, sel_div, fn_reqs,
                    exit_known, exit_uniform, &saw_exit);
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
        uint32_t sel_count = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
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
    size_t nsyms = v->res ? v->res->all_decl_count : 0;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (n && n->kind == WGSL_NODE_DECL_FUNCTION && n->child_count > 0) {
            /* §4.2: push a diagnostic scope for per-function
             * @diagnostic attributes.  Walk the fn's attribute range
             * for @diagnostic attrs and register their filters. */
            wgsl_diag_push_scope(v->diag);
            uint32_t FA = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            set_diagnostic_attr_filters(v, n->children, FA);

            uint8_t *sym_uniform = nsyms ? (uint8_t *)calloc(nsyms, 1) : NULL;
            uint8_t *sym_known = nsyms ? (uint8_t *)calloc(nsyms, 1) : NULL;
            UniformCtx ucx = {
                v, param_nonuniform, nsyms, 0, sym_uniform, sym_known
            };
            check_uniformity_block(&ucx, n->children[n->child_count - 1], 0, fn_reqs);
            free(sym_uniform);
            free(sym_known);
            wgsl_diag_pop_scope(v->diag);
        }
    }
    free(param_nonuniform);
    free(fn_reqs);
}
