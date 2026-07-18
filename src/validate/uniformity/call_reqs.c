/**
 * @file call_reqs.c — call CF requirements + pointer effects
 */
#include "internal/uniformity_priv.h"

int is_uniform_expr(UniformCtx *cx, WGSLNode *n) {
    return analyze_uniform_expr(cx, n, 0, NULL);
}

/* §15.6 — single embedded uniformity classification for builtin FUNCTIONS.
 * This one table + `builtin_call_uniformity` replaces the former scattered
 * strcmp-lists (`call_requires_uniform_cf` / `call_requires_subgroup_
 * uniform_cf` / `call_result_is_known_uniform` / `call_result_is_subgroup_
 * uniform`).  Hand-maintained single owner, NOT def/wgsl.def-generated: to
 * give a builtin a uniformity property, add/extend one row here.
 *
 * The subgroup* / quad* "requires subgroup-uniform CF" rule (§15.6.3/.4)
 * stays a name PREFIX in the lookup — drift-proof over the ~28 subgroup /
 * quad ops — so it is NOT repeated per row.  Rows map to the
 * `derivative_uniformity` / `subgroup_uniformity` filter rules (§3.8.3). */
static const struct { const char *n; uint32_t l; uint32_t f; } kBuiltinUnif[] = {
    /* §15.6.1 control barriers → must be called in uniform CF.
     * workgroupUniformLoad additionally RETURNS a known-uniform value. */
    { "workgroupBarrier",      16, UNIF_REQ_CF },
    { "storageBarrier",        14, UNIF_REQ_CF },
    { "textureBarrier",        14, UNIF_REQ_CF },
    { "workgroupUniformLoad",  20, UNIF_REQ_CF | UNIF_RESULT_KNOWN },
    /* §15.6.2 derivative-quad ops on invocation-specified values */
    { "dpdx",                   4, UNIF_REQ_CF },
    { "dpdxCoarse",            10, UNIF_REQ_CF },
    { "dpdxFine",               8, UNIF_REQ_CF },
    { "dpdy",                   4, UNIF_REQ_CF },
    { "dpdyCoarse",            10, UNIF_REQ_CF },
    { "dpdyFine",               8, UNIF_REQ_CF },
    { "fwidth",                 6, UNIF_REQ_CF },
    { "fwidthCoarse",          12, UNIF_REQ_CF },
    { "fwidthFine",            10, UNIF_REQ_CF },
    /* §15.6.2 texture sampling with implicit derivatives */
    { "textureSample",         13, UNIF_REQ_CF },
    { "textureSampleBias",     17, UNIF_REQ_CF },
    { "textureSampleCompare",  20, UNIF_REQ_CF },
    /* §15.6.3 subgroup reductions / broadcasts → result is subgroup-uniform.
     * (These also get UNIF_REQ_SG_CF from the subgroup* prefix below.) */
    { "subgroupAdd",            11, UNIF_RESULT_SG },
    { "subgroupMul",            11, UNIF_RESULT_SG },
    { "subgroupMax",            11, UNIF_RESULT_SG },
    { "subgroupMin",            11, UNIF_RESULT_SG },
    { "subgroupAll",            11, UNIF_RESULT_SG },
    { "subgroupAny",            11, UNIF_RESULT_SG },
    { "subgroupAnd",            11, UNIF_RESULT_SG },
    { "subgroupOr",             10, UNIF_RESULT_SG },
    { "subgroupXor",            11, UNIF_RESULT_SG },
    { "subgroupBallot",         14, UNIF_RESULT_SG },
    { "subgroupBroadcast",      17, UNIF_RESULT_SG },
    { "subgroupBroadcastFirst", 22, UNIF_RESULT_SG },
};

/* Single lookup: table flags for `nm`, OR'd with the subgroup / quad
 * prefix rule.  Replaces the four former predicate functions. */
uint32_t builtin_call_uniformity(const char *nm, uint32_t len) {
    uint32_t f = 0;
    for (size_t i = 0; i < sizeof kBuiltinUnif / sizeof kBuiltinUnif[0]; i++) {
        if (kBuiltinUnif[i].l == len &&
            memcmp(kBuiltinUnif[i].n, nm, len) == 0)
        {
            f |= kBuiltinUnif[i].f;
            break;
        }
    }
    /* §15.6.3/.4 — every subgroup* / quad* op runs in subgroup-uniform CF. */
    if ((len >= 8 && memcmp(nm, "subgroup", 8) == 0) ||
        (len >= 4 && memcmp(nm, "quad",     4) == 0))
    {
        f |= UNIF_REQ_SG_CF;
    }
    return f;
}

/* Per-fn uniformity-requirement bits — populated by `compute_fn_uniformity_reqs`
 * before `validate_uniformity` runs.  Indexed by symbol position in
 * `v->res->all_decls`. */
/* UR_*: uniformity_priv.h */

/* Pre-scan: every fn's body for direct calls to uniform-CF-requiring
 * builtins.  Returns per-fn bitset; later propagated through the call
 * graph by `propagate_fn_uniformity_reqs`. */
/* ReqScanCtx: uniformity_priv.h */
void scan_fn_direct_reqs(ReqScanCtx *cx, WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_CALL && n->child_count > 0) {
        WGSLNode *fn = n->children[0];
        if (fn && fn->kind == WGSL_NODE_EXPR_IDENT) {
            const char *nm = cx->v->src->bytes +
                             wgsl_node_name_span(fn).offset;
            uint32_t len = wgsl_node_name_span(fn).length;
            uint32_t uf = builtin_call_uniformity(nm, len);
            if (uf & UNIF_REQ_CF)    cx->bits |= UR_DERIVATIVE;
            if (uf & UNIF_REQ_SG_CF) cx->bits |= UR_SUBGROUP;
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

void prop_edge_cb(WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx) {
    /* In our propagation, edges go caller→callee in the call graph.
     * We walk *from* each fn's body and collect its callees' bits
     * back into the caller — so swap the direction by reading the
     * callee's bits and OR'ing them into the caller's slot. */
    PropCtx *cx = (PropCtx *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0) return;
    cx->caller_bits |= cx->reqs[idx];
}

uint32_t *compute_fn_uniformity_reqs(WGSLValidator *v) {
    if (!v->res) return NULL;
    size_t n = v->res->all_decl_count;
    uint32_t *reqs = (uint32_t *)calloc(n, sizeof *reqs);
    if (!reqs) return NULL;
    /* First pass: per-function direct-call scan. */
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
    /* Fixpoint pass: propagate via call graph (caller picks up
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
WGSLSymbol *call_user_fn_sym(WGSLNode *call) {
    if (!call || call->child_count == 0) return NULL;
    WGSLNode *fn = call->children[0];
    if (!fn || fn->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *s = wgsl_node_resolved_symbol(fn);
    if (!s || s->kind != WGSL_SYM_FUNCTION) return NULL;
    return s;
}

int uniform_param_builtin_name(
    UniformCtx *cx, WGSLSymbol *sym, const char **out_name, uint32_t *out_len)
{
    if (out_name) *out_name = NULL;
    if (out_len) *out_len = 0;
    if (!cx || !sym || !sym->ast || sym->ast->kind != WGSL_NODE_DECL_PARAM)
        return 0;
    uint32_t attr_count = wgsl_param_attr_count(sym->ast);
    WGSLNode *attr = wgsl_val_find_attr(cx->v, sym->ast, 0, attr_count, "builtin");
    if (!attr || attr->child_count < 2) return 0;
    WGSLNode *arg = attr->children[1];
    if (!arg || arg->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = wgsl_node_name_span(arg).offset;
    uint32_t len = wgsl_node_name_span(arg).length;
    if (out_name) *out_name = cx->v->src->bytes + off;
    if (out_len) *out_len = len;
    return 1;
}

WGSLSymbol *uniform_param_owner_function(UniformCtx *cx, WGSLSymbol *param) {
    if (!cx || !cx->v || !cx->v->res || !param) return NULL;
    for (size_t i = 0; i < cx->v->res->all_decl_count; i++) {
        WGSLSymbol *fn = cx->v->res->all_decls[i];
        if (!fn || fn->kind != WGSL_SYM_FUNCTION ||
            !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION)
            continue;
        uint32_t P = wgsl_fn_param_count(fn->ast);
        for (uint32_t p = 0; p < P; p++) {
            if (uniform_fn_param_symbol(cx->v, fn, p) == param) return fn;
        }
    }
    return NULL;
}

int uniform_param_builtin_is_uniform(UniformCtx *cx, WGSLSymbol *sym) {
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

int uniform_param_builtin_is_subgroup_uniform(UniformCtx *cx, WGSLSymbol *sym) {
    const char *nm = NULL;
    uint32_t len = 0;
    if (!uniform_param_builtin_name(cx, sym, &nm, &len)) return 0;
    return uniform_param_builtin_is_uniform(cx, sym) ||
           (len == 13 && memcmp(nm, "subgroup_size", 13) == 0) ||
           (len == 13 && memcmp(nm, "num_subgroups", 13) == 0) ||
           (len == 11 && memcmp(nm, "subgroup_id", 11) == 0);
}

int uniform_module_var_is_readonly(const WGSLSymbol *sym) {
    if (!sym || sym->kind != WGSL_SYM_VAR) return 0;
    if (sym->as == WGSL_AS_UNIFORM || sym->as == WGSL_AS_HANDLE)
        return 1;
    if (sym->as == WGSL_AS_STORAGE && sym->am == WGSL_ACCESS_READ)
        return 1;
    return 0;
}

int call_name(UniformCtx *cx, WGSLNode *call, const char **name, uint32_t *len) {
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
    *name = cx->v->src->bytes + wgsl_node_name_span(fn).offset;
    *len = wgsl_node_name_span(fn).length;
    return 1;
}

int uniform_param_index(
    WGSLValidator *v, const WGSLSymbol *fn, const WGSLSymbol *param,
    uint32_t *out)
{
    if (!v || !fn || !param || !out ||
        !fn->ast || fn->ast->kind != WGSL_NODE_DECL_FUNCTION)
        return 0;
    uint32_t P = wgsl_fn_param_count(fn->ast);
    for (uint32_t i = 0; i < P; i++) {
        if (uniform_fn_param_symbol(v, fn, i) == param) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

WGSLSymbol *uniform_lhs_pointer_param(WGSLNode *lhs) {
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

void apply_user_call_pointer_effects(
    UniformCtx *cx, WGSLNode *call, int is_divergent,
    const uint32_t *fn_reqs);

int uniform_pointer_param_lhs_address_is_uniform(
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

void apply_user_call_effects_from_stmt(
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

void apply_user_call_pointer_effects(
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

int call_reads_readwrite_storage_texture(
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

int call_result_may_be_nonuniform(
    UniformCtx *cx, WGSLNode *call, const char *nm, uint32_t len)
{
    if (builtin_call_uniformity(nm, len) & (UNIF_REQ_CF | UNIF_REQ_SG_CF))
        return 1;
    if (call_reads_readwrite_storage_texture(cx, call, nm, len)) return 1;
    return 0;
}

void uniform_collect_returns(
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

/* §15.2 — per-fn "does this fn always return a uniform value?" property.
 * It is a PURE per-fn property: return exprs reference only call-site-
 * independent sources (params via the global `param_nonuniform`, module
 * vars, literals/const/override, callee locals recomputed from their
 * inits, and nested user calls resolved through this same table).  So we
 * compute it ONCE by call-graph fixpoint instead of re-walking the callee
 * body at every call-site.  This replaces the old per-call-site walk and,
 * crucially, its `init_depth >= 16 → return 0` cap that FALSE-REJECTED
 * genuinely-uniform deep call chains (e.g. a 20-deep constant-return chain
 * guarding `workgroupBarrier`).
 *
 * Optimistic init (all fns uniform=1); a fn drops to 0 only when some
 * return expr is non-uniform.  `uniform_collect_returns` analyzes with
 * `UD_NONE`, so every diagnostic path in the analysis is guarded off — the
 * fixpoint emits ZERO diagnostics; flag writes are overwritten by the real
 * per-fn pass.  Nested user-calls hit the O(1) lookup below, so there is no
 * deep recursion here. */
uint8_t *compute_fn_returns_uniform(
    WGSLValidator *v, const uint32_t *fn_reqs, const uint8_t *param_nonuniform)
{
    if (!v || !v->res) return NULL;
    size_t n = v->res->all_decl_count;
    uint8_t *ret = (uint8_t *)malloc(n ? n : 1);
    if (!ret) return NULL;
    memset(ret, 1, n);                       /* optimistic: all uniform */
    int changed = 1, iter = 0;
    while (changed && iter < (int)n + 4) {
        changed = 0;
        for (size_t i = 0; i < n; i++) {
            WGSLSymbol *s = v->res->all_decls[i];
            if (!s || s->kind != WGSL_SYM_FUNCTION || !s->ast ||
                s->ast->child_count == 0)
                continue;
            WGSLNode *body = s->ast->children[s->ast->child_count - 1];
            if (!body) continue;
            uint8_t *sk = n ? (uint8_t *)calloc(n, 1) : NULL;
            uint8_t *su = n ? (uint8_t *)calloc(n, 1) : NULL;
            UniformCtx ucx = { v, param_nonuniform, n, 0, su, sk, ret };
            int saw_return = 0, all_uniform = 1;
            uniform_collect_returns(&ucx, body, fn_reqs, &saw_return, &all_uniform);
            uint8_t val = (!saw_return || all_uniform) ? 1u : 0u;
            free(sk);
            free(su);
            if (ret[i] != val) { ret[i] = val; changed = 1; }
        }
        iter++;
    }
    return ret;
}

/* O(1) lookup into the precomputed call-graph bitset.  Absent table
 * (should not happen in the main pass) or unknown fn → conservatively
 * uniform (accept), matching the pre-scan default. */
int user_function_result_is_uniform(
    UniformCtx *cx, WGSLSymbol *fn, const uint32_t *fn_reqs)
{
    (void)fn_reqs;
    if (!cx || !fn) return 1;
    if (!cx->fn_returns_uniform) return 1;
    int idx = wgsl_val_sym_index(cx->v, fn);
    if (idx < 0 || (size_t)idx >= cx->nsyms) return 1;
    return cx->fn_returns_uniform[idx];
}
