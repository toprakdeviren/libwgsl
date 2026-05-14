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

