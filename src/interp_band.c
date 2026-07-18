/**
 * @file interp_band.c — static interval cost-band analyzer.
 */
#include "internal/interp_priv.h"

/* static interval cost-band analyzer.
 *
 * A seedless, static pass: instead of one concrete gid, it propagates value
 * INTERVALS ([lo,hi]) through the entry body and counts ops per statement,
 * multiplying by an execution FREQUENCY derived from control flow — counted-loop
 * trip-count ranges and data-dependent branch forks.  The result is a
 * best/worst cost band (flops/iops/bytes → arithmetic-intensity band) for
 * data-dependent kernels, computed without running a seed.
 *
 * It is deliberately approximate: it tracks only scalar int/float locals as
 * intervals (aggregates / buffer reads are ⊤ = unknown), recognizes counted
 * `for`/`while` loops, forks on `if`/`switch`, and caps or flags anything it
 * cannot bound (data-dependent `while`, `break`/`continue` effects) — those
 * regions set `imprecise`, and the concrete focus run anchors the band as the
 * sampling fallback.  Shares helpers via internal/interp_priv.h. */

#define BAND_LOOP_CAP  4096.0   /* assumed trips for an unbounded loop         */
#define BAND_GRID      65536.0  /* assumed grid extent for gid-derived values  */

/* intervals. */

/* Ival: interp_priv.h */
   /* top = unbounded/unknown */

Ival iv_top(void)        { Ival v = { 0, 0, 1 }; return v; }
Ival iv_pt(double x)     { Ival v = { x, x, 0 }; return v; }
Ival iv_rng(double a, double b) { Ival v; v.lo = a < b ? a : b; v.hi = a < b ? b : a; v.top = 0; return v; }

Ival iv_add(Ival a, Ival b) { return (a.top || b.top) ? iv_top() : iv_rng(a.lo + b.lo, a.hi + b.hi); }
Ival iv_sub(Ival a, Ival b) { return (a.top || b.top) ? iv_top() : iv_rng(a.lo - b.hi, a.hi - b.lo); }
Ival iv_mul(Ival a, Ival b) {
    if (a.top || b.top) return iv_top();
    double p[4] = { a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi };
    double lo = p[0], hi = p[0];
    for (int i = 1; i < 4; i++) {
        if (p[i] < lo) lo = p[i];
        if (p[i] > hi) hi = p[i];
    }
    return iv_rng(lo, hi);
}
Ival iv_join(Ival a, Ival b) {   /* union (branch merge) */
    if (a.top || b.top) return iv_top();
    return iv_rng(a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi);
}

/* cost vectors / bands. */

/* CV: interp_priv.h */

CV cv0(void) { CV c = { 0, 0, 0, 0 }; return c; }
CV cv_add(CV a, CV b) { return (CV){ a.flops+b.flops, a.iops+b.iops, a.bytes_ld+b.bytes_ld, a.bytes_st+b.bytes_st }; }
CV cv_scale(CV a, double k) { return (CV){ a.flops*k, a.iops*k, a.bytes_ld*k, a.bytes_st*k }; }
double cv_total(CV a) { return a.flops + a.iops + a.bytes_ld + a.bytes_st; }

/* Band: interp_priv.h */

Band band_of(CV c) { Band b; b.min = c; b.max = c; return b; }
Band band_add(Band a, Band b) { Band r; r.min = cv_add(a.min, b.min); r.max = cv_add(a.max, b.max); return r; }

/* env: scalar local -> interval. */

/* IBind: interp_priv.h */

/* IEnv: interp_priv.h */


void env_set(IEnv *e, const WGSLSymbol *s, Ival iv) {
    if (!s) return;
    for (uint32_t i = 0; i < e->n; i++) if (e->v[i].sym == s) { e->v[i].iv = iv; return; }
    if (e->n == e->cap) {
        uint32_t nc = e->cap ? e->cap * 2 : 16;
        IBind *g = (IBind *)realloc(e->v, (size_t)nc * sizeof *g);
        if (!g) return;
        e->v = g; e->cap = nc;
    }
    e->v[e->n].sym = s; e->v[e->n].iv = iv; e->n++;
}
Ival env_get(const IEnv *e, const WGSLSymbol *s) {
    for (uint32_t i = 0; i < e->n; i++) if (e->v[i].sym == s) return e->v[i].iv;
    return iv_top();
}
void env_copy(IEnv *dst, const IEnv *src) {
    dst->n = 0;
    for (uint32_t i = 0; i < src->n; i++) env_set(dst, src->v[i].sym, src->v[i].iv);
}
void env_join(IEnv *dst, const IEnv *other) {   /* merge two branch envs */
    for (uint32_t i = 0; i < other->n; i++) {
        Ival a = env_get(dst, other->v[i].sym);
        env_set(dst, other->v[i].sym, iv_join(a, other->v[i].iv));
    }
}

/* analyzer context. */

/* loops: interp_priv.h */


/* Float vs int result domain of a typed expression. */
int band_is_float(const BandA *A, WGSLNode *n) {
    WGSLTypeInfo *t = wgsl_typecheck_type_of(A->tc, n);
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_REF) t = (WGSLTypeInfo *)t->ref;
    for (;;) {
        switch ((WGSLTypeKind)t->kind) {
        case WGSL_TYPE_F32: case WGSL_TYPE_F16: case WGSL_TYPE_ABSTRACT_FLOAT: return 1;
        case WGSL_TYPE_VEC: case WGSL_TYPE_MAT:
            t = (WGSLTypeInfo *)t->ref; if (!t) return 0; continue;
        default: return 0;
        }
    }
}
/* Component count of a typed expression (vector width → ops fold per lane). */
uint32_t band_width(const BandA *A, WGSLNode *n) {
    WGSLTypeInfo *t = wgsl_typecheck_type_of(A->tc, n);
    if (!t) return 1;
    if (t->kind == WGSL_TYPE_REF) t = (WGSLTypeInfo *)t->ref;
    if (t->kind == WGSL_TYPE_VEC) return t->width ? t->width : 1;
    if (t->kind == WGSL_TYPE_MAT) return (uint32_t)t->width * (t->rows ? t->rows : 1);
    return 1;
}

/* Replicated SFU weight (mirrors consteval builtin_flop_weight, which is static
 * in another TU); keep the two in sync if retuned. */
uint32_t band_flop_weight(const char *nm, uint32_t len) {
    #define EQ(s) (len == sizeof(s)-1 && memcmp(nm, s, len) == 0)
    if (EQ("sqrt")||EQ("inverseSqrt")||EQ("exp")||EQ("exp2")||EQ("log")||EQ("log2")||EQ("sin")||EQ("cos")) return 4;
    if (EQ("tan")||EQ("pow")||EQ("atan2")||EQ("atan")||EQ("asin")||EQ("acos")||EQ("sinh")||EQ("cosh")||
        EQ("tanh")||EQ("asinh")||EQ("acosh")||EQ("atanh")) return 8;
    return 1;
    #undef EQ
}

/* Address space of the root symbol an lvalue/access chain refers to. */
WGSLAddressSpace band_root_as(WGSLNode *n) {
    while (n) {
        if (n->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(n);
            return s ? (WGSLAddressSpace)s->as : WGSL_AS_NONE;
        }
        if (n->kind == WGSL_NODE_EXPR_INDEX || n->kind == WGSL_NODE_EXPR_MEMBER ||
            n->kind == WGSL_NODE_EXPR_PAREN)
            n = n->child_count ? n->children[0] : NULL;
        else return WGSL_AS_NONE;
    }
    return WGSL_AS_NONE;
}

/* interval value of an expression (control-flow-relevant scalars). */

Ival band_ival(BandA *A, WGSLNode *n, const IEnv *env) {
    if (!n) return iv_top();
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT: {
        /* Reuse the const evaluator to parse the literal precisely. */
        WGSLValue v; memset(&v, 0, sizeof v);
        if (wgsl_consteval_expr(A->cev, n, &v)) {
            if (v.kind == WGSL_VAL_INT)   return iv_pt((double)v.u.i);
            if (v.kind == WGSL_VAL_FLOAT) return iv_pt(v.u.f);
        }
        return iv_top();
    }
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (!s) return iv_top();
        /* Only const / override are compile-time-uniform values — resolve them
         * via the binding table (this is what makes override-bounded loops
         * statically bounded).  A local's binding table entry is stale state
         * left by the concrete run, so locals use the analyzer's interval env. */
        if (s->kind == WGSL_SYM_CONST || s->kind == WGSL_SYM_OVERRIDE) {
            const WGSLValue *cv = wgsl_consteval_value_of(A->cev, s);
            if (cv) {
                if (cv->kind == WGSL_VAL_INT)   return iv_pt((double)cv->u.i);
                if (cv->kind == WGSL_VAL_FLOAT) return iv_pt(cv->u.f);
            }
        }
        return env_get(env, s);
    }
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count ? band_ival(A, n->children[0], env) : iv_top();
    case WGSL_NODE_EXPR_UNARY: {
        if (n->child_count < 1) return iv_top();
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        Ival a = band_ival(A, n->children[n->child_count - 1], env);
        if (op == WGSL_TOK_MINUS) return iv_sub(iv_pt(0), a);
        return iv_top();
    }
    case WGSL_NODE_EXPR_BINARY: {
        if (n->child_count < 2) return iv_top();
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        Ival a = band_ival(A, n->children[0], env);
        Ival b = band_ival(A, n->children[1], env);
        switch (op) {
        case WGSL_TOK_PLUS:  return iv_add(a, b);
        case WGSL_TOK_MINUS: return iv_sub(a, b);
        case WGSL_TOK_STAR:  return iv_mul(a, b);
        default:             return iv_top();
        }
    }
    default:
        return iv_top();   /* index / member / call / builtin → unknown */
    }
}

/* Comparison determinacy for a branch/loop condition: +1 always-true,
 * -1 always-false, 0 unknown (fork). */
int band_cond(BandA *A, WGSLNode *cond, const IEnv *env) {
    if (!cond || cond->kind != WGSL_NODE_EXPR_BINARY || cond->child_count < 2) return 0;
    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(cond);
    Ival a = band_ival(A, cond->children[0], env);
    Ival b = band_ival(A, cond->children[1], env);
    if (a.top || b.top) return 0;
    switch (op) {
    case WGSL_TOK_LESS:          return a.hi <  b.lo ? 1 : (a.lo >= b.hi ? -1 : 0);
    case WGSL_TOK_LESS_EQUAL:    return a.hi <= b.lo ? 1 : (a.lo >  b.hi ? -1 : 0);
    case WGSL_TOK_GREATER:       return a.lo >  b.hi ? 1 : (a.hi <= b.lo ? -1 : 0);
    case WGSL_TOK_GREATER_EQUAL: return a.lo >= b.hi ? 1 : (a.hi <  b.lo ? -1 : 0);
    default:                     return 0;
    }
}

/* structural op count for one execution of an expression. */

CV band_count_ops(BandA *A, WGSLNode *n) {
    if (!n) return cv0();
    CV c = cv0();
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_BINARY: {
        for (uint32_t i = 0; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        double w = band_width(A, n->children[0] ? n->children[0] : n);
        int is_cmp = (op == WGSL_TOK_EQUAL_EQUAL || op == WGSL_TOK_BANG_EQUAL ||
                      op == WGSL_TOK_LESS || op == WGSL_TOK_LESS_EQUAL ||
                      op == WGSL_TOK_GREATER || op == WGSL_TOK_GREATER_EQUAL);
        int fl = is_cmp ? band_is_float(A, n->children[0]) : band_is_float(A, n);
        /* logical && || short-circuit: no arithmetic op */
        if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) break;
        if (fl) c.flops += w; else c.iops += w;
        break;
    }
    case WGSL_NODE_EXPR_UNARY: {
        for (uint32_t i = 0; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
        if (op == WGSL_TOK_MINUS || op == WGSL_TOK_BANG || op == WGSL_TOK_TILDE) {
            double w = band_width(A, n);
            if (band_is_float(A, n)) c.flops += w; else c.iops += w;
        }
        break;
    }
    case WGSL_NODE_EXPR_CALL: {
        /* args + builtin flop weight (transcendental/SFU) */
        WGSLNode *callee = n->child_count ? n->children[0] : NULL;
        for (uint32_t i = 1; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        if (callee) {
            WGSLNode *nm = callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT && callee->child_count
                           ? callee->children[0] : callee;
            uint32_t off = wgsl_node_name_span(nm).offset;
            uint32_t len = wgsl_node_name_span(nm).length;
            if (len && (size_t)off + len <= A->src->length && band_is_float(A, n)) {
                uint32_t wt = band_flop_weight(A->src->bytes + off, len);
                if (wt > 1) c.flops += (double)wt * band_width(A, n);   /* SFU */
            }
        }
        break;
    }
    case WGSL_NODE_EXPR_INDEX: {
        for (uint32_t i = 0; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        WGSLAddressSpace as = band_root_as(n->children[0]);
        if (as == WGSL_AS_STORAGE || as == WGSL_AS_UNIFORM) {
            WGSLTypeInfo *t = wgsl_typecheck_type_of(A->tc, n);
            if (t && t->kind == WGSL_TYPE_REF) t = (WGSLTypeInfo *)t->ref;
            c.bytes_ld += interp_leaf_bytes(t);
        }
        break;
    }
    case WGSL_NODE_EXPR_MEMBER:
    case WGSL_NODE_EXPR_PAREN:
        for (uint32_t i = 0; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        break;
    default:
        for (uint32_t i = 0; i < n->child_count; i++) c = cv_add(c, band_count_ops(A, n->children[i]));
        break;
    }
    return c;
}

/* DRAM store bytes for an assignment target rooted in storage. */
CV band_store_cost(BandA *A, WGSLNode *lhs) {
    CV c = cv0();
    WGSLAddressSpace as = band_root_as(lhs);
    if (as == WGSL_AS_STORAGE || as == WGSL_AS_UNIFORM) {
        WGSLTypeInfo *t = wgsl_typecheck_type_of(A->tc, lhs);
        if (t && t->kind == WGSL_TYPE_REF) t = (WGSLTypeInfo *)t->ref;
        c.bytes_st += interp_leaf_bytes(t);
    }
    return c;
}

/* statement band (control flow -> cost band). */

Band band_stmt(BandA *A, WGSLNode *n, IEnv *env);

Band band_block(BandA *A, WGSLNode *b, IEnv *env) {
    Band acc = band_of(cv0());
    if (b) for (uint32_t i = 0; i < b->child_count; i++)
        acc = band_add(acc, band_stmt(A, b->children[i], env));
    return acc;
}

/* Trip-count range for a counted `for`.  Recognizes `i {<,<=} N` with an
 * increasing induction var; returns 0 range + not-bounded if unrecognized. */
void band_trip(BandA *A, WGSLNode *cond, WGSLNode *update,
                      const IEnv *env, WGSLSymbol *ivar, Ival i0,
                      double *tmin, double *tmax, int *bounded) {
    *tmin = 0; *tmax = BAND_LOOP_CAP; *bounded = 0;
    if (!cond || cond->kind != WGSL_NODE_EXPR_BINARY || cond->child_count < 2 || !ivar) return;
    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(cond);
    if (op != WGSL_TOK_LESS && op != WGSL_TOK_LESS_EQUAL) return;
    if (wgsl_node_resolved_symbol(cond->children[0]) != ivar) return;
    Ival N = band_ival(A, cond->children[1], env);
    /* step: from `i += S` / `i++`; default 1 */
    double step = 1.0;
    if (update && update->kind == WGSL_NODE_STMT_COMPOUND_ASSIGN &&
        (WGSLTokenKind)wgsl_node_op_kind_u32(update) == WGSL_TOK_PLUS_EQUAL && update->child_count >= 2) {
        Ival s = band_ival(A, update->children[1], env);
        if (!s.top && s.lo == s.hi && s.lo > 0) step = s.lo;
    }
    if (N.top || i0.top || step <= 0) return;
    double adj = (op == WGSL_TOK_LESS_EQUAL) ? 1.0 : 0.0;
    double lo = ceil((N.lo + adj - i0.hi) / step);
    double hi = ceil((N.hi + adj - i0.lo) / step);
    if (lo < 0) lo = 0;
    if (hi < 0) hi = 0;
    if (hi > BAND_LOOP_CAP) { hi = BAND_LOOP_CAP; *tmin = lo; *tmax = hi; *bounded = 0; return; }
    *tmin = lo; *tmax = hi; *bounded = 1;
}

void band_note_loop(BandA *A, uint32_t line, double tmin, double tmax, int bounded) {
    if (!bounded) A->imprecise = 1;
    if (A->nloops < (int)(sizeof A->loops / sizeof A->loops[0])) {
        A->loops[A->nloops].line = line;
        A->loops[A->nloops].tmin = tmin;
        A->loops[A->nloops].tmax = tmax;
        A->loops[A->nloops].bounded = bounded;
        A->nloops++;
    }
}

Band band_stmt(BandA *A, WGSLNode *n, IEnv *env) {
    if (!n) return band_of(cv0());
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_COMPOUND:
        return band_block(A, n, env);

    case WGSL_NODE_DECL_VAR:
    case WGSL_NODE_DECL_LET: {
        WGSLSymbol *s = decl_symbol(A->cev, n);
        WGSLNode *init = n->child_count ? n->children[n->child_count - 1] : NULL;
        CV c = init ? band_count_ops(A, init) : cv0();
        if (s) env_set(env, s, init ? band_ival(A, init, env) : iv_top());
        return band_of(c);
    }
    case WGSL_NODE_STMT_ASSIGN: {
        WGSLNode *lhs = n->children[0], *rhs = n->children[1];
        CV c = cv_add(band_count_ops(A, rhs), band_store_cost(A, lhs));
        /* index expression on the lhs still costs its address arithmetic */
        if (lhs->kind == WGSL_NODE_EXPR_INDEX && lhs->child_count >= 2)
            c = cv_add(c, band_count_ops(A, lhs->children[1]));
        if (lhs->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(lhs);
            env_set(env, s, band_ival(A, rhs, env));
        }
        return band_of(c);
    }
    case WGSL_NODE_STMT_COMPOUND_ASSIGN: {
        WGSLNode *lhs = n->children[0], *rhs = n->children[1];
        CV c = cv_add(band_count_ops(A, lhs), band_count_ops(A, rhs));
        double w = band_width(A, lhs);
        if (band_is_float(A, lhs)) c.flops += w; else c.iops += w;
        c = cv_add(c, band_store_cost(A, lhs));
        if (lhs->kind == WGSL_NODE_EXPR_IDENT)
            env_set(env, wgsl_node_resolved_symbol(lhs), iv_top());
        return band_of(c);
    }
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT: {
        WGSLNode *lhs = n->child_count ? n->children[0] : NULL;
        CV c = cv0(); c.iops += 1;
        c = cv_add(c, band_store_cost(A, lhs));
        if (lhs && lhs->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(lhs);
            Ival cur = env_get(env, s);
            env_set(env, s, iv_add(cur, iv_pt(n->kind == WGSL_NODE_STMT_INCREMENT ? 1 : -1)));
        }
        return band_of(c);
    }
    case WGSL_NODE_STMT_IF: {
        /* Exactly ONE arm runs (then / an else-if / else / nothing), so the
         * band forks over the arms (cheapest..costliest), it does not sum them. */
        CV cc = band_count_ops(A, n->children[0]);
        int det = band_cond(A, n->children[0], env);
        IEnv post; memset(&post, 0, sizeof post); env_copy(&post, env);
        int have = 0; CV mn = cv0(), mx = cv0();
        #define FORK_ARM(b) do {                                                \
            CV _n = (b).min;                                                     \
            CV _x = (b).max;                                                     \
            if (!have || cv_total(_n) < cv_total(mn)) mn = _n;                   \
            if (!have || cv_total(_x) > cv_total(mx)) mx = _x;                   \
            have = 1;                                                            \
        } while (0)
        if (det >= 0) {                                     /* then arm possible */
            IEnv te; memset(&te, 0, sizeof te); env_copy(&te, env);
            Band tb = band_stmt(A, n->children[1], &te);
            FORK_ARM(tb);
            env_join(&post, &te);
            free(te.v);
        }
        if (det <= 0) {                                     /* else side possible */
            int has_final_else = 0;
            for (uint32_t i = 2; i < n->child_count; i++) {
                WGSLNode *cl = n->children[i];
                if (!cl) continue;
                IEnv ce; memset(&ce, 0, sizeof ce); env_copy(&ce, env);
                if (cl->kind == WGSL_NODE_STMT_ELSE_IF) {
                    cc = cv_add(cc, band_count_ops(A, cl->child_count ? cl->children[0] : NULL));
                    Band cb = band_stmt(A, cl->child_count >= 2 ? cl->children[1] : NULL, &ce);
                    FORK_ARM(cb);
                    env_join(&post, &ce);
                } else if (cl->kind == WGSL_NODE_STMT_ELSE) {
                    has_final_else = 1;
                    Band cb = band_stmt(A, cl->child_count ? cl->children[cl->child_count-1] : NULL, &ce);
                    FORK_ARM(cb);
                    env_join(&post, &ce);
                }
                free(ce.v);
            }
            if (!has_final_else) {
                FORK_ARM(band_of(cv0()));  /* fall-through: 0 cost */
            }
        }
        #undef FORK_ARM
        if (det == 0) A->ddbranches++;
        env_copy(env, &post); free(post.v);
        Band r; r.min = cv_add(cc, mn); r.max = cv_add(cc, mx);
        return r;
    }
    case WGSL_NODE_STMT_FOR: {
        uint32_t flags = wgsl_for_flags(n);
        uint32_t idx = 0;
        WGSLNode *init = (flags & 1u) && idx < n->child_count ? n->children[idx++] : NULL;
        WGSLNode *cond = (flags & 2u) && idx < n->child_count ? n->children[idx++] : NULL;
        WGSLNode *upd  = (flags & 4u) && idx < n->child_count ? n->children[idx++] : NULL;
        WGSLNode *body = idx < n->child_count ? n->children[idx] : NULL;
        Band ib = init ? band_stmt(A, init, env) : band_of(cv0());
        WGSLSymbol *ivar = (init && (init->kind == WGSL_NODE_DECL_VAR || init->kind == WGSL_NODE_DECL_LET))
                           ? decl_symbol(A->cev, init) : NULL;
        Ival i0 = ivar ? env_get(env, ivar) : iv_top();
        double tmin, tmax; int bounded;
        band_trip(A, cond, upd, env, ivar, i0, &tmin, &tmax, &bounded);
        band_note_loop(A, line_of(A->src, n), tmin, tmax, bounded);
        /* Analyze the body once with i spanning [i0.lo, N-1] (best effort). */
        if (ivar && cond && cond->child_count >= 2) {
            Ival N = band_ival(A, cond->children[1], env);
            if (!N.top && !i0.top) env_set(env, ivar, iv_rng(i0.lo, N.hi));
        }
        CV per = cv_add(band_count_ops(A, cond), band_count_ops(A, upd));
        Band bb = band_stmt(A, body, env);
        CV iter_lo = cv_add(bb.min, per);
        CV iter_hi = cv_add(bb.max, per);
        Band r;
        r.min = cv_add(ib.min, cv_scale(iter_lo, tmin));
        r.max = cv_add(ib.max, cv_add(cv_scale(iter_hi, tmax), band_count_ops(A, cond)));  /* final cond check */
        return r;
    }
    case WGSL_NODE_STMT_WHILE: {
        WGSLNode *cond = n->child_count > 0 ? n->children[0] : NULL;
        WGSLNode *body = n->child_count > 1 ? n->children[1] : NULL;
        /* No induction var recognized → conservative [0, cap]; flag imprecise. */
        band_note_loop(A, line_of(A->src, n), 0, BAND_LOOP_CAP, 0);
        CV cc = band_count_ops(A, cond);
        Band bb = band_stmt(A, body, env);
        Band r;
        r.min = cc;                                    /* condition false immediately */
        r.max = cv_add(cv_scale(cv_add(bb.max, cc), BAND_LOOP_CAP), cc);
        return r;
    }
    case WGSL_NODE_STMT_LOOP: {
        band_note_loop(A, line_of(A->src, n), 0, BAND_LOOP_CAP, 0);
        uint32_t nbody = n->child_count;
        Band body = band_of(cv0());
        for (uint32_t i = 0; i < nbody; i++)
            body = band_add(body, band_stmt(A, n->children[i], env));
        Band r; r.min = cv0(); r.max = cv_scale(body.max, BAND_LOOP_CAP);
        return r;
    }
    case WGSL_NODE_STMT_SWITCH: {
        if (n->child_count == 0) return band_of(cv0());
        CV sc = band_count_ops(A, n->children[0]);
        Band lo = band_of(cv0()), hi = band_of(cv0());
        int first = 1;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *cl = n->children[i];
            if (!cl || cl->kind != WGSL_NODE_STMT_CASE_CLAUSE || cl->child_count == 0) continue;
            IEnv ce; memset(&ce, 0, sizeof ce); env_copy(&ce, env);
            Band cb = band_stmt(A, cl->children[cl->child_count - 1], &ce);
            if (first) { lo = cb; hi = cb; first = 0; }
            else {
                if (cv_total(cb.min) < cv_total(lo.min)) lo.min = cb.min;
                if (cv_total(cb.max) > cv_total(hi.max)) hi.max = cb.max;
            }
            free(ce.v);
        }
        if (!first) A->ddbranches++;
        Band r; r.min = cv_add(sc, lo.min); r.max = cv_add(sc, hi.max);
        return r;
    }
    case WGSL_NODE_STMT_RETURN:
        return band_of(n->child_count ? band_count_ops(A, n->children[0]) : cv0());
    case WGSL_NODE_STMT_FN_CALL:
        return band_of(n->child_count ? band_count_ops(A, n->children[0]) : cv0());

    /* break/continue affect trip counts; not modelled precisely (flag). */
    case WGSL_NODE_STMT_BREAK:
    case WGSL_NODE_STMT_CONTINUE:
    case WGSL_NODE_STMT_BREAK_IF:
        A->imprecise = 1;
        return band_of(cv0());
    default:
        return band_of(cv0());
    }
}
