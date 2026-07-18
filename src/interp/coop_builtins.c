/**
 * @file coop_builtins.c — subgroup/quad cooperative ops + atomics/barrier builtins
 */
#include "internal/interp_priv.h"


/* cooperative ops (subgroup* / quad*).
 *
 * These are the whole reason for lockstep execution: instead of folding to 0,
 * they reduce / shuffle / broadcast across the lanes actually converged at this
 * instruction (`ip->act`), intersected with the caller's subgroup.  Because the
 * min-PC scheduler converges all lanes that reach a given pc, `ip->act` at a
 * cooperative op IS the active subgroup mask, so the result is exact. */

/* Combine a OP b for a reduction/scan; component-wise min/max for Min/Max. */
int coop_minmax(Interp *ip, const WGSLValue *a, const WGSLValue *b,
                       int is_max, WGSLValue *out) {
    if (a->kind == WGSL_VAL_INVALID) { *out = *b; return b->kind != WGSL_VAL_INVALID; }
    if (b->kind == WGSL_VAL_INVALID) { *out = *a; return 1; }
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        int is_u = a->type && a->type->kind == WGSL_TYPE_U32;
        int64_t x = a->u.i, y = b->u.i;
        *out = *a;
        out->u.i = is_u ? (is_max ? ((uint64_t)x >= (uint64_t)y ? x : y)
                                  : ((uint64_t)x <= (uint64_t)y ? x : y))
                        : (is_max ? (x >= y ? x : y) : (x <= y ? x : y));
        return 1;
    }
    if (a->kind == WGSL_VAL_FLOAT && b->kind == WGSL_VAL_FLOAT) {
        double x = a->u.f, y = b->u.f;
        *out = *a; out->u.f = is_max ? (x >= y ? x : y) : (x <= y ? x : y);
        return 1;
    }
    if (a->kind == WGSL_VAL_VEC && b->kind == WGSL_VAL_VEC &&
        a->u.agg.count == b->u.agg.count) {
        uint32_t nn = a->u.agg.count;
        WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(ip->cev->arena,
                                                     sizeof(WGSLValue) * (nn ? nn : 1));
        if (!e) { *out = *a; return 1; }
        for (uint32_t i = 0; i < nn; i++)
            coop_minmax(ip, &a->u.agg.elems[i], &b->u.agg.elems[i], is_max, &e[i]);
        *out = *a; out->u.agg.elems = e;
        return 1;
    }
    *out = *a; return 1;
}

/* Gather the operand `arg` from every active lane in the caller's subgroup,
 * into `lanes_out`/`vals_out` (ascending lane id = subgroup-invocation order).
 * Returns the count.  Saves/restores the current lane around the cross-lane
 * evaluations; copies `ip->act` first since a nested call can overwrite it. */
uint32_t coop_gather(Interp *ip, WGSLNode *arg, uint32_t sgw,
                            uint32_t *lanes_out, WGSLValue *vals_out) {
    uint32_t self = ip->cev->simt.cur_lane, self_sg = self / sgw;
    uint32_t na = ip->nact; if (na > INTERP_MAX_LANES) na = INTERP_MAX_LANES;
    uint32_t local[INTERP_MAX_LANES];
    for (uint32_t k = 0; k < na; k++) local[k] = ip->act[k];
    uint32_t saved = ip->cev->simt.cur_lane, count = 0;
    int saved_gather = ip->in_gather;
    ip->in_gather = 1;   /* peer-operand re-evals aren't new warp memory accesses */
    for (uint32_t k = 0; k < na; k++) {
        uint32_t lj = local[k];
        if (lj / sgw != self_sg) continue;
        ip->cev->simt.cur_lane = lj;
        WGSLValue v; memset(&v, 0, sizeof v);
        if (!arg || !eval(ip, arg, &v)) memset(&v, 0, sizeof v);
        lanes_out[count] = lj; vals_out[count] = v; count++;
    }
    ip->cev->simt.cur_lane = saved;
    ip->in_gather = saved_gather;
    return count;
}

void coop_bool(Interp *ip, WGSLValue *out, int b) {
    memset(out, 0, sizeof *out);
    out->kind = WGSL_VAL_BOOL;
    out->type = ip->cev->types ? ip->cev->types->t_bool : NULL;
    out->u.b = b ? true : false;
}

/* A cooperative op / barrier inside a helper function is run one lane at a time
 * (helpers use a single-lane nested VM), so it cannot see the workgroup — its
 * result is not modelled.  Flag it once so the trace reads "partial" instead of
 * silently mis-reducing.  No-op for the entry body (call_depth == 0) or N == 1. */
void warn_helper_coop(Interp *ip, const WGSLNode *n) {
    if (ip->call_depth <= 0 || ip->nlanes <= 1 || ip->helper_coop_warned) return;
    ip->helper_coop_warned = 1;
    wgsl_diag_emit_at(ip->cev->diag, ip->src, WGSL_DIAG_WARNING,
                      n ? n->span_offset : 0, n ? n->span_length : 0, NULL,
                      "cooperative op / barrier inside a called function is "
                      "modelled single-lane, not across the workgroup");
}

/* A subgroup or quad call.  Returns 1 handled, 0 not a cooperative op / not yet
 * modelled (caller then emits the generic unsupported-builtin warning). */
int coop_op(Interp *ip, WGSLNode *n, const char *nm, uint32_t len, WGSLValue *out) {
    #define CIS(s) (len == sizeof(s) - 1 && memcmp(nm, s, len) == 0)
    int is_quad = (len >= 4 && memcmp(nm, "quad", 4) == 0);
    int is_sub  = (len >= 8 && memcmp(nm, "subgroup", 8) == 0);
    if (!is_quad && !is_sub) return 0;
    warn_helper_coop(ip, n);   /* single-lane in a helper → flag as partial */
    uint32_t sgw = is_quad ? 4u : INTERP_SUBGROUP_WIDTH;

    /* Heap, not static/stack: a nested cooperative op in the argument
     * (`subgroupAdd(subgroupMax(x))`) re-enters this function, and a large
     * stack array would risk overflow under deep nesting. */
    uint32_t  *lanes = (uint32_t *)malloc(INTERP_MAX_LANES * sizeof *lanes);
    WGSLValue *vals  = (WGSLValue *)malloc(INTERP_MAX_LANES * sizeof *vals);
    if (!lanes || !vals) { free(lanes); free(vals); memset(out, 0, sizeof *out); return 1; }

    WGSLNode *arg = n->child_count >= 2 ? n->children[1] : NULL;
    uint32_t cnt = coop_gather(ip, arg, sgw, lanes, vals);
    uint32_t self = ip->cev->simt.cur_lane, sgbase = (self / sgw) * sgw;
    uint32_t self_pos = 0; int found = 0;
    for (uint32_t k = 0; k < cnt; k++) if (lanes[k] == self) { self_pos = k; found = 1; break; }

    int ret = 1;   /* handled unless we fall through to the unmodelled default */
    memset(out, 0, sizeof *out);

    /* reductions & scans (Add/Mul/And/Or/Xor + Min/Max). */
    WGSLTokenKind fold = WGSL_TOK_PLUS; int use_fold = 0, is_minmax = 0, is_max = 0;
    if      (CIS("subgroupAdd") || CIS("subgroupInclusiveAdd") || CIS("subgroupExclusiveAdd")) { fold = WGSL_TOK_PLUS; use_fold = 1; }
    else if (CIS("subgroupMul") || CIS("subgroupInclusiveMul") || CIS("subgroupExclusiveMul")) { fold = WGSL_TOK_STAR; use_fold = 1; }
    else if (CIS("subgroupAnd")) { fold = WGSL_TOK_AMP;   use_fold = 1; }
    else if (CIS("subgroupOr"))  { fold = WGSL_TOK_PIPE;  use_fold = 1; }
    else if (CIS("subgroupXor")) { fold = WGSL_TOK_CARET; use_fold = 1; }
    else if (CIS("subgroupMin")) { is_minmax = 1; is_max = 0; }
    else if (CIS("subgroupMax")) { is_minmax = 1; is_max = 1; }

    if (use_fold || is_minmax) {
        int inclusive = (len >= 17 && memcmp(nm, "subgroupInclusive", 17) == 0);
        int exclusive = (len >= 17 && memcmp(nm, "subgroupExclusive", 17) == 0);
        uint32_t lo = 0, hi = cnt;
        if (inclusive) hi = found ? self_pos + 1 : 0;
        else if (exclusive) hi = found ? self_pos : 0;
        if (cnt == 0 || hi <= lo) {              /* empty range → identity */
            double idv = (fold == WGSL_TOK_STAR && use_fold) ? 1.0 : 0.0;
            if (found && cnt) *out = vals[self_pos];
            if (out->kind == WGSL_VAL_INT)   out->u.i = (int64_t)idv;
            else if (out->kind == WGSL_VAL_FLOAT) out->u.f = idv;
            else if (out->kind == WGSL_VAL_VEC && found && cnt) {
                /* zero/one vector of the operand's shape, not INVALID */
                uint32_t nn = out->u.agg.count;
                WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(ip->cev->arena,
                                                            sizeof(WGSLValue) * (nn ? nn : 1));
                if (e) {
                    for (uint32_t q = 0; q < nn; q++) {
                        e[q] = out->u.agg.elems[q];
                        if (e[q].kind == WGSL_VAL_INT)   e[q].u.i = (int64_t)idv;
                        else if (e[q].kind == WGSL_VAL_FLOAT) e[q].u.f = idv;
                    }
                    out->u.agg.elems = e;
                } else memset(out, 0, sizeof *out);
            }
            else memset(out, 0, sizeof *out);
        } else {
            WGSLValue acc = vals[lo];
            for (uint32_t k = lo + 1; k < hi; k++) {
                WGSLValue r;
                if (use_fold) { if (!wgsl_consteval_binop(ip->cev, n, fold, &acc, &vals[k], &r)) r = acc; }
                else coop_minmax(ip, &acc, &vals[k], is_max, &r);
                acc = r;
            }
            *out = acc;
        }
    }
    /* broadcast / shuffle. */
    else if (CIS("subgroupBroadcastFirst")) { if (cnt) *out = vals[0]; }
    else if (CIS("subgroupElect")) { coop_bool(ip, out, cnt > 0 && lanes[0] == self); }
    else if (CIS("subgroupBroadcast") || CIS("subgroupShuffle") || CIS("quadBroadcast")) {
        int64_t id = 0;
        if (n->child_count >= 3) {
            WGSLValue t;
            if (eval(ip, n->children[2], &t) && t.kind == WGSL_VAL_INT) id = t.u.i;
        }
        uint32_t want = sgbase + (uint32_t)id;
        *out = found ? vals[self_pos] : *out;
        for (uint32_t k = 0; k < cnt; k++) if (lanes[k] == want) { *out = vals[k]; break; }
    }
    else if (CIS("subgroupShuffleXor") || CIS("subgroupShuffleUp") || CIS("subgroupShuffleDown")) {
        int64_t d = 0;
        if (n->child_count >= 3) {
            WGSLValue t;
            if (eval(ip, n->children[2], &t) && t.kind == WGSL_VAL_INT) d = t.u.i;
        }
        int64_t sid = (int64_t)(self - sgbase), tid;
        if      (CIS("subgroupShuffleXor")) tid = sid ^ d;
        else if (CIS("subgroupShuffleUp"))  tid = sid - d;
        else                                tid = sid + d;
        uint32_t want = sgbase + (uint32_t)tid;
        *out = found ? vals[self_pos] : *out;
        for (uint32_t k = 0; k < cnt; k++) if (lanes[k] == want) { *out = vals[k]; break; }
    }
    else if (CIS("quadSwapX") || CIS("quadSwapY") || CIS("quadSwapDiagonal")) {
        uint32_t qbase = (self / 4) * 4, qid = self - qbase, tid;
        tid = CIS("quadSwapX") ? (qid ^ 1u) : CIS("quadSwapY") ? (qid ^ 2u) : (qid ^ 3u);
        uint32_t want = qbase + tid;
        *out = found ? vals[self_pos] : *out;
        for (uint32_t k = 0; k < cnt; k++) if (lanes[k] == want) { *out = vals[k]; break; }
    }
    /* predicates / ballot. */
    else if (CIS("subgroupAny") || CIS("subgroupAll")) {
        int any = 0, all = 1;
        for (uint32_t k = 0; k < cnt; k++) { int t = is_true(&vals[k]); any |= t; all &= t; }
        coop_bool(ip, out, CIS("subgroupAny") ? any : all);
    }
    else if (CIS("subgroupBallot")) {
        uint32_t bits[4] = { 0, 0, 0, 0 };
        for (uint32_t k = 0; k < cnt; k++) {
            if (!is_true(&vals[k])) continue;
            uint32_t sid = lanes[k] - sgbase;
            if (sid < 128) bits[sid >> 5] |= (1u << (sid & 31));
        }
        WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(ip->cev->arena, sizeof(WGSLValue) * 4);
        if (e) {
            for (int i = 0; i < 4; i++) {
                memset(&e[i], 0, sizeof e[i]);
                e[i].kind = WGSL_VAL_INT;
                e[i].type = ip->cev->types ? ip->cev->types->t_u32 : NULL;
                e[i].u.i  = bits[i];
            }
            out->kind = WGSL_VAL_VEC; out->u.agg.count = 4; out->u.agg.elems = e;
        }
    }
    else {
        ret = 0;   /* unmodelled subgroup/quad op → generic warning */
    }

    free(lanes); free(vals);
    return ret;
    #undef CIS
}

/* Interpreter builtin hook: atomics / arrayLength / workgroupUniformLoad —
 * predeclared functions the const-folder can't handle because they take a
 * mutable memory location (`&ptr`).  Single-lane, so an atomic RMW is just a
 * load-modify-store on the shared slot (exact at N=1).  Returns 1 handled,
 * -1 handled-with-error, 0 not handled (fall through to the generic error). */
int builtin_call_cb(WGSLConstEvaluator *cev, WGSLNode *n,
                           const char *nm, uint32_t len, WGSLValue *out, void *ud) {
    Interp *ip = (Interp *)ud;
    #define NAME_IS(s) (len == sizeof(s) - 1 && memcmp(nm, s, len) == 0)

    /* Control barriers are real join points, but the join is provided for free
     * by the min-PC warp scheduler: no lane can pass the barrier instruction
     * while another lane is still behind it, so all lanes reconverge here.  The
     * memory-fence half is a no-op in this sequential model.  Handle them as
     * void so they no longer badge the trace "partial". */
    if (NAME_IS("workgroupBarrier") || NAME_IS("storageBarrier") ||
        NAME_IS("textureBarrier")   || NAME_IS("subgroupBarrier")) {
        warn_helper_coop(ip, n);   /* a barrier in a helper can't join the workgroup */
        ip->barrier_pending = 1;   /* advance the race epoch after this step */
        /* Mark the barrier in the focus-lane timeline (epoch boundary). */
        record_step(ip, line_of(ip->src, n), "barrier", "barrier", NULL, NULL);
        memset(out, 0, sizeof *out);
        return 1;
    }

    /* Cooperative ops: real cross-lane reduce/shuffle/broadcast/ballot. */
    { int r = coop_op(ip, n, nm, len, out); if (r) return r; }

    /* First arg is `&expr` or a bare pointer value.  Resolve through
     * WGSL_VAL_PTR + mem_load/mem_store so shared atomic/workgroup stores
     * never bypass race_check (the silent hole the value model closes). */
    WGSLNode *parg = n->child_count >= 2 ? n->children[1] : NULL;
    uint32_t pline = line_of(ip->src, n);

    /* Eval pointer arg → WGSL_VAL_PTR (ADDR_OF builds it; let-bound ptr loads it). */
    WGSLValue pval;
    memset(&pval, 0, sizeof pval);
    int have_ptr = 0;
    if (parg) {
        if (parg->kind == WGSL_NODE_EXPR_ADDR_OF && parg->child_count)
            have_ptr = ptr_from_lvalue(ip, parg->children[0], &pval);
        else
            have_ptr = eval(ip, parg, &pval) && pval.kind == WGSL_VAL_PTR;
    }

    if (NAME_IS("arrayLength")) {
        WGSLValue *slot = have_ptr ? mem_slot(ip, &pval) : NULL;
        memset(out, 0, sizeof *out);
        out->kind = WGSL_VAL_INT;
        out->type = cev->types ? cev->types->t_u32 : NULL;
        out->u.i  = (slot && (slot->kind == WGSL_VAL_ARRAY || slot->kind == WGSL_VAL_VEC))
                    ? (int64_t)slot->u.agg.count : 0;
        return 1;
    }
    if (NAME_IS("workgroupUniformLoad")) {
        if (!have_ptr || !mem_load(ip, &pval, out, pline)) return 0;
        return 1;
    }
    if (len >= 6 && memcmp(nm, "atomic", 6) == 0) {
        if (!have_ptr) return 0;
        WGSLValue oldv;
        if (!mem_load(ip, &pval, &oldv, pline) || oldv.kind != WGSL_VAL_INT) return 0;
        int64_t old = oldv.u.i;
        int is_u = oldv.type && oldv.type->kind == WGSL_TYPE_U32;

        if (NAME_IS("atomicLoad")) { *out = oldv; return 1; }

        WGSLValue v;
        int have = n->child_count >= 3 && eval(ip, n->children[2], &v) &&
                   v.kind == WGSL_VAL_INT;

        if (NAME_IS("atomicStore")) {
            if (!have) return -1;
            if (!mem_store(ip, &pval, &v, pline)) return -1;
            memset(out, 0, sizeof *out);            /* void */
            return 1;
        }

        /* §17.8 atomicCompareExchangeWeak(ptr, cmp, v) →
         *   __atomic_compare_exchange_result<T> { old_value, exchanged }
         * Strong semantics on this sequential SIMT model (no spurious fail). */
        if (NAME_IS("atomicCompareExchangeWeak")) {
            WGSLValue cmpv, newv;
            int have_cmp = n->child_count >= 3 && eval(ip, n->children[2], &cmpv) &&
                           cmpv.kind == WGSL_VAL_INT;
            int have_new = n->child_count >= 4 && eval(ip, n->children[3], &newv) &&
                           newv.kind == WGSL_VAL_INT;
            if (!have_cmp || !have_new) return -1;
            int exchanged;
            if (is_u)
                exchanged = ((uint32_t)old == (uint32_t)cmpv.u.i);
            else
                exchanged = ((int32_t)old == (int32_t)cmpv.u.i);
            if (exchanged) {
                if (!mem_store(ip, &pval, &newv, pline)) return -1;
            }
            cev->cost.iop_count += 1;
            WGSLValue *fields = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, 2 * sizeof *fields);
            if (!fields) return -1;
            WGSLTypeInfo *T = oldv.type;
            memset(fields, 0, 2 * sizeof *fields);
            fields[0].kind = WGSL_VAL_INT;
            fields[0].type = T;
            fields[0].u.i  = old;                    /* old_value */
            fields[1].kind = WGSL_VAL_BOOL;
            fields[1].type = cev->types ? cev->types->t_bool : NULL;
            fields[1].u.b  = exchanged ? 1 : 0;      /* exchanged */
            memset(out, 0, sizeof *out);
            out->kind = WGSL_VAL_STRUCT;
            out->type = cev->types
                ? wgsl_type_atomic_cx_result(cev->types, T) : NULL;
            out->u.agg.count = 2;
            out->u.agg.elems = fields;
            return 1;
        }

        int64_t nv;
        if      (NAME_IS("atomicExchange")) { if (!have) return -1; nv = v.u.i; }
        else if (NAME_IS("atomicAdd"))      { if (!have) return -1; nv = old + v.u.i; }
        else if (NAME_IS("atomicSub"))      { if (!have) return -1; nv = old - v.u.i; }
        else if (NAME_IS("atomicAnd"))      { if (!have) return -1; nv = old & v.u.i; }
        else if (NAME_IS("atomicOr"))       { if (!have) return -1; nv = old | v.u.i; }
        else if (NAME_IS("atomicXor"))      { if (!have) return -1; nv = old ^ v.u.i; }
        else if (NAME_IS("atomicMax"))      { if (!have) return -1;
            nv = is_u ? ((uint32_t)old >= (uint32_t)v.u.i ? old : v.u.i)
                      : (( int32_t)old >= ( int32_t)v.u.i ? old : v.u.i); }
        else if (NAME_IS("atomicMin"))      { if (!have) return -1;
            nv = is_u ? ((uint32_t)old <= (uint32_t)v.u.i ? old : v.u.i)
                      : (( int32_t)old <= ( int32_t)v.u.i ? old : v.u.i); }
        else return 0;   /* unknown atomic* */

        WGSLValue newv = oldv;
        newv.u.i = nv;
        if (!mem_store(ip, &pval, &newv, pline)) return -1;
        cev->cost.iop_count += 1;
        memset(out, 0, sizeof *out);
        out->kind = WGSL_VAL_INT;
        out->type = oldv.type;
        out->u.i  = old;                            /* atomics return the old value */
        return 1;
    }
    return 0;
    #undef NAME_IS
}
