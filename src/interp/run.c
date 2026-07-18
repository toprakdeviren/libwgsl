/**
 * @file run.c — seeding, workgroup setup, public wgsl_interp entry
 */
#include "internal/interp_priv.h"


/* seeding. */

WGSLNode *find_fn(WGSLNode *tu, const WGSLSource *src, const char *name) {
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *d = tu->children[i];
        if (d->kind != WGSL_NODE_DECL_FUNCTION) continue;
        char nm[96]; name_text(src, d, nm, sizeof nm);
        if (strcmp(nm, name) == 0) return d;
    }
    return NULL;
}

/* Build a zero-initialised value of `t` (recursion covers vec/mat/array/
 * struct).  Runtime-sized arrays get `default_len` elements. */
WGSLValue zero_value(WGSLConstEvaluator *cev, const WGSLTypeChecker *tc,
                            WGSLTypeInfo *t, uint32_t default_len) {
    WGSLValue v; memset(&v, 0, sizeof v);
    v.type = t;
    if (!t) return v;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL: v.kind = WGSL_VAL_BOOL; break;
    case WGSL_TYPE_I32: case WGSL_TYPE_U32: case WGSL_TYPE_ABSTRACT_INT:
        v.kind = WGSL_VAL_INT; break;
    case WGSL_TYPE_F32: case WGSL_TYPE_F16: case WGSL_TYPE_ABSTRACT_FLOAT:
        v.kind = WGSL_VAL_FLOAT; break;
    case WGSL_TYPE_ATOMIC:
        v.kind = WGSL_VAL_INT; v.type = (WGSLTypeInfo *)t->ref; break;
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT: {
        uint32_t n = t->kind == WGSL_TYPE_VEC
                   ? t->width : (uint32_t)t->width * t->rows;
        WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(cev->arena, sizeof(WGSLValue) * (n ? n : 1));
        if (!e) { v.kind = WGSL_VAL_INVALID; break; }
        for (uint32_t i = 0; i < n; i++) e[i] = zero_value(cev, tc, (WGSLTypeInfo *)t->ref, default_len);
        v.kind = t->kind == WGSL_TYPE_VEC ? WGSL_VAL_VEC : WGSL_VAL_MAT;
        v.u.agg.count = n; v.u.agg.elems = e;
        break;
    }
    case WGSL_TYPE_ARRAY: {
        uint32_t n = t->array_len ? t->array_len : default_len;
        if (n > 65536) n = 65536;
        WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(cev->arena, sizeof(WGSLValue) * (n ? n : 1));
        if (!e) { v.kind = WGSL_VAL_INVALID; break; }
        for (uint32_t i = 0; i < n; i++) e[i] = zero_value(cev, tc, (WGSLTypeInfo *)t->ref, default_len);
        v.kind = WGSL_VAL_ARRAY; v.u.agg.count = n; v.u.agg.elems = e;
        break;
    }
    case WGSL_TYPE_STRUCT: {
        WGSLNode *sd = (WGSLNode *)t->ref;
        uint32_t cnt = 0;
        for (uint32_t i = 0; sd && i < sd->child_count; i++)
            if (sd->children[i] && sd->children[i]->kind == WGSL_NODE_DECL_STRUCT_MEMBER) cnt++;
        WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(cev->arena, sizeof(WGSLValue) * (cnt ? cnt : 1));
        if (!e) { v.kind = WGSL_VAL_INVALID; break; }
        uint32_t k = 0;
        for (uint32_t i = 0; sd && i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            WGSLTypeInfo *mt = attrs < m->child_count ? wgsl_typecheck_type_of(tc, m->children[attrs]) : NULL;
            if (mt && mt->kind == WGSL_TYPE_REF) mt = (WGSLTypeInfo *)mt->ref;
            e[k++] = zero_value(cev, tc, mt, default_len);
        }
        v.kind = WGSL_VAL_STRUCT; v.u.agg.count = cnt; v.u.agg.elems = e;
        break;
    }
    default:
        v.kind = WGSL_VAL_INVALID;
    }
    return v;
}

/* Emit a short WGSL-ish type name. */
void fmt_type(SB *s, const WGSLTypeInfo *t) {
    if (!t) { sb_puts(s, "?"); return; }
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL: sb_puts(s, "bool"); break;
    case WGSL_TYPE_I32: sb_puts(s, "i32"); break;
    case WGSL_TYPE_U32: sb_puts(s, "u32"); break;
    case WGSL_TYPE_F32: sb_puts(s, "f32"); break;
    case WGSL_TYPE_F16: sb_puts(s, "f16"); break;
    case WGSL_TYPE_ATOMIC: sb_puts(s, "atomic<"); fmt_type(s, (const WGSLTypeInfo *)t->ref); sb_putc(s, '>'); break;
    case WGSL_TYPE_VEC: sb_puts(s, "vec"); sb_u(s, t->width); sb_putc(s, '<'); fmt_type(s, (const WGSLTypeInfo *)t->ref); sb_putc(s, '>'); break;
    case WGSL_TYPE_MAT: sb_puts(s, "mat"); sb_u(s, t->width); sb_putc(s, 'x'); sb_u(s, t->rows); sb_putc(s, '<'); fmt_type(s, (const WGSLTypeInfo *)t->ref); sb_putc(s, '>'); break;
    case WGSL_TYPE_ARRAY:
        sb_puts(s, "array<"); fmt_type(s, (const WGSLTypeInfo *)t->ref);
        if (t->array_len) { sb_puts(s, ", "); sb_u(s, t->array_len); }
        sb_putc(s, '>'); break;
    case WGSL_TYPE_STRUCT: {
        WGSLNode *sd = (WGSLNode *)t->ref;
        if (sd) { /* struct name is the decl's payload[0] span — unavailable here */ }
        sb_puts(s, "struct"); break;
    }
    default: sb_puts(s, "?");
    }
}

/* pipeline. */

typedef struct {
    WGSLArena arena; WGSLSource src; WGSLDiagBag diag; WGSLLexResult lex;
    WGSLAst ast; WGSLTypeStore types; WGSLResolver res; WGSLConstEvaluator cev; WGSLTypeChecker tc;
    int ok;
} P;

int build(P *p, const char *text, size_t len) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    if (!wgsl_source_init(&p->src, text, len)) return 0;
    if (!wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex)) return 0;
    if (!wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast) || !p->ast.root) return 0;
    if (!wgsl_types_init(&p->types, &p->arena)) return 0;
    wgsl_resolve(&p->ast, &p->src, &p->arena, &p->diag, &p->types, &p->res);
    wgsl_consteval(&p->ast, &p->src, &p->arena, &p->diag, &p->types, &p->res, &p->cev);
    wgsl_typecheck(&p->ast, &p->src, &p->arena, &p->diag, &p->types, &p->res, &p->cev, &p->tc);
    p->ok = 1;
    return 1;
}

void destroy(P *p) {
    wgsl_typecheck_destroy(&p->tc);
    wgsl_consteval_destroy(&p->cev);
    wgsl_resolver_destroy(&p->res);
    wgsl_types_destroy(&p->types);
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

/* Return a fresh malloc'd JSON error document. */
char *interp_error(const char *entry, unsigned gx, unsigned gy, unsigned gz, const char *msg) {
    SB s; memset(&s, 0, sizeof s);
    sb_puts(&s, "{\"ok\":false,\"entry\":\"");
    sb_json(&s, entry ? entry : "", entry ? strlen(entry) : 0);
    sb_puts(&s, "\",\"gid\":["); sb_u(&s, gx); sb_putc(&s, ','); sb_u(&s, gy); sb_putc(&s, ','); sb_u(&s, gz);
    sb_puts(&s, "],\"error\":\""); sb_json(&s, msg, strlen(msg)); sb_puts(&s, "\"}");
    return s.buf ? s.buf : NULL;
}

/* input seeding.
 * Generic: `seeds` is a newline-separated list of `path=value`, where path is
 * a module-scope var name (`foo=…`) or a struct member (`params.count=…`).
 * Lets a trace run with realistic loop bounds instead of all-zero buffers, so
 * data-dependent kernels do real work (and the cost model sees real FLOPs). */
int seed_find(const char *seeds, const char *path, double *out) {
    if (!seeds) return 0;
    size_t pl = strlen(path);
    for (const char *p = seeds; *p; ) {
        const char *eol = p; while (*eol && *eol != '\n') eol++;
        const char *eq = p; while (eq < eol && *eq != '=') eq++;
        if (eq < eol && (size_t)(eq - p) == pl && memcmp(p, path, pl) == 0) {
            *out = atof(eq + 1);
            return 1;
        }
        p = (*eol) ? eol + 1 : eol;
    }
    return 0;
}
void seed_set_scalar(WGSLValue *v, double val) {
    if (v->kind == WGSL_VAL_INT)        v->u.i = (int64_t)val;
    else if (v->kind == WGSL_VAL_FLOAT) v->u.f = val;
}
void seed_apply(const WGSLSource *src, const char *seeds,
                       const char *vname, WGSLValue *zv, WGSLTypeInfo *bt) {
    if (!seeds || !*seeds) return;
    double val;
    if (seed_find(seeds, vname, &val)) { seed_set_scalar(zv, val); return; }
    if (zv->kind == WGSL_VAL_STRUCT && bt && bt->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)bt->ref;
        char path[160], mem[80];
        uint32_t idx = 0;
        for (uint32_t i = 0; sd && i < sd->child_count; i++) {
            WGSLNode *mm = sd->children[i];
            if (!mm || mm->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            name_text(src, mm, mem, sizeof mem);
            snprintf(path, sizeof path, "%s.%s", vname, mem);
            if (seed_find(seeds, path, &val) && idx < zv->u.agg.count)
                seed_set_scalar(&zv->u.agg.elems[idx], val);
            idx++;
        }
    }
}

/* N-lane workgroup setup.
 * `gx/gy/gz` name which *workgroup* to simulate; the lanes are that whole
 * workgroup (one workgroup, N = ∏ workgroup_size).  Each lane gets its own
 * SIMT position builtins; storage/uniform/workgroup memory is shared. */

/* Find an attribute named `want` among `parent`'s first `count` children. */
WGSLNode *interp_find_attr(const WGSLSource *src, const WGSLNode *parent,
                                  uint32_t count, const char *want) {
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && i < parent->child_count; i++) {
        WGSLNode *a = parent->children[i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE || a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = wgsl_node_name_span(nm).offset;
        uint32_t len = wgsl_node_name_span(nm).length;
        if (len == wl && (size_t)off + len <= src->length &&
            memcmp(src->bytes + off, want, wl) == 0) return a;
    }
    return NULL;
}

/* Read the entry's `@workgroup_size(x,y,z)` into `dims`; returns the lane
 * count ∏dims (>=1).  Missing/omitted dimensions default to 1. */
uint32_t interp_wg_size(WGSLConstEvaluator *cev, const WGSLSource *src,
                               WGSLNode *fn, uint32_t dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;
    uint32_t fa = wgsl_fn_attr_count(fn);
    WGSLNode *wgs = interp_find_attr(src, fn, fa, "workgroup_size");
    if (wgs) {
        for (uint32_t k = 1; k < wgs->child_count && k <= 3; k++) {
            WGSLValue v; memset(&v, 0, sizeof v);
            if (wgsl_consteval_expr(cev, wgs->children[k], &v) &&
                v.kind == WGSL_VAL_INT && v.u.i > 0)
                dims[k - 1] = (uint32_t)v.u.i;
        }
    }
    uint64_t prod = (uint64_t)dims[0] * dims[1] * dims[2];
    if (prod < 1) prod = 1;
    return prod > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)prod;
}

/* Which SIMT-position builtin a parameter carries (@builtin(...)), if any. */
typedef enum {
    BI_NONE, BI_LOCAL_ID, BI_LOCAL_INDEX, BI_GLOBAL_ID, BI_WORKGROUP_ID,
    BI_NUM_WORKGROUPS, BI_SUBGROUP_SIZE, BI_SUBGROUP_INVOCATION_ID,
} BuiltinKind;

BuiltinKind param_builtin(const WGSLSource *src, WGSLNode *param) {
    uint32_t ac = wgsl_param_attr_count(param);
    WGSLNode *b = interp_find_attr(src, param, ac, "builtin");
    if (!b || b->child_count < 2 || !b->children[1]) return BI_NONE;
    char nm[48]; name_text(src, b->children[1], nm, sizeof nm);
    if (!strcmp(nm, "local_invocation_id"))    return BI_LOCAL_ID;
    if (!strcmp(nm, "local_invocation_index")) return BI_LOCAL_INDEX;
    if (!strcmp(nm, "global_invocation_id"))   return BI_GLOBAL_ID;
    if (!strcmp(nm, "workgroup_id"))           return BI_WORKGROUP_ID;
    if (!strcmp(nm, "num_workgroups"))         return BI_NUM_WORKGROUPS;
    if (!strcmp(nm, "subgroup_size"))          return BI_SUBGROUP_SIZE;
    if (!strcmp(nm, "subgroup_invocation_id")) return BI_SUBGROUP_INVOCATION_ID;
    return BI_NONE;
}

/* Seed lane `l`'s value for one @builtin parameter into `g` (a zero value of
 * the parameter's type).  `dims` is the workgroup size, `wgid` the workgroup
 * id (gx,gy,gz), `nlanes` the workgroup lane count. */
void seed_builtin(BuiltinKind bk, WGSLValue *g, uint32_t l,
                         const uint32_t dims[3], const uint32_t wgid[3],
                         uint32_t nlanes) {
    uint32_t lid[3] = { l % dims[0], (l / dims[0]) % dims[1],
                        l / (dims[0] * dims[1]) };
    if (bk == BI_LOCAL_INDEX) { if (g->kind == WGSL_VAL_INT) g->u.i = l; return; }
    if (bk == BI_SUBGROUP_INVOCATION_ID) {
        if (g->kind == WGSL_VAL_INT) g->u.i = l % INTERP_SUBGROUP_WIDTH;
        return;
    }
    if (bk == BI_SUBGROUP_SIZE) {
        if (g->kind == WGSL_VAL_INT) {
            uint32_t base = (l / INTERP_SUBGROUP_WIDTH) * INTERP_SUBGROUP_WIDTH;
            uint32_t rem = nlanes - base;
            g->u.i = rem < INTERP_SUBGROUP_WIDTH ? rem : INTERP_SUBGROUP_WIDTH;
        }
        return;
    }
    if (g->kind != WGSL_VAL_VEC) return;
    for (uint32_t k = 0; k < 3 && k < g->u.agg.count; k++) {
        int64_t val = 0;
        switch (bk) {
        case BI_LOCAL_ID:       val = lid[k]; break;
        case BI_GLOBAL_ID:      val = (int64_t)wgid[k] * dims[k] + lid[k]; break;
        case BI_WORKGROUP_ID:   val = wgid[k]; break;
        case BI_NUM_WORKGROUPS: val = 1; break;   /* single workgroup */
        default: break;
        }
        g->u.agg.elems[k].u.i = val;
    }
}

/* cost band: interp_band.c */

char *wgsl_interp(const char *src, const char *entry,
                  unsigned gx, unsigned gy, unsigned gz,
                  unsigned default_len, const char *seeds) {
    if (!src || !entry) return interp_error(entry, gx, gy, gz, "null source or entry");
    if (default_len == 0) default_len = 8;

    P p;
    if (!build(&p, src, strlen(src))) { destroy(&p); return interp_error(entry, gx, gy, gz, "frontend failed to parse the module"); }

    WGSLNode *fn = find_fn(p.ast.root, &p.src, entry);
    if (!fn) { char *e = interp_error(entry, gx, gy, gz, "entry point not found"); destroy(&p); return e; }

    Interp ip; memset(&ip, 0, sizeof ip);
    ip.cev = &p.cev; ip.tc = &p.tc; ip.src = &p.src; ip.default_len = default_len;
    p.cev.simt.user_call = user_call_cb;
    p.cev.simt.user_call_ud = &ip;
    p.cev.simt.builtin_call = builtin_call_cb;
    p.cev.simt.builtin_call_ud = &ip;
    p.cev.simt.mem_access = mem_access_cb;
    p.cev.simt.mem_access_ud = &ip;
    p.cev.simt.runtime = 1;
    p.cev.simt.cur_lane = 0;

    /* `gx/gy/gz` is the *focus* invocation's global id — the one whose trace and
     * per-invocation cost we report (backward compatible with the single-
     * invocation inspector).  To make cooperative ops (subgroup/barrier) correct
     * for it, we run its entire workgroup: N = ∏ @workgroup_size lanes sharing
     * one storage/uniform/workgroup backing store, each lane seeded its own
     * SIMT-position builtins.  Lane-aware environment: per-lane symbols
     * (locals/params/private) resolve against `cur_lane`; shared symbols alias
     * one slot the whole workgroup sees. */
    uint32_t dims[3];
    uint32_t nlanes = interp_wg_size(&p.cev, &p.src, fn, dims);
    int lanes_clamped = 0;
    if (nlanes > INTERP_MAX_LANES) { nlanes = INTERP_MAX_LANES; lanes_clamped = 1; }
    /* The workgroup containing the focus invocation, and its lane within it. */
    uint32_t wgid[3] = { gx / dims[0], gy / dims[1], gz / dims[2] };
    uint32_t flid[3] = { gx % dims[0], gy % dims[1], gz % dims[2] };
    uint32_t focus = flid[0] + flid[1] * dims[0] + flid[2] * dims[0] * dims[1];
    if (focus >= nlanes) focus = 0;
    ip.nlanes = nlanes; ip.focus_lane = focus;
    p.cev.simt.nlanes = nlanes;

    /* Seed every module-scope var and remember them so we can report the
     * post-run contents.  Shared (storage/uniform/workgroup) vars get one slot
     * the whole workgroup aliases; `var<private>` is per-invocation, so each
     * lane gets its own zero copy. */
    WGSLSymbol **bufsym = NULL; uint32_t nbuf = 0, bufcap = 0;
    for (uint32_t i = 0; i < p.ast.root->child_count; i++) {
        WGSLNode *d = p.ast.root->children[i];
        if (!d || d->kind != WGSL_NODE_DECL_VAR) continue;
        WGSLSymbol *s = decl_symbol(&p.cev, d);
        if (!s || !s->type) continue;
        WGSLTypeInfo *bt = s->type;
        if (bt->kind == WGSL_TYPE_REF) bt = (WGSLTypeInfo *)bt->ref;
        char vname[80]; name_text(&p.src, d, vname, sizeof vname);
        int per_lane = wgsl_consteval_sym_is_per_lane(s);
        uint32_t seed_lanes = per_lane ? nlanes : 1;
        for (uint32_t l = 0; l < seed_lanes; l++) {
            p.cev.simt.cur_lane = l;
            WGSLValue zv = zero_value(&p.cev, &p.tc, bt, default_len);
            seed_apply(&p.src, seeds, vname, &zv, bt);   // realistic inputs (else all-zero)
            /* Flag DRAM-backed buffers so the cost model counts loads through
             * them; workgroup memory is banked (bank-conflict analysis). */
            if (s->as == WGSL_AS_STORAGE || s->as == WGSL_AS_UNIFORM)
                zv.flags_ |= WGSL_VAL_FLAG_STORAGE;
            else if (s->as == WGSL_AS_WORKGROUP)
                zv.flags_ |= WGSL_VAL_FLAG_WORKGROUP;
            wgsl_consteval_set_value(&p.cev, s, &zv);
        }
        p.cev.simt.cur_lane = 0;
        if (nbuf == bufcap) {           /* grow: don't leave later vars unseeded */
            uint32_t nc = bufcap ? bufcap * 2 : 16;
            WGSLSymbol **g = (WGSLSymbol **)realloc(bufsym, (size_t)nc * sizeof *g);
            if (!g) break;              /* OOM: report what we have */
            bufsym = g; bufcap = nc;
        }
        bufsym[nbuf++] = s;
    }

    /* Seed pipeline-overridable constants: fold the default value (or a
     * `seeds` override) into the binding so override-driven loop bounds/branches
     * are concrete for both the run and the static cost band.  Overrides are
     * uniform → one shared binding. */
    for (uint32_t i = 0; i < p.ast.root->child_count; i++) {
        WGSLNode *d = p.ast.root->children[i];
        if (!d || d->kind != WGSL_NODE_DECL_OVERRIDE) continue;
        WGSLSymbol *s = decl_symbol(&p.cev, d);
        if (!s) continue;
        int has_init = wgsl_varlike_has_init(d);
        WGSLValue v; memset(&v, 0, sizeof v);
        int have = has_init && d->child_count &&
                   wgsl_consteval_expr(&p.cev, d->children[d->child_count - 1], &v);
        char vname[80]; name_text(&p.src, d, vname, sizeof vname);
        double sval;
        if (seed_find(seeds, vname, &sval)) {          /* explicit override value */
            if (!have) { v.kind = WGSL_VAL_INT; v.type = s->type; have = 1; }
            seed_set_scalar(&v, sval);
        }
        if (have) wgsl_consteval_set_value(&p.cev, s, &v);
    }

    /* Seed each lane's entry parameters: @builtin(...) params get that lane's
     * SIMT position; everything else is zeroed.  (Done per-lane directly rather
     * than via a single arg vector, since builtins differ per lane.) */
    for (uint32_t l = 0; l < nlanes; l++) {
        p.cev.simt.cur_lane = l;
        for (uint32_t c = 0; c < fn->child_count; c++) {
            WGSLNode *pn = fn->children[c];
            if (pn->kind != WGSL_NODE_DECL_PARAM) continue;
            WGSLSymbol *ps = decl_symbol(&p.cev, pn);
            if (!ps) continue;
            WGSLTypeInfo *pt = ps->type;
            WGSLValue g = zero_value(&p.cev, &p.tc, pt, default_len);
            seed_builtin(param_builtin(&p.src, pn), &g, l, dims, wgid, nlanes);
            wgsl_consteval_set_value(&p.cev, ps, &g);
        }
    }
    p.cev.simt.cur_lane = 0;

    /* Reset the cost counters so only the entry-body execution is measured
     * (module-scope const-eval during build ran with runtime = 0 anyway).
     * Cost counters (flop/iop/bytes/op_bucket/f16/packed) are the *focus*
     * lane's alone — peer lanes run with cost deltas rolled back (see
     * cost_snapshot/cost_rollback in vm.c).  Shared buffers still reflect
     * the whole workgroup. */
    p.cev.cost.flop_count = p.cev.cost.iop_count = p.cev.cost.bytes_loaded = p.cev.cost.bytes_stored = 0;
    memset(p.cev.cost.op_bucket, 0, sizeof p.cev.cost.op_bucket);
    p.cev.cost.flop_f16 = p.cev.cost.packed_ops = 0;

    WGSLValue result; memset(&result, 0, sizeof result);
    /* Diagnostics emitted DURING execution flag constructs the interpreter does
     * not model (subgroup/quad ops in helper scope, unimplemented builtins).
     * They make the trace "partial": surface them as `warnings` and clear `ok`
     * so the studio can badge it.  A clamped lane count is likewise partial. */
    size_t warn_lo = wgsl_diag_count(&p.diag);
    if (lanes_clamped)
        wgsl_diag_emit_at(&p.diag, &p.src, WGSL_DIAG_WARNING,
                          fn->span_offset, fn->span_length, NULL,
                          "workgroup has more than %u invocations; simulating "
                          "the first %u lanes only", INTERP_MAX_LANES, nlanes);
    const Prog *prog = get_prog(&ip, fn_body(fn));
    uint32_t *lane_ids = (uint32_t *)malloc((size_t)nlanes * sizeof *lane_ids);
    WGSLValue *rets = (WGSLValue *)malloc((size_t)nlanes * sizeof *rets);
    int ran = 0;
    if (prog && lane_ids && rets) {
        for (uint32_t l = 0; l < nlanes; l++) lane_ids[l] = l;
        vm_run(&ip, prog, lane_ids, nlanes, rets);
        result = rets[ip.focus_lane];   /* focus lane's return (void for compute) */
        ran = 1;
    }
    free(lane_ids); free(rets);
    p.cev.simt.cur_lane = ip.focus_lane;      /* report the focus lane's private state */
    size_t warn_hi = wgsl_diag_count(&p.diag);
    /* Truncation (STEP_CAP) is a shared all-lane budget: a heavily divergent
     * peer can exhaust it and cut the focus lane's work short, so a truncated
     * run's reported result may be incomplete → not ok. */
    int ok = (warn_hi == warn_lo) && !ip.truncated;

    /* serialise. */
    SB out; memset(&out, 0, sizeof out);
    sb_puts(&out, ok ? "{\"ok\":true,\"entry\":\"" : "{\"ok\":false,\"entry\":\"");
    sb_json(&out, entry, strlen(entry));
    /* `gid` is the focus invocation's global id (gx,gy,gz) — the invocation
     * whose trace/cost this run reports.  Its whole workgroup is simulated for
     * correct cooperative ops; `lanes` / `workgroup_size` describe that group. */
    sb_puts(&out, "\",\"gid\":["); sb_u(&out, gx); sb_putc(&out, ','); sb_u(&out, gy); sb_putc(&out, ','); sb_u(&out, gz); sb_putc(&out, ']');
    sb_puts(&out, ",\"workgroup_size\":["); sb_u(&out, dims[0]); sb_putc(&out, ',');
    sb_u(&out, dims[1]); sb_putc(&out, ','); sb_u(&out, dims[2]); sb_putc(&out, ']');
    sb_puts(&out, ",\"lanes\":"); sb_u(&out, nlanes);

    sb_puts(&out, ",\"steps\":[");
    if (ip.trace.buf) sb_write(&out, ip.trace.buf, ip.trace.len);
    sb_putc(&out, ']');

    sb_puts(&out, ",\"buffers\":[");
    for (uint32_t i = 0; i < nbuf; i++) {
        if (i) sb_putc(&out, ',');
        const WGSLValue *bv = wgsl_consteval_value_of(&p.cev, bufsym[i]);
        sb_puts(&out, "{\"name\":\""); sb_json(&out, bufsym[i]->name, bufsym[i]->name_len); sb_puts(&out, "\",\"type\":\"");
        WGSLTypeInfo *bt = bufsym[i]->type;
        if (bt && bt->kind == WGSL_TYPE_REF) bt = (WGSLTypeInfo *)bt->ref;
        fmt_type(&out, bt);
        sb_puts(&out, "\",\"values\":[");
        if (bv && (bv->kind == WGSL_VAL_ARRAY || bv->kind == WGSL_VAL_VEC || bv->kind == WGSL_VAL_STRUCT || bv->kind == WGSL_VAL_MAT)) {
            uint32_t cap = bv->u.agg.count > WGSL_INTERP_BUFFER_VALUES_CAP
                ? WGSL_INTERP_BUFFER_VALUES_CAP : bv->u.agg.count;
            for (uint32_t j = 0; j < cap; j++) {
                if (j) sb_putc(&out, ',');
                ip.scratch.len = 0; fmt_value(&ip.scratch, &bv->u.agg.elems[j]);
                sb_putc(&out, '"'); sb_json(&out, ip.scratch.buf ? ip.scratch.buf : "", ip.scratch.len); sb_putc(&out, '"');
            }
        } else if (bv && bv->kind != WGSL_VAL_INVALID) {
            ip.scratch.len = 0; fmt_value(&ip.scratch, bv);
            sb_putc(&out, '"'); sb_json(&out, ip.scratch.buf ? ip.scratch.buf : "", ip.scratch.len); sb_putc(&out, '"');
        }
        sb_putc(&out, ']'); sb_putc(&out, '}');
    }
    sb_putc(&out, ']');

    sb_puts(&out, ",\"anomalies\":{\"nan\":"); sb_u(&out, (unsigned long)ip.nan_hits);
    sb_puts(&out, ",\"inf\":"); sb_u(&out, (unsigned long)ip.inf_hits); sb_putc(&out, '}');

    /* Static cost model (per-invocation): FLOPs + storage bytes moved.  The
     * ratio flops/bytes is the arithmetic intensity — scale-invariant, so this
     * single thread answers the roofline "memory- vs compute-bound" verdict. */
    sb_puts(&out, ",\"cost\":{\"flops\":"); sb_u(&out, (unsigned long)p.cev.cost.flop_count);
    sb_puts(&out, ",\"iops\":"); sb_u(&out, (unsigned long)p.cev.cost.iop_count);
    sb_puts(&out, ",\"bytes_loaded\":"); sb_u(&out, (unsigned long)p.cev.cost.bytes_loaded);
    sb_puts(&out, ",\"bytes_stored\":"); sb_u(&out, (unsigned long)p.cev.cost.bytes_stored);

    /* Instruction-mix breakdown: per-op-class occurrence counts, so the
     * roofline card shows the mix, not one scalar.  Buckets refine flops/iops. */
    {
        static const char *MIX[WGSL_OPB_COUNT] = {
            "fadd", "fmul", "fdiv", "ftrans", "iadd", "imul", "idiv", "ishift", "cmp" };
        sb_puts(&out, ",\"mix\":{");
        for (int i = 0; i < WGSL_OPB_COUNT; i++) {
            if (i) sb_putc(&out, ',');
            sb_putc(&out, '"'); sb_puts(&out, MIX[i]); sb_puts(&out, "\":");
            sb_u(&out, (unsigned long)p.cev.cost.op_bucket[i]);
        }
        sb_putc(&out, '}');
    }
    sb_putc(&out, '}');   /* close cost */

    /* Roofline placement: the kernel's arithmetic intensity (flop/byte)
     * fixes which side of each arch's ridge point it sits — memory- vs
     * compute-bound — independent of absolute counts.  Per-arch peak FLOP/s +
     * bandwidth presets let the studio plot the kernel on a drawn roofline.  No
     * wall-clock is modelled, so `attainable_gflops` is the ceiling for a kernel
     * of this intensity, not a measured rate. */
    {
        static const struct { const char *name; double gflops, gbps; } ARCH[] = {
            { "generic-gpu",  10000.0,  400.0 },   /* ~10 TFLOP/s f32, 400 GB/s */
            { "apple-m3-max", 14200.0,  400.0 },
            { "rtx-4090",     82580.0, 1008.0 },
            { "a100-sxm",     19500.0, 2039.0 },
        };
        uint64_t flops = p.cev.cost.flop_count, iops = p.cev.cost.iop_count;
        uint64_t bytes = p.cev.cost.bytes_loaded + p.cev.cost.bytes_stored;
        double fi = bytes ? (double)flops / (double)bytes : 0.0;
        double oi = bytes ? (double)(flops + iops) / (double)bytes : 0.0;
        char t[48];
        sb_puts(&out, ",\"roofline\":{\"bytes\":"); sb_u(&out, (unsigned long)bytes);
        int n = snprintf(t, sizeof t, "%.4g", fi);
        if (n > 0) { sb_puts(&out, ",\"flop_intensity\":"); sb_write(&out, t, (size_t)n); }
        n = snprintf(t, sizeof t, "%.4g", oi);
        if (n > 0) { sb_puts(&out, ",\"op_intensity\":"); sb_write(&out, t, (size_t)n); }
        /* f16 / packed-math: GPUs run f16 at ~2× f32 throughput, so the
         * effective compute discounts f16 flops by half; packed dot instructions
         * are tensor-path eligible. */
        double eff = (double)flops - 0.5 * (double)p.cev.cost.flop_f16;
        sb_puts(&out, ",\"f16_flops\":"); sb_u(&out, (unsigned long)p.cev.cost.flop_f16);
        sb_puts(&out, ",\"packed_ops\":"); sb_u(&out, (unsigned long)p.cev.cost.packed_ops);
        n = snprintf(t, sizeof t, "%.1f", eff < 0 ? 0.0 : eff);
        if (n > 0) { sb_puts(&out, ",\"effective_flops\":"); sb_write(&out, t, (size_t)n); }
        sb_puts(&out, ",\"tensor_path\":"); sb_puts(&out, p.cev.cost.packed_ops ? "true" : "false");
        sb_puts(&out, ",\"presets\":[");
        for (size_t a = 0; a < sizeof ARCH / sizeof ARCH[0]; a++) {
            if (a) sb_putc(&out, ',');
            double ridge = ARCH[a].gflops / ARCH[a].gbps;           /* flop/byte at the knee */
            double attain = fi * ARCH[a].gbps;                       /* bandwidth-bound ceiling */
            if (attain > ARCH[a].gflops) attain = ARCH[a].gflops;    /* clamp to compute peak */
            const char *bound = (fi < ridge) ? "memory" : "compute";
            sb_puts(&out, "{\"name\":\""); sb_puts(&out, ARCH[a].name);
            n = snprintf(t, sizeof t, "%.0f", ARCH[a].gflops); sb_puts(&out, "\",\"peak_gflops\":"); if (n>0) sb_write(&out, t, (size_t)n);
            n = snprintf(t, sizeof t, "%.0f", ARCH[a].gbps);   sb_puts(&out, ",\"bandwidth_gbps\":"); if (n>0) sb_write(&out, t, (size_t)n);
            n = snprintf(t, sizeof t, "%.4g", ridge);          sb_puts(&out, ",\"ridge\":");   if (n>0) sb_write(&out, t, (size_t)n);
            n = snprintf(t, sizeof t, "%.4g", attain);         sb_puts(&out, ",\"attainable_gflops\":"); if (n>0) sb_write(&out, t, (size_t)n);
            sb_puts(&out, ",\"bound\":\""); sb_puts(&out, bound); sb_puts(&out, "\"}");
        }
        sb_puts(&out, "]}");
    }

    /* Warp memory analysis.  `coalescing` = DRAM bytes the warp asked
     * for ÷ bytes actually moved (128B lines) — 1.0 = perfectly coalesced,
     * lower = strided/scattered.  `bank_conflicts` = extra serialized shared-
     * memory transactions from lanes colliding on one of 32 banks.  Measured on
     * direct `buf[idx]` accesses (the dominant pattern). */
    {
        uint64_t thr = ip.dram_theoretical, eff = ip.dram_effective;
        char t[32];
        sb_puts(&out, ",\"memory\":{\"dram_theoretical_bytes\":"); sb_u(&out, (unsigned long)thr);
        sb_puts(&out, ",\"dram_effective_bytes\":"); sb_u(&out, (unsigned long)eff);
        double coal = eff ? (double)thr / (double)eff : 1.0;
        int n = snprintf(t, sizeof t, "%.4f", coal);
        if (n > 0) { sb_puts(&out, ",\"coalescing\":"); sb_write(&out, t, (size_t)n); }
        sb_puts(&out, ",\"shared_accesses\":"); sb_u(&out, (unsigned long)ip.shared_accesses);
        sb_puts(&out, ",\"bank_conflicts\":"); sb_u(&out, (unsigned long)ip.bank_conflicts);
        sb_putc(&out, '}');
    }

    /* Data races: shared cells two lanes touch in one barrier epoch with
     * at least one write.  The interpreter doubles as a correctness linter —
     * the reported trace is one arbitrary interleaving of a racy program. */
    {
        static const char *RK[3] = { "write-write", "read-after-write", "write-after-read" };
        sb_puts(&out, ",\"races\":{\"count\":"); sb_u(&out, (unsigned long)ip.race_total);
        if (ip.race_cells_truncated)
            sb_puts(&out, ",\"truncated\":true");
        sb_puts(&out, ",\"samples\":[");
        for (uint32_t i = 0; i < ip.nraces; i++) {
            if (i) sb_putc(&out, ',');
            sb_puts(&out, "{\"line\":"); sb_u(&out, ip.races[i].line);
            sb_puts(&out, ",\"kind\":\""); sb_puts(&out, RK[ip.races[i].kind <= 2 ? ip.races[i].kind : 0]);
            sb_puts(&out, "\"}");
        }
        sb_puts(&out, "]}");
    }

    /* Cost band: a seedless static interval analysis derives best/worst
     * flops/iops/bytes across the input space (loop trip-count ranges + branch
     * forks), and the concrete focus run above anchors it as a sampled point. */
    {
        BandA A; memset(&A, 0, sizeof A);
        A.tc = &p.tc; A.src = &p.src; A.cev = &p.cev;
        IEnv env; memset(&env, 0, sizeof env);
        /* Seed the entry's scalar @builtin params as [0, grid); vecs stay ⊤. */
        for (uint32_t c = 0; c < fn->child_count; c++) {
            WGSLNode *pn = fn->children[c];
            if (pn->kind != WGSL_NODE_DECL_PARAM) continue;
            WGSLSymbol *ps = decl_symbol(&p.cev, pn);
            if (ps && ps->type && ps->type->kind == WGSL_TYPE_U32)
                env_set(&env, ps, iv_rng(0, BAND_GRID - 1));
        }
        Band bd = band_stmt(&A, fn_body(fn), &env);
        free(env.v);

        /* Anchor to the concrete run (sampling fallback): the band must contain
         * the measured focus-lane cost. */
        double cflop = (double)p.cev.cost.flop_count, ciop = (double)p.cev.cost.iop_count;
        double cbytes = (double)(p.cev.cost.bytes_loaded + p.cev.cost.bytes_stored);
        double fmin = bd.min.flops, fmax = bd.max.flops;
        double imin = bd.min.iops,  imax = bd.max.iops;
        double bmin = bd.min.bytes_ld + bd.min.bytes_st;
        double bmax = bd.max.bytes_ld + bd.max.bytes_st;
        if (cflop < fmin) fmin = cflop;
        if (cflop > fmax) fmax = cflop;
        if (ciop < imin) imin = ciop;
        if (ciop > imax) imax = ciop;
        if (cbytes < bmin) bmin = cbytes;
        if (cbytes > bmax) bmax = cbytes;

        sb_puts(&out, ",\"cost_band\":{\"method\":\"");
        sb_puts(&out, A.imprecise ? "interval+sampled" : "interval");
        sb_puts(&out, "\"");
        #define BAND_PAIR(key, lo, hi) do {                          \
            sb_puts(&out, ",\"" key "\":{\"min\":"); sb_dbl(&out, "%.0f", (lo)); \
            sb_puts(&out, ",\"max\":"); sb_dbl(&out, "%.0f", (hi)); sb_putc(&out,'}'); \
        } while (0)
        BAND_PAIR("flops", fmin, fmax);
        BAND_PAIR("iops",  imin, imax);
        BAND_PAIR("bytes", bmin, bmax);
        #undef BAND_PAIR
        /* Arithmetic-intensity band (flop/byte): least = min flops / max bytes. */
        double i_lo = bmax > 0 ? fmin / bmax : 0.0;
        double i_hi = bmin > 0 ? fmax / bmin : (fmax > 0 ? 1e9 : 0.0);
        sb_puts(&out, ",\"intensity\":{\"min\":"); sb_dbl(&out, "%.4g", i_lo);
        sb_puts(&out, ",\"max\":"); sb_dbl(&out, "%.4g", i_hi); sb_putc(&out,'}');
        sb_puts(&out, ",\"data_dependent_branches\":"); sb_u(&out, (unsigned long)A.ddbranches);
        {
            double spread = (fmax - fmin) + (imax - imin) + (bmax - bmin);
            unsigned long hint = A.ddbranches ? (unsigned long)A.ddbranches : 0;
            for (int i = 0; i < A.nloops; i++) {
                if (A.loops[i].bounded) {
                    double trips = A.loops[i].tmax - A.loops[i].tmin;
                    if (trips < 1.0) trips = A.loops[i].tmax;
                    if (trips > 0.0) hint += (unsigned long)(trips > 32.0 ? 32.0 : trips);
                } else {
                    hint += 4;
                }
            }
            if (spread > 0.0) {
                unsigned long s = (unsigned long)(spread > 64.0 ? 64.0 : spread);
                hint += s ? s : 1;
            }
            sb_puts(&out, ",\"savings_hint\":"); sb_u(&out, hint);
        }
        sb_puts(&out, ",\"loops\":[");
        for (int i = 0; i < A.nloops; i++) {
            if (i) sb_putc(&out, ',');
            sb_puts(&out, "{\"line\":"); sb_u(&out, A.loops[i].line);
            sb_puts(&out, ",\"trip_min\":"); sb_dbl(&out, "%.0f", A.loops[i].tmin);
            sb_puts(&out, ",\"trip_max\":"); sb_dbl(&out, "%.0f", A.loops[i].tmax);
            sb_puts(&out, ",\"bounded\":"); sb_puts(&out, A.loops[i].bounded ? "true" : "false");
            sb_putc(&out, '}');
        }
        sb_puts(&out, "]}");
    }

    /* Divergence / occupancy.  `occupancy` is SIMT efficiency — the mean
     * fraction of the N lanes active per scheduler step (1.0 = fully converged);
     * `tax` = 1 − occupancy is the share of warp slots wasted on divergence.
     * `by_line` is the per-source-line lane-activity heatmap. */
    {
        double occ = (ip.warp_steps && nlanes)
            ? (double)ip.lane_steps / ((double)ip.warp_steps * nlanes) : 1.0;
        char t[32];
        sb_puts(&out, ",\"divergence\":{\"warp_steps\":"); sb_u(&out, (unsigned long)ip.warp_steps);
        sb_puts(&out, ",\"lane_steps\":"); sb_u(&out, (unsigned long)ip.lane_steps);
        int n = snprintf(t, sizeof t, "%.4f", occ);        if (n > 0) { sb_puts(&out, ",\"occupancy\":"); sb_write(&out, t, (size_t)n); }
        n = snprintf(t, sizeof t, "%.4f", 1.0 - occ);      if (n > 0) { sb_puts(&out, ",\"tax\":"); sb_write(&out, t, (size_t)n); }
        sb_puts(&out, ",\"by_line\":[");
        int first = 1;
        for (uint32_t ln = 1; ln <= ip.max_line && ln < ip.linediv_cap; ln++) {
            struct LineDiv *d = &ip.linediv[ln];
            if (d->steps == 0) continue;
            if (!first) sb_putc(&out, ',');
            first = 0;
            sb_puts(&out, "{\"line\":"); sb_u(&out, ln);
            sb_puts(&out, ",\"steps\":"); sb_u(&out, d->steps);
            n = snprintf(t, sizeof t, "%.2f", (double)d->active_sum / d->steps);
            if (n > 0) { sb_puts(&out, ",\"avg_active\":"); sb_write(&out, t, (size_t)n); }
            sb_puts(&out, ",\"min_active\":"); sb_u(&out, d->min_active);
            sb_puts(&out, ",\"max_active\":"); sb_u(&out, d->max_active);
            sb_putc(&out, '}');
        }
        sb_puts(&out, "]}");
    }

    sb_puts(&out, ",\"return\":");
    if (ran && result.kind != WGSL_VAL_INVALID) {
        ip.scratch.len = 0; fmt_value(&ip.scratch, &result);
        sb_putc(&out, '"'); sb_json(&out, ip.scratch.buf ? ip.scratch.buf : "", ip.scratch.len); sb_putc(&out, '"');
    } else {
        sb_puts(&out, "null");
    }

    sb_puts(&out, ",\"steps_executed\":"); sb_u(&out, (unsigned long)ip.steps);
    sb_puts(&out, ",\"truncated\":"); sb_puts(&out, ip.truncated ? "true" : "false");

    /* Runtime warnings: constructs encountered but not modelled by the
     * single-lane interpreter (studio badges the trace "partial"). */
    sb_puts(&out, ",\"warnings\":[");
    for (size_t i = warn_lo; i < warn_hi; i++) {
        const WGSLDiagnostic *w = wgsl_diag_at(&p.diag, i);
        if (i > warn_lo) sb_putc(&out, ',');
        sb_puts(&out, "{\"line\":"); sb_u(&out, (unsigned long)(w ? w->line : 0));
        sb_puts(&out, ",\"message\":\"");
        if (w && w->message) sb_json(&out, w->message, strlen(w->message));
        sb_puts(&out, "\"}");
    }
    sb_putc(&out, ']');

    sb_putc(&out, '}');

    free(ip.trace.buf);
    free(ip.scratch.buf);
    free(bufsym);
    free(ip.linediv);
    free(ip.macc);
    free(ip.cells);
    free(ip.cell_htab);
    progs_free(&ip);
    destroy(&p);
    return out.buf ? out.buf : interp_error(entry, gx, gy, gz, "out of memory");
}

void wgsl_free_string(char *s) { free(s); }
