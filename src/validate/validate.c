/**
 * @file validate.c — WGSL validator driver.
 *
 * Slim driver that ties together the validator's split translation
 * units:
 *   validate/validate.c   — this file: cycle DFS infra, all cycle
 *                           passes, public `wgsl_validate` entry, and
 *                           const_assert evaluation.
 *   validate/attrs.c      — §12 attribute conformance.
 *   validate/io.c         — §13 entry-point IO shape.
 *   validate/behavior.c   — §9.7 behavior-set analysis.
 *   validate/access.c     — §14.3 + §14.5 + §9.4.11 reachability.
 *   validate/layout.c     — §14.4 + §13.3.2 resource bindings.
 *   validate/uniformity.c — §15.2 uniformity analysis.
 *
 * Cross-file sharing happens through `internal/validate_priv.h`.
 */
#include "internal/validate.h"
#include "internal/validate_priv.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/ast.h"
#include "internal/layout.h"

/* Diagnostics. */

void wgsl_val_error(
    WGSLValidator *v, const WGSLNode *at, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(v->diag, v->src, WGSL_DIAG_ERROR,
                      off, len, NULL, "%s", buf);
    v->had_error = 1;
}

/** Emit a filterable diagnostic with the given rule name.  The
 *  diagnostic filter in `WGSLDiagBag` may suppress or reclassify it
 *  based on `diagnostic(...)` directives / `@diagnostic` attributes. */
void wgsl_val_filterable(
    WGSLValidator *v, const WGSLNode *at,
    WGSLDiagSeverity severity, const char *rule,
    const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(v->diag, v->src, severity,
                      off, len, rule, "%s", buf);
    /* Don't set had_error — the filter may have suppressed the
     * diagnostic or downgraded it to info/warning. */
}

/* Symbol -> all_decls index. */

int wgsl_val_sym_index(const WGSLValidator *v, const WGSLSymbol *s) {
    return v ? wgsl_resolver_index_for_symbol(v->res, s) : -1;
}

/* Generic cycle DFS. */

/* Color / EdgeCb / EdgeEnum typedefs moved to internal/validate_priv.h
 * for cross-file sharing with access.c (analyze_static_access) and
 * uniformity.c (compute_fn_uniformity_reqs). */

typedef struct {
    Color    *state;
    EdgeEnum  edges;
    const char *cycle_label;     /* "recursion", "struct cycle", … */
    int       had_cycle;         /* set when DFS reports one         */
} CycleCtx;

static void dfs_visit(
    WGSLValidator *v, const WGSLSymbol *s, CycleCtx *cx);

static void dfs_edge(
    WGSLValidator *v,
    const WGSLSymbol *to, const WGSLNode *at, void *ctx)
{
    CycleCtx *cx = (CycleCtx *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0) return;

    if (cx->state[idx] == COLOR_GRAY) {
        wgsl_val_error(v, at,
            "%s: '%.*s' is referenced through a back-edge",
            cx->cycle_label,
            (int)to->name_len, to->name);
        cx->had_cycle = 1;
        return;
    }
    if (cx->state[idx] == COLOR_WHITE) {
        dfs_visit(v, to, cx);
    }
}

static void dfs_visit(
    WGSLValidator *v, const WGSLSymbol *s, CycleCtx *cx)
{
    int idx = wgsl_val_sym_index(v, s);
    if (idx < 0) return;
    cx->state[idx] = COLOR_GRAY;
    cx->edges(v, s, dfs_edge, cx);
    cx->state[idx] = COLOR_BLACK;
}

int wgsl_val_run_cycle_check(
    WGSLValidator *v,
    int (*pick)(const WGSLSymbol *),
    EdgeEnum edges, const char *label)
{
    if (!v->res) return 1;
    size_t n = v->res->all_decl_count;
    Color *state = (Color *)calloc(n, sizeof *state);
    if (!state) return 0;

    CycleCtx cx = { state, edges, label, 0 };

    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!pick(s)) continue;
        if (state[i] != COLOR_WHITE) continue;
        dfs_visit(v, s, &cx);
    }

    free(state);
    return !cx.had_cycle;
}

/* Edge enumerators. */

static int pick_function(const WGSLSymbol *s) { return s->kind == WGSL_SYM_FUNCTION; }
static int pick_struct  (const WGSLSymbol *s) { return s->kind == WGSL_SYM_STRUCT;   }
static int pick_alias   (const WGSLSymbol *s) { return s->kind == WGSL_SYM_TYPE_ALIAS; }
static int pick_const   (const WGSLSymbol *s) { return s->kind == WGSL_SYM_CONST;    }
static int pick_override(const WGSLSymbol *s) { return s->kind == WGSL_SYM_OVERRIDE; }
static int pick_module_var(const WGSLSymbol *s) {
    /* §5 — module-scope `var`s only.  Fn-local vars carry AS==FUNCTION
     * (or NONE) and can't form init-expr cycles in any case. */
    return s->kind == WGSL_SYM_VAR && s->as != WGSL_AS_FUNCTION &&
           s->as != WGSL_AS_NONE;
}

/* — call-graph: scan a fn body for EXPR_CALL whose callee is a user fn — */

static void scan_calls(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_CALL && n->child_count >= 1) {
        WGSLNode *callee = n->children[0];
        if (callee) {
            WGSLNode *head = callee;
            if (head->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
                head->child_count > 0)
            {
                head = head->children[0];
            }
            if (head && head->kind == WGSL_NODE_EXPR_IDENT) {
                WGSLSymbol *target = wgsl_node_resolved_symbol(head);
                if (target && target->kind == WGSL_SYM_FUNCTION) {
                    cb(v, target, n, ctx);
                }
            }
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_calls(v, n->children[i], cb, ctx);
    }
}

void wgsl_val_enum_call_edges(
    WGSLValidator *v, const WGSLSymbol *fn, EdgeCb cb, void *ctx)
{
    if (!fn->ast) return;
    /* fn body = last child of DECL_FUNCTION (a STMT_COMPOUND). */
    if (fn->ast->child_count == 0) return;
    WGSLNode *body = fn->ast->children[fn->ast->child_count - 1];
    scan_calls(v, body, cb, ctx);
}

/* — struct cycle: edge per field type that unwraps to another struct —
 *
 * Walks the AST type-specifier directly (rather than going through the
 * TC side-table, which doesn't currently store type-spec → TypeInfo
 * mappings).  Recurses through `array<T, N>` and through alias chains
 * to bottom out at a STRUCT or scalar leaf. */

static WGSLSymbol *resolve_typespec_struct(
    WGSLValidator *v, WGSLNode *tspec)
{
    if (!tspec) return NULL;
    if (tspec->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(tspec);
        if (!sym) return NULL;
        if (sym->kind == WGSL_SYM_STRUCT) return sym;
        if (sym->kind == WGSL_SYM_TYPE_ALIAS && sym->type) {
            /* Look the alias's RHS struct up by AST. */
            const WGSLNode *want = NULL;
            const WGSLTypeInfo *t = sym->type;
            while (t) {
                if (t->kind == WGSL_TYPE_STRUCT) {
                    want = (const WGSLNode *)t->ref; break;
                }
                if (t->kind == WGSL_TYPE_ARRAY) {
                    t = (const WGSLTypeInfo *)t->ref; continue;
                }
                break;
            }
            if (!want) return NULL;
            for (size_t i = 0; i < v->res->all_decl_count; i++) {
                WGSLSymbol *cand = v->res->all_decls[i];
                if (cand->ast == want && cand->kind == WGSL_SYM_STRUCT) {
                    return cand;
                }
            }
        }
        return NULL;
    }
    if (tspec->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (tspec->child_count < 2) return NULL;
        WGSLNode *head = tspec->children[0];
        WGSLSymbol *hsym = wgsl_node_resolved_symbol(head);
        if (!hsym) return NULL;
        /* `array<T, N>` — element type is the cycle-relevant subtree. */
        if (hsym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
            hsym->name_len == 5 &&
            memcmp(hsym->name, "array", 5) == 0)
        {
            return resolve_typespec_struct(v, tspec->children[1]);
        }
        /* vec / mat element types are scalars, so no struct cycle. */
        /* atomic / ptr — wrappers, but their subjects can't be struct
         * (per spec), so safe to ignore. */
        return NULL;
    }
    return NULL;
}

static void scan_const_struct_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx,
    int want_const, int want_struct);

static void scan_member_attr_value_refs(
    WGSLValidator *v, const WGSLNode *m, EdgeCb cb, void *ctx,
    int want_const, int want_struct);

static void enum_struct_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *sd = s->ast;
    if (!sd) return;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = wgsl_member_attr_count(m);
        if (attrs >= m->child_count) continue;
        WGSLNode *tspec = m->children[attrs];
        WGSLSymbol *target = resolve_typespec_struct(v, tspec);
        if (target) cb(v, target, m, ctx);
        scan_member_attr_value_refs(v, m, cb, ctx, 0, 1);
    }
}

/* — alias cycle: alias RHS type-spec → other aliases — */

static void scan_alias_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && t->kind == WGSL_SYM_TYPE_ALIAS) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_alias_refs(v, n->children[i], cb, ctx);
    }
}

static void enum_alias_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *ad = s->ast;
    if (!ad || ad->child_count == 0) return;
    /* DECL_ALIAS children = [type-spec]; recursive scan handles the
     * full type-spec subtree (incl. template args). */
    scan_alias_refs(v, ad->children[0], cb, ctx);
}

/* — const cycle: init-expr ident → another const symbol — */

static void scan_const_struct_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx,
    int want_const, int want_struct)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && ((want_const && t->kind == WGSL_SYM_CONST) ||
                  (want_struct && t->kind == WGSL_SYM_STRUCT))) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_const_struct_refs(v, n->children[i], cb, ctx,
                               want_const, want_struct);
    }
}

static void scan_member_attr_value_refs(
    WGSLValidator *v, const WGSLNode *m, EdgeCb cb, void *ctx,
    int want_const, int want_struct)
{
    if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) return;
    uint32_t attrs = wgsl_member_attr_count(m);
    for (uint32_t i = 0; i < attrs && i < m->child_count; i++) {
        WGSLNode *attr = m->children[i];
        if (!attr || attr->kind != WGSL_NODE_ATTRIBUTE) continue;
        for (uint32_t a = 1; a < attr->child_count; a++) {
            scan_const_struct_refs(v, attr->children[a], cb, ctx,
                                   want_const, want_struct);
        }
    }
}

static void enum_const_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *cd = s->ast;
    if (!cd) return;
    if (s->kind == WGSL_SYM_STRUCT) {
        for (uint32_t i = 0; i < cd->child_count; i++) {
            scan_member_attr_value_refs(v, cd->children[i], cb, ctx, 1, 1);
        }
        return;
    }
    if (s->kind != WGSL_SYM_CONST) return;
    /* DECL_CONST children = [type? init].  Use payload[1] low bit
     * for has_type. */
    int has_type = wgsl_letlike_has_type(cd);
    if ((uint32_t)(has_type ? 2 : 1) > cd->child_count) return;
    WGSLNode *init = cd->children[has_type ? 1 : 0];
    scan_const_struct_refs(v, init, cb, ctx, 1, 1);
}

/* — override cycle: init-expr ident → another override symbol — */

static void scan_override_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && t->kind == WGSL_SYM_OVERRIDE) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_override_refs(v, n->children[i], cb, ctx);
    }
}

static void enum_override_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *od = s->ast;
    if (!od) return;
    /* DECL_OVERRIDE: [attrs (count A)][type?][init?].  payload[2] =
     * has_type | has_init<<32; payload[1] = attr_count (low). */
    uint32_t A = wgsl_varlike_attr_count(od);
    int has_type = wgsl_varlike_has_type(od);
    int has_init = wgsl_varlike_has_init(od);
    if (!has_init) return;
    uint32_t init_idx = A + (has_type ? 1u : 0u);
    if (init_idx >= od->child_count) return;
    scan_override_refs(v, od->children[init_idx], cb, ctx);
}

/* — module-`var` cycle: init-expr ident → another module-scope `var` — */

static void scan_module_var_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && pick_module_var(t)) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_module_var_refs(v, n->children[i], cb, ctx);
    }
}

static void enum_module_var_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *vd = s->ast;
    if (!vd) return;
    /* DECL_VAR: [attrs (count A)][template args (tpl_count)][type?][init?].
     * payload[1] = attr_count (low) | tpl_count (high);
     * payload[2] = has_type | has_init<<32. */
    uint32_t A   = wgsl_varlike_attr_count(vd);
    uint32_t TPL = wgsl_varlike_tpl_count(vd);
    int has_type = wgsl_varlike_has_type(vd);
    int has_init = wgsl_varlike_has_init(vd);
    if (!has_init) return;
    uint32_t init_idx = A + TPL + (has_type ? 1u : 0u);
    if (init_idx >= vd->child_count) return;
    scan_module_var_refs(v, vd->children[init_idx], cb, ctx);
}

/* — Cross-kind value cycle: init-expr ident → any of {const, override,
 * module-var}.  Catches mixed-kind cycles (e.g., `const A = B;
 * override B = A;`) that the per-kind passes above skip because each
 * filters its edges to a single symbol kind.  Reports only cycles whose
 * members span ≥ 2 kinds, so it never duplicates a same-kind diag. */

static int pick_value_decl(const WGSLSymbol *s) {
    return s->kind == WGSL_SYM_CONST    ||
           s->kind == WGSL_SYM_OVERRIDE ||
           pick_module_var(s);
}

static void scan_value_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && pick_value_decl(t)) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_value_refs(v, n->children[i], cb, ctx);
    }
}

static void enum_value_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    if (!s || !s->ast) return;
    WGSLNode *d = s->ast;
    switch ((WGSLSymKind)s->kind) {
    case WGSL_SYM_CONST: {
        int has_type = wgsl_letlike_has_type(d);
        if ((uint32_t)(has_type ? 2 : 1) > d->child_count) return;
        scan_value_refs(v, d->children[has_type ? 1 : 0], cb, ctx);
        return;
    }
    case WGSL_SYM_OVERRIDE: {
        uint32_t A = wgsl_varlike_attr_count(d);
        int has_type = wgsl_varlike_has_type(d);
        int has_init = wgsl_varlike_has_init(d);
        if (!has_init) return;
        uint32_t init_idx = A + (has_type ? 1u : 0u);
        if (init_idx >= d->child_count) return;
        scan_value_refs(v, d->children[init_idx], cb, ctx);
        return;
    }
    case WGSL_SYM_VAR: {
        uint32_t A   = wgsl_varlike_attr_count(d);
        uint32_t TPL = wgsl_varlike_tpl_count(d);
        int has_type = wgsl_varlike_has_type(d);
        int has_init = wgsl_varlike_has_init(d);
        if (!has_init) return;
        uint32_t init_idx = A + TPL + (has_type ? 1u : 0u);
        if (init_idx >= d->child_count) return;
        scan_value_refs(v, d->children[init_idx], cb, ctx);
        return;
    }
    default: return;
    }
}

/* Path-tracking DFS specialised for the cross-kind detector.  Reports
 * the cycle only when its members span more than one symbol kind. */
typedef struct {
    Color           *state;
    WGSLSymbol     **path;
    int              path_count;
    int              path_cap;
    int              had_cycle;
} XKindCtx;

static void xkind_visit(WGSLValidator *v, const WGSLSymbol *s, XKindCtx *cx);

static void xkind_edge(
    WGSLValidator *v, const WGSLSymbol *to, const WGSLNode *at, void *ctx)
{
    XKindCtx *cx = (XKindCtx *)ctx;
    int idx = wgsl_val_sym_index(v, to);
    if (idx < 0) return;

    if (cx->state[idx] == COLOR_GRAY) {
        /* Back-edge — extract the cycle slice from path[i..top] and
         * gather its kinds.  Suppress single-kind cycles (per-kind
         * pass already reported them). */
        uint32_t mask = (1u << to->kind);
        for (int i = cx->path_count - 1; i >= 0; i--) {
            mask |= (1u << cx->path[i]->kind);
            if (cx->path[i] == to) break;
        }
        int distinct = 0;
        for (int b = 0; b < 32; b++) if (mask & (1u << b)) distinct += 1;
        if (distinct >= 2) {
            wgsl_val_error(v, at,
                "cross-kind value cycle: '%.*s' is referenced through "
                "a back-edge",
                (int)to->name_len, to->name);
            cx->had_cycle = 1;
        }
        return;
    }
    if (cx->state[idx] == COLOR_WHITE) {
        xkind_visit(v, to, cx);
    }
}

static void xkind_visit(WGSLValidator *v, const WGSLSymbol *s, XKindCtx *cx) {
    int idx = wgsl_val_sym_index(v, s);
    if (idx < 0) return;
    cx->state[idx] = COLOR_GRAY;
    if (cx->path_count == cx->path_cap) {
        if (cx->path_cap > INT32_MAX / 2) {
            cx->state[idx] = COLOR_BLACK;
            return;
        }
        int ncap = cx->path_cap ? cx->path_cap * 2 : 16;
        if ((size_t)ncap > SIZE_MAX / sizeof *cx->path) {
            cx->state[idx] = COLOR_BLACK;
            return;
        }
        WGSLSymbol **np = (WGSLSymbol **)realloc(
            cx->path, (size_t)ncap * sizeof *np);
        if (!np) { cx->state[idx] = COLOR_BLACK; return; }
        cx->path = np;
        cx->path_cap = ncap;
    }
    cx->path[cx->path_count++] = (WGSLSymbol *)s;
    enum_value_edges(v, s, xkind_edge, cx);
    cx->path_count -= 1;
    cx->state[idx] = COLOR_BLACK;
}

static void run_xkind_value_cycle(WGSLValidator *v) {
    if (!v->res) return;
    size_t n = v->res->all_decl_count;
    Color *state = (Color *)calloc(n, sizeof *state);
    if (!state) return;
    XKindCtx cx = { state, NULL, 0, 0, 0 };
    for (size_t i = 0; i < n; i++) {
        WGSLSymbol *s = v->res->all_decls[i];
        if (!pick_value_decl(s)) continue;
        if (state[i] != COLOR_WHITE) continue;
        xkind_visit(v, s, &cx);
    }
    free(cx.path);
    free(state);
}


/* Public entry. */

/* Per-decl × stage access bitsets, one for reads and one for writes.
 * §14.5 operation classification distinguishes the two so that, e.g.,
 * a vertex-stage read of `var<storage, read_write>` doesn't false-flag
 * (only writes from vertex are spec violations). */
/* Per-validator entry hooks defined in attrs.c / behavior.c / io.c /
 * access.c / layout.c / uniformity.c — declared in
 * internal/validate_priv.h so the driver can call them directly. */

static WGSLTypeInfo *validator_store_type(WGSLTypeInfo *t) {
    if (t && t->kind == WGSL_TYPE_REF) return (WGSLTypeInfo *)t->ref;
    return t;
}

static const WGSLNode *const_assert_find_nonconst(const WGSLNode *n) {
    if (!n) return n;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return NULL;
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        return s && s->kind == WGSL_SYM_CONST ? NULL : n;
    }
    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 ? const_assert_find_nonconst(n->children[0]) : n;
    case WGSL_NODE_EXPR_BINARY:
        if (n->child_count < 2) return n;
        {
            const WGSLNode *bad = const_assert_find_nonconst(n->children[0]);
            return bad ? bad : const_assert_find_nonconst(n->children[1]);
        }
    case WGSL_NODE_EXPR_CALL:
        if (n->child_count < 1) return 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            const WGSLNode *bad = const_assert_find_nonconst(n->children[i]);
            if (bad) return bad;
        }
        return NULL;
    case WGSL_NODE_EXPR_INDEX: {
        if (n->child_count < 2) return n;
        const WGSLNode *bad = const_assert_find_nonconst(n->children[0]);
        return bad ? bad : const_assert_find_nonconst(n->children[1]);
    }
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count >= 1 ? const_assert_find_nonconst(n->children[0]) : n;
    default:
        return n;
    }
}

static void validate_const_asserts(WGSLValidator *v, WGSLNode *n) {
    if (!n) return;
    if (n->kind == WGSL_NODE_DECL_CONST_ASSERT) {
        if (n->child_count >= 1 && n->children[0]) {
            WGSLTypeInfo *t = validator_store_type(
                wgsl_typecheck_type_of(v->tc, n->children[0]));
            if (t && t == v->types->t_bool) {
                const WGSLNode *bad = const_assert_find_nonconst(n->children[0]);
                if (bad) {
                    WGSLSymbol *s = bad->kind == WGSL_NODE_EXPR_IDENT
                        ? wgsl_node_resolved_symbol(bad) : NULL;
                    if (s && s->name) {
                        wgsl_val_error(v, bad, "'%.*s' is not a constant",
                            (int)s->name_len, s->name);
                    } else {
                        wgsl_val_error(v, bad,
                            "const_assert condition must be a const-expression");
                    }
                } else {
                    WGSLValue val = {0};
                    if (wgsl_consteval_expr(v->cev, n->children[0], &val)) {
                        if (val.kind == WGSL_VAL_BOOL && !val.u.b) {
                            wgsl_val_error(v, n, "const_assert failed");
                        }
                    } else {
                        v->had_error = 1;
                    }
                }
            }
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        validate_const_asserts(v, n->children[i]);
    }
}

int wgsl_validate(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *tc,
    WGSLValidator       *out)
{
    if (!ast || !src || !arena || !diag || !types || !res || !tc || !out) {
        return 0;
    }
    memset(out, 0, sizeof *out);
    out->arena = arena;
    out->diag  = diag;
    out->types = types;
    out->src   = src;
    out->res   = res;
    out->cev   = cev;
    out->tc    = tc;

    if (!ast->root) return 0;

    /* Refuse to run the recursive validators on a pathologically deep AST
     * (e.g. a long `a+a+…` chain or deep block nesting); they would
     * otherwise overflow the stack.  One diagnostic, then skip the pass. */
    {
        const WGSLNode *deep =
            wgsl_ast_first_over_depth(ast->root, WGSL_MAX_AST_DEPTH);
        if (deep) {
            wgsl_val_error(out, deep,
                "expression or statement nested too deeply (max %d) — "
                "validation skipped", WGSL_MAX_AST_DEPTH);
            return 0;
        }
    }

    wgsl_val_run_cycle_check(out, pick_function,   wgsl_val_enum_call_edges,        "recursion");
    wgsl_val_run_cycle_check(out, pick_struct,     enum_struct_edges,      "struct cycle");
    wgsl_val_run_cycle_check(out, pick_alias,      enum_alias_edges,       "alias cycle");
    wgsl_val_run_cycle_check(out, pick_const,      enum_const_edges,       "const cycle");
    wgsl_val_run_cycle_check(out, pick_override,   enum_override_edges,    "override cycle");
    wgsl_val_run_cycle_check(out, pick_module_var, enum_module_var_edges,  "module var cycle");
    run_xkind_value_cycle(out);

    validate_const_asserts (out, ast->root);
    validate_attributes    (out, ast->root);
    validate_behavior      (out, ast->root);
    validate_entry_points  (out, ast->root);
    AccessTables at = analyze_static_access(out);
    validate_address_spaces(out, ast->root, at.read_table, at.write_table);
    validate_alias_analysis(out);
    validate_resource_bindings(out, ast->root);
    validate_layouts       (out, ast->root);
    validate_discard_stages(out);
    validate_uniformity    (out, ast->root);

    free(at.read_table);
    free(at.write_table);
    return out->had_error ? 0 : 1;
}

void wgsl_validator_destroy(WGSLValidator *v) {
    if (!v) return;
    memset(v, 0, sizeof *v);
}


WGSLSymbol *wgsl_val_find_symbol_for_ast(const WGSLValidator *v, const WGSLNode *ast) {
    if (!v || !v->res || !ast) return NULL;
    return wgsl_resolver_symbol_for_decl(v->res, ast);
}
