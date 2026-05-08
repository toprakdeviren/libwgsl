/**
 * @file validate.c — WGSL validator (Phase 8).
 *
 * Iter A — cycle detection family (recursion / struct / alias / const /
 *          override) via shared DFS coloring engine.
 * Iter B — attribute conformance (placement + arity, table-driven) +
 *          behavior-set analysis (§9.7).
 * Iter C — entry-point shape rules (§13) + address-space conformance
 *          subset (§14.3: uniform/storage `var` requires @group +
 *          @binding; at most one stage attribute per fn).
 *
 * v1 deferrals (DESIGN.md): host-shareable layout (§14.4), uniformity
 * analysis (§15.2 — explicit multi-week task per risk register), and
 * alias analysis (§11.4.1).
 */
#include "internal/validate.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Diagnostics ───────────────────────────────────────────────────── */

static void val_error(
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

/* ── Symbol → all_decls index ──────────────────────────────────────── */

static int sym_index(const WGSLValidator *v, const WGSLSymbol *s) {
    if (!v->res) return -1;
    for (size_t i = 0; i < v->res->all_decl_count; i++) {
        if (v->res->all_decls[i] == s) return (int)i;
    }
    return -1;
}

/* ── Generic cycle DFS ─────────────────────────────────────────────── */

typedef enum { COLOR_WHITE = 0, COLOR_GRAY, COLOR_BLACK } Color;

/* `enumerate_edges`: for the given `from` symbol, invoke `out_cb` once
 * per outgoing edge with the target symbol and a representative AST
 * node (used for diagnostic spans).  The DFS engine is opaque to the
 * concrete edge shape. */
typedef void (*EdgeCb)(
    WGSLValidator *v,
    const WGSLSymbol *to, const WGSLNode *at, void *ctx);

typedef void (*EdgeEnum)(
    WGSLValidator *v,
    const WGSLSymbol *from, EdgeCb cb, void *ctx);

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
    int idx = sym_index(v, to);
    if (idx < 0) return;

    if (cx->state[idx] == COLOR_GRAY) {
        val_error(v, at,
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
    int idx = sym_index(v, s);
    if (idx < 0) return;
    cx->state[idx] = COLOR_GRAY;
    cx->edges(v, s, dfs_edge, cx);
    cx->state[idx] = COLOR_BLACK;
}

static int run_cycle_check(
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

/* ── Edge enumerators ──────────────────────────────────────────────── */

static int pick_function(const WGSLSymbol *s) { return s->kind == WGSL_SYM_FUNCTION; }
static int pick_struct  (const WGSLSymbol *s) { return s->kind == WGSL_SYM_STRUCT;   }
static int pick_alias   (const WGSLSymbol *s) { return s->kind == WGSL_SYM_TYPE_ALIAS; }
static int pick_const   (const WGSLSymbol *s) { return s->kind == WGSL_SYM_CONST;    }
static int pick_override(const WGSLSymbol *s) { return s->kind == WGSL_SYM_OVERRIDE; }

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

static void enum_call_edges(
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
        /* vec / mat element types are scalars in v1 — no struct.       */
        /* atomic / ptr — wrappers, but their subjects can't be struct
         * (per spec), so safe to ignore. */
        return NULL;
    }
    return NULL;
}

static void enum_struct_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *sd = s->ast;
    if (!sd) return;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
        if (attrs >= m->child_count) continue;
        WGSLNode *tspec = m->children[attrs];
        WGSLSymbol *target = resolve_typespec_struct(v, tspec);
        if (target) cb(v, target, m, ctx);
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

static void scan_const_refs(
    WGSLValidator *v, WGSLNode *n, EdgeCb cb, void *ctx)
{
    if (!n) return;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *t = wgsl_node_resolved_symbol(n);
        if (t && t->kind == WGSL_SYM_CONST) {
            cb(v, t, n, ctx);
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        scan_const_refs(v, n->children[i], cb, ctx);
    }
}

static void enum_const_edges(
    WGSLValidator *v, const WGSLSymbol *s, EdgeCb cb, void *ctx)
{
    WGSLNode *cd = s->ast;
    if (!cd) return;
    /* DECL_CONST children = [type? init].  Use payload[1] low bit
     * for has_type. */
    int has_type = (int)(cd->payload[1] & 1u);
    if ((uint32_t)(has_type ? 2 : 1) > cd->child_count) return;
    WGSLNode *init = cd->children[has_type ? 1 : 0];
    scan_const_refs(v, init, cb, ctx);
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
    uint32_t A = (uint32_t)(od->payload[1] & 0xFFFFFFFFu);
    int has_type = (int)(od->payload[2] & 1u);
    int has_init = (int)(od->payload[2] >> 32);
    if (!has_init) return;
    uint32_t init_idx = A + (has_type ? 1u : 0u);
    if (init_idx >= od->child_count) return;
    scan_override_refs(v, od->children[init_idx], cb, ctx);
}

/* ── Attribute conformance (§12) ───────────────────────────────────── */

/* Bitmask of contexts an attribute may legally appear in. */
enum {
    AC_FN_DECL       = 1u << 0,   /* `fn` declaration                       */
    AC_FN_PARAM      = 1u << 1,   /* function parameter                     */
    AC_FN_RETURN     = 1u << 2,   /* function return type slot              */
    AC_VAR_MODULE    = 1u << 3,   /* module-scope `var<…>` decl             */
    AC_OVERRIDE      = 1u << 4,   /* `override` decl                        */
    AC_STRUCT_MEMBER = 1u << 5,   /* struct member                          */
    AC_MODULE        = 1u << 6,   /* module-level (e.g., `@diagnostic`)     */
};

typedef struct {
    const char *name;
    int         min_args;
    int         max_args;
    uint32_t    contexts;
} AttrEntry;

/* Per WGSL §12.  Attribute names are context-keyed identifiers, not
 * keywords — the parser leaves them as IDENTs.  Entries cover every
 * §12.1–12.15 attribute. */
static const AttrEntry kAttrTable[] = {
    /* Stage attributes (§12.15). */
    { "vertex",         0, 0, AC_FN_DECL },
    { "fragment",       0, 0, AC_FN_DECL },
    { "compute",        0, 0, AC_FN_DECL },

    /* Compute (§12.14). */
    { "workgroup_size", 1, 3, AC_FN_DECL },

    /* Resources (§12.4 / §12.5). */
    { "binding",        1, 1, AC_VAR_MODULE },
    { "group",          1, 1, AC_VAR_MODULE },

    /* Entry-point IO (§12.7 / §12.6 / §12.10 / §12.11 / §12.13). */
    { "location",       1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "builtin",        1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "interpolate",    1, 2, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "invariant",      0, 0, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },
    { "blend_src",      1, 1, AC_FN_PARAM | AC_FN_RETURN | AC_STRUCT_MEMBER },

    /* Layout (§12.1 / §12.12). */
    { "align",          1, 1, AC_STRUCT_MEMBER },
    { "size",           1, 1, AC_STRUCT_MEMBER },

    /* Function (§12.2 / §12.9). */
    { "must_use",       0, 0, AC_FN_DECL },
    { "const",          0, 0, AC_FN_DECL },

    /* Override (§12.8). */
    { "id",             1, 1, AC_OVERRIDE },

    /* Diagnostic filter (§4.2). */
    { "diagnostic",     2, 3, AC_MODULE | AC_FN_DECL },
};

static const AttrEntry *attr_lookup(
    const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < sizeof kAttrTable / sizeof kAttrTable[0]; i++) {
        const AttrEntry *e = &kAttrTable[i];
        size_t l = strlen(e->name);
        if (l == name_len && memcmp(e->name, name, l) == 0) return e;
    }
    return NULL;
}

static const char *ctx_name(uint32_t bit) {
    switch (bit) {
    case AC_FN_DECL:       return "function declarations";
    case AC_FN_PARAM:      return "function parameters";
    case AC_FN_RETURN:     return "function return types";
    case AC_VAR_MODULE:    return "module-scope `var` declarations";
    case AC_OVERRIDE:      return "`override` declarations";
    case AC_STRUCT_MEMBER: return "struct members";
    case AC_MODULE:        return "the module level";
    default:               return "this context";
    }
}

/* Validate one ATTRIBUTE node found in `ctx_bit` placement. */
static void check_attr(
    WGSLValidator *v, WGSLNode *attr, uint32_t ctx_bit)
{
    if (!attr || attr->child_count == 0) return;
    WGSLNode *name_node = attr->children[0];
    if (!name_node) return;
    uint32_t off = (uint32_t)(name_node->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(name_node->payload[0] >> 32);
    const char *name = v->src->bytes + off;

    const AttrEntry *e = attr_lookup(name, len);
    if (!e) {
        val_error(v, attr,
            "unknown attribute '@%.*s'", (int)len, name);
        return;
    }
    if ((e->contexts & ctx_bit) == 0) {
        val_error(v, attr,
            "attribute '@%.*s' is not allowed on %s",
            (int)len, name, ctx_name(ctx_bit));
        return;
    }
    /* Argument-count check: child[0] is the name; rest are args. */
    int argc = (int)attr->child_count - 1;
    if (argc < e->min_args || argc > e->max_args) {
        val_error(v, attr,
            "attribute '@%.*s' expects %d-%d arguments, got %d",
            (int)len, name, e->min_args, e->max_args, argc);
        return;
    }
}

/* Walk every attribute on a decl-shaped node and check it against
 * `ctx_bit`.  Layout-driven helpers know which children are attrs. */
static void check_attrs_range(
    WGSLValidator *v, WGSLNode *parent,
    uint32_t first, uint32_t count, uint32_t ctx_bit)
{
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (a && a->kind == WGSL_NODE_ATTRIBUTE) {
            check_attr(v, a, ctx_bit);
        }
    }
}

static void validate_attributes(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        switch ((WGSLNodeKind)n->kind) {

        case WGSL_NODE_DECL_FUNCTION: {
            uint32_t FA = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            uint32_t P  = (uint32_t)(n->payload[1] >> 32);
            uint32_t RA = (uint32_t)(n->payload[2] & 0xFFFFFFFFu);
            check_attrs_range(v, n, 0,        FA, AC_FN_DECL);
            /* Return-attrs sit between params and (optional) ret_type. */
            check_attrs_range(v, n, FA + P,   RA, AC_FN_RETURN);

            /* Per-param attrs. */
            for (uint32_t k = 0; k < P; k++) {
                uint32_t pi = FA + k;
                if (pi >= n->child_count) break;
                WGSLNode *p = n->children[pi];
                if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
                uint32_t PA = (uint32_t)(p->payload[1] & 0xFFFFFFFFu);
                check_attrs_range(v, p, 0, PA, AC_FN_PARAM);
            }
            break;
        }

        case WGSL_NODE_DECL_VAR: {
            uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            check_attrs_range(v, n, 0, A, AC_VAR_MODULE);
            break;
        }

        case WGSL_NODE_DECL_OVERRIDE: {
            uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
            check_attrs_range(v, n, 0, A, AC_OVERRIDE);
            break;
        }

        case WGSL_NODE_DECL_STRUCT: {
            for (uint32_t k = 0; k < n->child_count; k++) {
                WGSLNode *m = n->children[k];
                if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
                uint32_t MA = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
                check_attrs_range(v, m, 0, MA, AC_STRUCT_MEMBER);
            }
            break;
        }

        case WGSL_NODE_DIR_DIAGNOSTIC:
            /* `diagnostic(...)` directive is a top-level attribute-shape
             * node (different from `@diagnostic` attributes), but
             * placement-wise behaves the same. */
            break;

        default:
            break;
        }
    }
}

/* ── Behavior-set analysis (§9.7) ──────────────────────────────────── */

typedef uint8_t Behavior;

static Behavior beh_stmt(WGSLValidator *v, WGSLNode *n);

static Behavior beh_compound(WGSLValidator *v, WGSLNode *n) {
    Behavior b = WGSL_BEHAVIOR_NEXT;
    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *c = n->children[i];
        if (!c) continue;
        if ((b & WGSL_BEHAVIOR_NEXT) == 0) {
            /* Unreachable code — Phase 8 is permissive in v1. */
            break;
        }
        Behavior s = beh_stmt(v, c);
        b = (Behavior)((b & ~WGSL_BEHAVIOR_NEXT) | s);
    }
    return b;
}

static Behavior beh_stmt(WGSLValidator *v, WGSLNode *n) {
    if (!n) return WGSL_BEHAVIOR_NEXT;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_STMT_RETURN:   return WGSL_BEHAVIOR_RETURN;
    case WGSL_NODE_STMT_BREAK:    return WGSL_BEHAVIOR_BREAK;
    case WGSL_NODE_STMT_CONTINUE: return WGSL_BEHAVIOR_CONTINUE;
    case WGSL_NODE_STMT_DISCARD:  return WGSL_BEHAVIOR_NEXT;

    case WGSL_NODE_STMT_COMPOUND: return beh_compound(v, n);

    case WGSL_NODE_STMT_IF: {
        /* children: [cond, then-compound, (else_if|else)*] */
        Behavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            if (c->kind == WGSL_NODE_STMT_COMPOUND ||
                c->kind == WGSL_NODE_STMT_ELSE_IF ||
                c->kind == WGSL_NODE_STMT_ELSE)
            {
                result = (Behavior)(result | beh_stmt(v, c));
            }
        }
        if (!saw_else) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        return result;
    }
    case WGSL_NODE_STMT_ELSE_IF: {
        /* children: [cond, then-compound, ...] — same as STMT_IF but
         * already inside an `if` chain. */
        Behavior result = 0;
        int saw_else = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind == WGSL_NODE_STMT_ELSE) saw_else = 1;
            if (c->kind == WGSL_NODE_STMT_COMPOUND ||
                c->kind == WGSL_NODE_STMT_ELSE_IF ||
                c->kind == WGSL_NODE_STMT_ELSE)
            {
                result = (Behavior)(result | beh_stmt(v, c));
            }
        }
        if (!saw_else) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        return result;
    }
    case WGSL_NODE_STMT_ELSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return beh_stmt(v, c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }

    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_LOOP: {
        /* Loops always fall through to Next (zero-iteration / break-
         * exit).  Return propagates if the body has it. */
        Behavior body = 0;
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                body = (Behavior)(body | beh_stmt(v, c));
            }
        }
        Behavior r = WGSL_BEHAVIOR_NEXT;
        if (body & WGSL_BEHAVIOR_RETURN) r |= WGSL_BEHAVIOR_RETURN;
        return r;
    }

    case WGSL_NODE_STMT_SWITCH: {
        /* children: [selector, case_clauses...] */
        Behavior result = 0;
        int has_default = 0;
        for (uint32_t i = 1; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c || c->kind != WGSL_NODE_STMT_CASE_CLAUSE) continue;
            int is_default_alone = (int)((c->payload[0] >> 32) & 1u);
            if (is_default_alone) has_default = 1;
            result = (Behavior)(result | beh_stmt(v, c));
        }
        if (!has_default) result = (Behavior)(result | WGSL_BEHAVIOR_NEXT);
        /* Switch absorbs Break into Next (a `break` falls through past
         * the switch). */
        if (result & WGSL_BEHAVIOR_BREAK) {
            result = (Behavior)((result & ~WGSL_BEHAVIOR_BREAK) | WGSL_BEHAVIOR_NEXT);
        }
        return result;
    }
    case WGSL_NODE_STMT_CASE_CLAUSE: {
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (c && c->kind == WGSL_NODE_STMT_COMPOUND) {
                return beh_stmt(v, c);
            }
        }
        return WGSL_BEHAVIOR_NEXT;
    }

    case WGSL_NODE_STMT_BREAK_IF:    return WGSL_BEHAVIOR_BREAK | WGSL_BEHAVIOR_NEXT;
    case WGSL_NODE_STMT_CONTINUING:  return WGSL_BEHAVIOR_NEXT;

    /* Plain statements continue to next. */
    default:
        return WGSL_BEHAVIOR_NEXT;
    }
}

/* Validate every user fn's body satisfies §9.7's body constraint:
 *   behaviors ⊆ {Next, Return}; non-void fn must have Return without
 *   Next (i.e. all paths return). */
static void validate_behavior(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t HR = (uint32_t)(fn->payload[2] >> 32);
        if (fn->child_count == 0) continue;
        WGSLNode *body = fn->children[fn->child_count - 1];
        if (!body || body->kind != WGSL_NODE_STMT_COMPOUND) continue;

        Behavior b = beh_stmt(v, body);

        if (b & WGSL_BEHAVIOR_BREAK) {
            val_error(v, body,
                "function body must not have `break` outside a loop or switch");
            continue;
        }
        if (b & WGSL_BEHAVIOR_CONTINUE) {
            val_error(v, body,
                "function body must not have `continue` outside a loop");
            continue;
        }
        if (HR && (b & WGSL_BEHAVIOR_NEXT)) {
            const char *fname = "<anonymous>";
            uint32_t fname_len = 11;
            uint32_t no = (uint32_t)(fn->payload[0] & 0xFFFFFFFFu);
            uint32_t nl = (uint32_t)(fn->payload[0] >> 32);
            if (nl > 0) {
                fname = v->src->bytes + no;
                fname_len = nl;
            }
            val_error(v, body,
                "function '%.*s' has a return type but does not return on all paths",
                (int)fname_len, fname);
        }
    }
}

/* ── Entry-point shape (§13) ───────────────────────────────────────── */

/* Look up an attribute by name within `[first..first+count)` of
 * `parent->children`.  Returns NULL when not found. */
static WGSLNode *find_attr(
    const WGSLValidator *v, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = (uint32_t)(nm->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(nm->payload[0] >> 32);
        if (len == wl && memcmp(v->src->bytes + off, want, wl) == 0) {
            return a;
        }
    }
    return NULL;
}

static void validate_entry_points(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *fn = tu->children[i];
        if (!fn || fn->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t FA = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);

        WGSLNode *vx  = find_attr(v, fn, 0, FA, "vertex");
        WGSLNode *fr  = find_attr(v, fn, 0, FA, "fragment");
        WGSLNode *co  = find_attr(v, fn, 0, FA, "compute");
        WGSLNode *wgs = find_attr(v, fn, 0, FA, "workgroup_size");

        int stages = (vx ? 1 : 0) + (fr ? 1 : 0) + (co ? 1 : 0);
        if (stages > 1) {
            val_error(v, fn,
                "function declares more than one shader-stage attribute");
            continue;
        }

        if (stages == 0) {
            /* Plain helper function — `@workgroup_size` is illegal here. */
            if (wgs) {
                val_error(v, wgs,
                    "@workgroup_size is only valid on `@compute` entry points");
            }
            continue;
        }

        v->entry_point_count += 1;

        if (co && !wgs) {
            val_error(v, fn,
                "`@compute` entry point requires a `@workgroup_size` attribute");
        }
        if ((vx || fr) && wgs) {
            val_error(v, wgs,
                "@workgroup_size is only valid on `@compute` entry points");
        }
    }
}

/* ── Address-space conformance (§14.3) ─────────────────────────────── */

static int ident_text_eq(
    const WGSLValidator *v, const WGSLNode *id, const char *want)
{
    if (!id || id->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = (uint32_t)(id->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(id->payload[0] >> 32);
    size_t wl = strlen(want);
    if (len != wl) return 0;
    return memcmp(v->src->bytes + off, want, wl) == 0;
}

static void validate_address_spaces(WGSLValidator *v, WGSLNode *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;

        uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        uint32_t T = (uint32_t)(n->payload[1] >> 32);
        if (T == 0) continue;     /* No address-space template — implicit. */

        /* Template list: children[A..A+T) — first arg is address space. */
        if (A >= n->child_count) continue;
        WGSLNode *as_node = n->children[A];
        if (!as_node) continue;

        int is_uniform = ident_text_eq(v, as_node, "uniform");
        int is_storage = ident_text_eq(v, as_node, "storage");
        if (!is_uniform && !is_storage) continue;

        /* Uniform / storage vars require both @group and @binding. */
        WGSLNode *gp = find_attr(v, n, 0, A, "group");
        WGSLNode *bd = find_attr(v, n, 0, A, "binding");
        const char *as_name = is_uniform ? "uniform" : "storage";
        if (!gp) {
            val_error(v, n,
                "`var<%s, …>` requires a `@group` attribute",
                as_name);
        }
        if (!bd) {
            val_error(v, n,
                "`var<%s, …>` requires a `@binding` attribute",
                as_name);
        }
    }
}

/* ── Public entry ──────────────────────────────────────────────────── */

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

    run_cycle_check(out, pick_function, enum_call_edges,     "recursion");
    run_cycle_check(out, pick_struct,   enum_struct_edges,   "struct cycle");
    run_cycle_check(out, pick_alias,    enum_alias_edges,    "alias cycle");
    run_cycle_check(out, pick_const,    enum_const_edges,    "const cycle");
    run_cycle_check(out, pick_override, enum_override_edges, "override cycle");

    validate_attributes    (out, ast->root);
    validate_behavior      (out, ast->root);
    validate_entry_points  (out, ast->root);
    validate_address_spaces(out, ast->root);

    return out->had_error ? 0 : 1;
}

void wgsl_validator_destroy(WGSLValidator *v) {
    if (!v) return;
    memset(v, 0, sizeof *v);
}
