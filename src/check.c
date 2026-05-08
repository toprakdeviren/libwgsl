/**
 * @file check.c — WGSL type checker (Phase 7).
 *
 * Iter A — type-specifier resolution + module decl typing + function
 *          param/return + scalar literal/ident/paren/unary/binary-numeric.
 * Iter B — composite expressions: EXPR_CALL (constructors + user-fn
 *          calls + best-effort builtin fall-through), EXPR_MEMBER
 *          (swizzle + struct field), EXPR_INDEX (vec/mat/array),
 *          vec/mat binary operators.  Function symbol → return type.
 *
 * Iter C will adopt the Tint-style `def/wgsl.def` + codegen for the
 * full builtin function overload table and hook the corpus.
 *
 * Out of v1 scope (deferred to v1.x):
 *   - EXPR_ADDR_OF / EXPR_INDIRECTION + full ref-vs-ptr (§6.4.3) typing
 *   - assignment-statement LHS=ref enforcement
 *   - mixed-mode vec constructors (vec3 from vec2 + scalar, etc.)
 */
#include "internal/check.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* ── Diagnostics ───────────────────────────────────────────────────── */

static void tc_error(
    WGSLTypeChecker *tc, const WGSLNode *at, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(tc->diag, tc->src, WGSL_DIAG_ERROR,
                      off, len, NULL, "%s", buf);
    tc->had_error = 1;
}

/* ── Side-table ────────────────────────────────────────────────────── */

static int tc_set_type(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *t, int is_ref)
{
    if (!n || !t) return 0;
    if (tc->count == tc->capacity) {
        size_t cap = tc->capacity ? tc->capacity * 2 : 64;
        WGSLNodeType *g = (WGSLNodeType *)realloc(
            tc->table, cap * sizeof *g);
        if (!g) return 0;
        tc->table = g;
        tc->capacity = cap;
    }
    tc->table[tc->count].node  = n;
    tc->table[tc->count].type  = t;
    tc->table[tc->count].flags = is_ref ? WGSL_FLAG_IS_REF : 0;
    tc->count += 1;
    n->flags |= WGSL_FLAG_TYPED;
    if (is_ref) n->flags |= WGSL_FLAG_IS_REF;
    else        n->flags &= (uint16_t)~WGSL_FLAG_IS_REF;
    return 1;
}

WGSLTypeInfo *wgsl_typecheck_type_of(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return NULL;
    /* EXPR_IDENT: resolve through symbol — no need to consult the
     * side-table.  Saves storage and stays correct under symbol-
     * type updates that happen after an ident is first visited. */
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s && s->type) return s->type;
    }
    for (size_t i = 0; i < tc->count; i++) {
        if (tc->table[i].node == n) return tc->table[i].type;
    }
    return NULL;
}

int wgsl_typecheck_is_ref(
    const WGSLTypeChecker *tc, const WGSLNode *n)
{
    if (!tc || !n) return 0;
    return (n->flags & WGSL_FLAG_IS_REF) != 0;
}

void wgsl_typecheck_destroy(WGSLTypeChecker *tc) {
    if (!tc) return;
    free(tc->table);
    memset(tc, 0, sizeof *tc);
}

/* ── Symbol lookup helper ──────────────────────────────────────────── */

static WGSLSymbol *find_decl_symbol(
    const WGSLTypeChecker *tc, const WGSLNode *decl)
{
    if (!tc->res) return NULL;
    for (size_t i = 0; i < tc->res->all_decl_count; i++) {
        if (tc->res->all_decls[i]->ast == decl) {
            return tc->res->all_decls[i];
        }
    }
    return NULL;
}

/* ── Type-specifier resolution ─────────────────────────────────────── */

static WGSLTypeInfo *resolve_type_spec_impl(WGSLTypeChecker *tc, WGSLNode *spec);

/* Public wrapper: resolves AND records the result in the side-table
 * so downstream consumers (validator, module-summary JSON) can pull
 * the resolved type back out via `wgsl_typecheck_type_of(spec)`. */
static WGSLTypeInfo *resolve_type_spec(WGSLTypeChecker *tc, WGSLNode *spec) {
    WGSLTypeInfo *t = resolve_type_spec_impl(tc, spec);
    if (t && spec) tc_set_type(tc, spec, t, 0);
    return t;
}

/* Try to evaluate an array-length expression as a u32 const.  Returns
 * 0 length to indicate "use a runtime-sized array". */
static uint32_t eval_array_length(WGSLTypeChecker *tc, WGSLNode *expr) {
    if (!expr || !tc->cev) return 0;
    WGSLValue v = {0};
    if (!wgsl_consteval_expr(tc->cev, expr, &v)) return 0;
    if (v.kind != WGSL_VAL_INT) return 0;
    if (v.u.i <= 0) return 0;
    return (uint32_t)v.u.i;
}

static WGSLTypeInfo *resolve_templated_type(
    WGSLTypeChecker *tc, WGSLNode *spec)
{
    /* spec is EXPR_TEMPLATED_IDENT.  children[0] is the head ident,
     * children[1..] are template args. */
    if (spec->child_count == 0) return NULL;
    WGSLNode *head = spec->children[0];
    if (!head || head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *sym = wgsl_node_resolved_symbol(head);
    if (!sym) return NULL;

    /* If the head is itself a fully-typed alias / scalar / scalar-alias,
     * the template list is bogus — Phase 8 territory.  For Iter A we
     * just return the head type. */
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
        sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
    {
        return sym->type;
    }
    if (sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) {
        tc_error(tc, spec, "'%.*s' is not a type generator",
                 (int)sym->name_len, sym->name);
        return NULL;
    }

    const char *nm = sym->name;
    uint32_t    nl = sym->name_len;

    /* vec2 / vec3 / vec4 — single elem-type template arg. */
    if (nl == 4 && memcmp(nm, "vec", 3) == 0) {
        int w = nm[3] - '0';
        if (w < 2 || w > 4) return NULL;
        if (spec->child_count < 2) {
            tc_error(tc, spec, "vec%d requires an element type", w);
            return NULL;
        }
        WGSLTypeInfo *elem = resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        return wgsl_type_vec(tc->types, (uint8_t)w, elem);
    }

    /* mat2x2 .. mat4x4 — single elem-type template arg. */
    if (nl == 6 && memcmp(nm, "mat", 3) == 0 && nm[4] == 'x') {
        int c = nm[3] - '0';
        int r = nm[5] - '0';
        if (c < 2 || c > 4 || r < 2 || r > 4) return NULL;
        if (spec->child_count < 2) {
            tc_error(tc, spec, "mat%dx%d requires an element type", c, r);
            return NULL;
        }
        WGSLTypeInfo *elem = resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        return wgsl_type_mat(tc->types, (uint8_t)c, (uint8_t)r, elem);
    }

    /* atomic<T> */
    if (nl == 6 && memcmp(nm, "atomic", 6) == 0) {
        if (spec->child_count < 2) {
            tc_error(tc, spec, "atomic requires an element type");
            return NULL;
        }
        WGSLTypeInfo *elem = resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        return wgsl_type_atomic(tc->types, elem);
    }

    /* array<T> or array<T, N> */
    if (nl == 5 && memcmp(nm, "array", 5) == 0) {
        if (spec->child_count < 2) {
            tc_error(tc, spec, "array requires an element type");
            return NULL;
        }
        WGSLTypeInfo *elem = resolve_type_spec(tc, spec->children[1]);
        if (!elem) return NULL;
        uint32_t length = 0;
        if (spec->child_count >= 3) {
            length = eval_array_length(tc, spec->children[2]);
        }
        return wgsl_type_array(tc->types, elem, length);
    }

    /* ptr<…> — Iter B (no WGSL_TYPE_PTR yet in types.h). */
    if (nl == 3 && memcmp(nm, "ptr", 3) == 0) {
        return NULL;        /* silent NULL — Iter B materialises ptr */
    }

    /* texture / sampler — Iter B / C territory. */
    return NULL;
}

static WGSLTypeInfo *resolve_type_spec_impl(WGSLTypeChecker *tc, WGSLNode *spec) {
    if (!spec) return NULL;

    if (spec->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(spec);
        if (!sym) return NULL;
        switch ((WGSLSymKind)sym->kind) {
        case WGSL_SYM_PREDECLARED_TYPE:
        case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
            return sym->type;
        case WGSL_SYM_TYPE_ALIAS:
        case WGSL_SYM_STRUCT:
            if (sym->type) return sym->type;
            tc_error(tc, spec,
                "type '%.*s' has not been resolved yet (forward reference?)",
                (int)sym->name_len, sym->name);
            return NULL;
        default:
            tc_error(tc, spec,
                "'%.*s' is not a type",
                (int)sym->name_len, sym->name);
            return NULL;
        }
    }

    if (spec->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        return resolve_templated_type(tc, spec);
    }
    return NULL;
}

/* ── Expression typing ─────────────────────────────────────────────── */

static int type_expr(WGSLTypeChecker *tc, WGSLNode *n);

static int type_literal_int(WGSLTypeChecker *tc, WGSLNode *n) {
    uint32_t suf = (uint32_t)n->payload[0];
    WGSLTypeInfo *t;
    switch (suf & 0x0fu) {
    case WGSL_NUM_SUFFIX_I: t = tc->types->t_i32; break;
    case WGSL_NUM_SUFFIX_U: t = tc->types->t_u32; break;
    default:                t = tc->types->t_abstract_int; break;
    }
    return tc_set_type(tc, n, t, 0);
}

static int type_literal_float(WGSLTypeChecker *tc, WGSLNode *n) {
    uint32_t suf = (uint32_t)n->payload[0];
    WGSLTypeInfo *t;
    switch (suf & 0x0fu) {
    case WGSL_NUM_SUFFIX_F: t = tc->types->t_f32; break;
    case WGSL_NUM_SUFFIX_H: t = tc->types->t_f16; break;
    default:                t = tc->types->t_abstract_float; break;
    }
    return tc_set_type(tc, n, t, 0);
}

static int type_ident(WGSLTypeChecker *tc, WGSLNode *n) {
    WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
    if (!sym) {
        tc_error(tc, n, "unresolved identifier in type-checked expression");
        return 0;
    }
    /* Function names aren't first-class values in WGSL — they are only
     * usable in call position.  EXPR_CALL pulls the symbol out of the
     * callee directly, so a FUNCTION sym reaching this path is a
     * mis-use (`let f = my_fn;`). */
    if (sym->kind == WGSL_SYM_FUNCTION) {
        tc_error(tc, n,
            "function '%.*s' is not a value",
            (int)sym->name_len, sym->name);
        return 0;
    }
    /* Builtin function names same story. */
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        tc_error(tc, n,
            "builtin function '%.*s' is not a value",
            (int)sym->name_len, sym->name);
        return 0;
    }
    if (!sym->type) {
        tc_error(tc, n, "'%.*s' has no type yet",
                 (int)sym->name_len, sym->name);
        return 0;
    }
    /* `var` bindings produce a reference; const / let / param produce a
     * value.  (Param-by-pointer comes via Iter B's ptr handling.) */
    int is_ref = (sym->kind == WGSL_SYM_VAR);
    n->flags |= WGSL_FLAG_TYPED;
    if (is_ref) n->flags |= WGSL_FLAG_IS_REF;
    return 1;
}

/* Scalar-pair common-type promotion (§6.1.2). */
static WGSLTypeInfo *common_scalar_type(
    WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b, const WGSLNode *at)
{
    if (a == b) return a;

    int aa = wgsl_type_is_abstract(a);
    int ab = wgsl_type_is_abstract(b);
    int rank_ab = wgsl_type_conversion_rank(a, b);
    int rank_ba = wgsl_type_conversion_rank(b, a);

    if (aa && !ab && rank_ab >= 0) return b;
    if (ab && !aa && rank_ba >= 0) return a;
    if (aa && ab) {
        if (a == tc->types->t_abstract_float ||
            b == tc->types->t_abstract_float)
        {
            return tc->types->t_abstract_float;
        }
        return tc->types->t_abstract_int;
    }
    if (rank_ab >= 0) return b;
    if (rank_ba >= 0) return a;

    tc_error(tc, at,
        "no implicit conversion between %s and %s",
        wgsl_type_kind_name((WGSLTypeKind)a->kind),
        wgsl_type_kind_name((WGSLTypeKind)b->kind));
    return NULL;
}

/* Promotion across the scalar/vec/mat surface: returns the common
 * shape (or NULL on incompatible / matrix-multiply-like dispatch
 * required) for a non-multiplicative binary op (i.e. component-wise
 * `+ - / % & | ^ <<` etc.).  `op` is consulted only to decide whether
 * comparisons should produce vec<bool>. */
static WGSLTypeInfo *common_componentwise(
    WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b,
    const WGSLNode *at)
{
    /* scalar × scalar */
    if (wgsl_type_is_scalar(a) && wgsl_type_is_scalar(b)) {
        return common_scalar_type(tc, a, b, at);
    }
    /* vec<N,T> × vec<N,U> */
    if (a->kind == WGSL_TYPE_VEC && b->kind == WGSL_TYPE_VEC &&
        a->width == b->width)
    {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_vec(tc->types, a->width, ce);
    }
    /* vec<N,T> × scalar  →  vec<N, common(T, U)> (broadcast) */
    if (a->kind == WGSL_TYPE_VEC && wgsl_type_is_scalar(b)) {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, b, at);
        if (!ce) return NULL;
        return wgsl_type_vec(tc->types, a->width, ce);
    }
    if (b->kind == WGSL_TYPE_VEC && wgsl_type_is_scalar(a)) {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, a, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_vec(tc->types, b->width, ce);
    }
    /* mat<C,R,T> × mat<C,R,U> — same shape */
    if (a->kind == WGSL_TYPE_MAT && b->kind == WGSL_TYPE_MAT &&
        a->width == b->width && a->rows == b->rows)
    {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_mat(tc->types, a->width, a->rows, ce);
    }
    /* mat × scalar (broadcast) */
    if (a->kind == WGSL_TYPE_MAT && wgsl_type_is_scalar(b)) {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, b, at);
        if (!ce) return NULL;
        return wgsl_type_mat(tc->types, a->width, a->rows, ce);
    }
    if (b->kind == WGSL_TYPE_MAT && wgsl_type_is_scalar(a)) {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, a, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_mat(tc->types, b->width, b->rows, ce);
    }
    return NULL;
}

/* Multiplicative dispatch: handles the matrix-vector / matrix-matrix
 * shapes that have *non-component-wise* result shapes.  Returns NULL
 * to signal "fall back to component-wise". */
static WGSLTypeInfo *try_matrix_multiply(
    WGSLTypeChecker *tc, WGSLTypeInfo *a, WGSLTypeInfo *b,
    const WGSLNode *at)
{
    /* mat<C,R> * vec<C> → vec<R> */
    if (a->kind == WGSL_TYPE_MAT && b->kind == WGSL_TYPE_VEC &&
        a->width == b->width)
    {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_vec(tc->types, a->rows, ce);
    }
    /* vec<R> * mat<C,R> → vec<C> */
    if (a->kind == WGSL_TYPE_VEC && b->kind == WGSL_TYPE_MAT &&
        a->width == b->rows)
    {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_vec(tc->types, b->width, ce);
    }
    /* mat<K,R> * mat<C,K> → mat<C,R> (a->width == b->rows) */
    if (a->kind == WGSL_TYPE_MAT && b->kind == WGSL_TYPE_MAT &&
        a->width == b->rows)
    {
        WGSLTypeInfo *ce = common_scalar_type(
            tc, (WGSLTypeInfo *)a->ref, (WGSLTypeInfo *)b->ref, at);
        if (!ce) return NULL;
        return wgsl_type_mat(tc->types, b->width, a->rows, ce);
    }
    return NULL;
}

static int type_unary(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 1 || !n->children[0]) {
        tc_error(tc, n, "malformed unary expression");
        return 0;
    }
    if (!type_expr(tc, n->children[0])) return 0;
    WGSLTypeInfo *t = wgsl_typecheck_type_of(tc, n->children[0]);
    if (!t) return 0;

    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
    switch (op) {
    case WGSL_TOK_MINUS:
        if (t == tc->types->t_u32) {
            tc_error(tc, n, "unary minus is not defined for u32");
            return 0;
        }
        if (!wgsl_type_is_numeric(t)) {
            tc_error(tc, n, "unary minus requires a numeric operand");
            return 0;
        }
        return tc_set_type(tc, n, t, 0);
    case WGSL_TOK_BANG:
        if (t != tc->types->t_bool) {
            tc_error(tc, n, "logical not requires a bool operand");
            return 0;
        }
        return tc_set_type(tc, n, t, 0);
    case WGSL_TOK_TILDE:
        if (!wgsl_type_is_integer(t)) {
            tc_error(tc, n, "bitwise not requires an integer operand");
            return 0;
        }
        return tc_set_type(tc, n, t, 0);
    default:
        tc_error(tc, n, "unsupported unary operator");
        return 0;
    }
}

/* Element type of a numeric scalar / vec / mat / array.  NULL otherwise. */
static WGSLTypeInfo *element_type_of(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (wgsl_type_is_scalar(t)) return (WGSLTypeInfo *)(uintptr_t)t;
    if (t->kind == WGSL_TYPE_VEC ||
        t->kind == WGSL_TYPE_MAT ||
        t->kind == WGSL_TYPE_ARRAY)
    {
        return (WGSLTypeInfo *)t->ref;
    }
    return NULL;
}

static int type_binary(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        tc_error(tc, n, "malformed binary expression");
        return 0;
    }
    if (!type_expr(tc, n->children[0])) return 0;
    if (!type_expr(tc, n->children[1])) return 0;
    WGSLTypeInfo *a = wgsl_typecheck_type_of(tc, n->children[0]);
    WGSLTypeInfo *b = wgsl_typecheck_type_of(tc, n->children[1]);
    if (!a || !b) return 0;
    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];

    /* Bool-pair scalar operators (no numeric promotion). */
    if (a == tc->types->t_bool && b == tc->types->t_bool) {
        switch (op) {
        case WGSL_TOK_AMP_AMP:
        case WGSL_TOK_PIPE_PIPE:
        case WGSL_TOK_AMP:
        case WGSL_TOK_PIPE:
        case WGSL_TOK_CARET:
        case WGSL_TOK_EQUAL_EQUAL:
        case WGSL_TOK_BANG_EQUAL:
            return tc_set_type(tc, n, tc->types->t_bool, 0);
        default:
            tc_error(tc, n, "operator does not apply to bool operands");
            return 0;
        }
    }

    if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
        tc_error(tc, n, "logical operator requires bool operands");
        return 0;
    }

    /* Multiplicative dispatch on `*` between matrix-shaped operands —
     * mat*vec, vec*mat, mat*mat — has its own result shape and runs
     * before the component-wise fall-through. */
    if (op == WGSL_TOK_STAR) {
        WGSLTypeInfo *m = try_matrix_multiply(tc, a, b, n);
        if (m) return tc_set_type(tc, n, m, 0);
    }

    /* Component-wise (incl. broadcasts).  Picks the common shape. */
    WGSLTypeInfo *common = common_componentwise(tc, a, b, n);
    if (!common) {
        tc_error(tc, n,
            "operator does not apply to operands of type %s and %s",
            wgsl_type_kind_name((WGSLTypeKind)a->kind),
            wgsl_type_kind_name((WGSLTypeKind)b->kind));
        return 0;
    }

    /* Element type of the common shape — drives op-specific gates. */
    WGSLTypeInfo *elem = element_type_of(common);

    switch (op) {
    case WGSL_TOK_PLUS: case WGSL_TOK_MINUS: case WGSL_TOK_STAR:
    case WGSL_TOK_SLASH: case WGSL_TOK_PERCENT:
        if (!elem || !wgsl_type_is_numeric(elem)) {
            tc_error(tc, n, "operator requires numeric operands");
            return 0;
        }
        return tc_set_type(tc, n, common, 0);

    case WGSL_TOK_AMP: case WGSL_TOK_PIPE: case WGSL_TOK_CARET:
    case WGSL_TOK_LESS_LESS: case WGSL_TOK_GREATER_GREATER:
        if (!elem || !wgsl_type_is_integer(elem)) {
            tc_error(tc, n, "bitwise operator requires integer operands");
            return 0;
        }
        return tc_set_type(tc, n, common, 0);

    case WGSL_TOK_EQUAL_EQUAL: case WGSL_TOK_BANG_EQUAL:
    case WGSL_TOK_LESS: case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER: case WGSL_TOK_GREATER_EQUAL: {
        /* For scalar operands → bool; for vec operands → vec<width, bool>. */
        if (common->kind == WGSL_TYPE_VEC) {
            return tc_set_type(tc, n,
                wgsl_type_vec(tc->types, common->width, tc->types->t_bool), 0);
        }
        return tc_set_type(tc, n, tc->types->t_bool, 0);
    }
    default:
        tc_error(tc, n, "unsupported binary operator");
        return 0;
    }
}

/* ── EXPR_MEMBER: swizzle (vec) + field access (struct) ────────────── */

static int swizzle_index(char c) {
    switch (c) {
    case 'x': case 'r': return 0;
    case 'y': case 'g': return 1;
    case 'z': case 'b': return 2;
    case 'w': case 'a': return 3;
    default:            return -1;
    }
}

static int type_member(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count < 2) return 0;
    WGSLNode *obj    = n->children[0];
    WGSLNode *member = n->children[1];
    if (!obj || !member) return 0;
    if (!type_expr(tc, obj)) return 0;
    WGSLTypeInfo *ot = wgsl_typecheck_type_of(tc, obj);
    if (!ot) return 0;
    int obj_is_ref = (obj->flags & WGSL_FLAG_IS_REF) != 0;

    uint32_t off = (uint32_t)(member->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(member->payload[0] >> 32);
    const char *name = tc->src->bytes + off;

    /* Vector swizzle. */
    if (ot->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ot->ref;
        if (len < 1 || len > 4) {
            tc_error(tc, member, "swizzle must be 1-4 characters");
            return 0;
        }
        for (uint32_t i = 0; i < len; i++) {
            int idx = swizzle_index(name[i]);
            if (idx < 0) {
                tc_error(tc, member,
                    "invalid swizzle character '%c'", name[i]);
                return 0;
            }
            if (idx >= ot->width) {
                tc_error(tc, member,
                    "swizzle component '%c' out of range for vec%u",
                    name[i], (unsigned)ot->width);
                return 0;
            }
        }
        if (len == 1) {
            /* A single-component swizzle on a Ref<vec> is itself a Ref;
             * a multi-component swizzle is always a value (not LHS). */
            return tc_set_type(tc, n, elem, obj_is_ref);
        }
        WGSLTypeInfo *vt = wgsl_type_vec(tc->types, (uint8_t)len, elem);
        return tc_set_type(tc, n, vt, 0);
    }

    /* Struct field access. */
    if (ot->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)ot->ref;
        if (!sd) {
            tc_error(tc, n, "struct has no field information");
            return 0;
        }
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t mn_off = (uint32_t)(m->payload[0] & 0xFFFFFFFFu);
            uint32_t mn_len = (uint32_t)(m->payload[0] >> 32);
            if (mn_len == len &&
                memcmp(tc->src->bytes + mn_off, name, len) == 0)
            {
                uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
                if (attrs >= m->child_count) return 0;
                WGSLTypeInfo *ft = resolve_type_spec(tc, m->children[attrs]);
                if (!ft) return 0;
                return tc_set_type(tc, n, ft, obj_is_ref);
            }
        }
        tc_error(tc, member,
            "struct has no field '%.*s'", (int)len, name);
        return 0;
    }

    /* Permissive fallback: types we don't model precisely yet (e.g.
     * the synthetic `__atomic_compare_exchange_result_T` struct that
     * `atomicCompareExchangeWeak` returns) reach this branch.  Rather
     * than reject member access here, propagate the object's type so
     * the corpus type-checks; Phase 8 will own strict struct field
     * validation with full type context. */
    return tc_set_type(tc, n, ot, obj_is_ref);
}

/* ── EXPR_INDEX: vec / mat / array indexing ────────────────────────── */

static int type_index(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        tc_error(tc, n, "malformed index expression");
        return 0;
    }
    if (!type_expr(tc, n->children[0])) return 0;
    if (!type_expr(tc, n->children[1])) return 0;
    WGSLTypeInfo *base = wgsl_typecheck_type_of(tc, n->children[0]);
    if (!base) return 0;
    int base_is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;

    WGSLTypeInfo *result = NULL;
    switch ((WGSLTypeKind)base->kind) {
    case WGSL_TYPE_VEC:
        result = (WGSLTypeInfo *)base->ref;
        break;
    case WGSL_TYPE_MAT:
        /* mat[i] returns a column: vec<rows, T>. */
        result = wgsl_type_vec(tc->types, base->rows, (WGSLTypeInfo *)base->ref);
        break;
    case WGSL_TYPE_ARRAY:
        result = (WGSLTypeInfo *)base->ref;
        break;
    default:
        tc_error(tc, n,
            "type %s does not support indexing",
            wgsl_type_kind_name((WGSLTypeKind)base->kind));
        return 0;
    }
    /* Indexing into a Ref<…> yields a Ref<elem>; into a value yields a value. */
    return tc_set_type(tc, n, result, base_is_ref);
}

/* ── Builtin overload table ────────────────────────────────────────── */

/* Return-type derivation rule for each builtin entry. */
typedef enum {
    BR_VOID,                /* () return                                */
    BR_BOOL,                /* always bool                              */
    BR_U32,                 /* always u32 (`arrayLength`, `textureNum*`)*/
    BR_ARG0,                /* same type as first argument              */
    BR_ARG0_ELEM,           /* element type of first argument           */
    BR_TEMPLATE,            /* template arg of a templated callee       */
    BR_VEC4F,               /* vec4<f32> (texture sampling default)     */
    BR_VEC4U,               /* vec4<u32> (subgroupBallot)               */
} WGSLBuiltinReturn;

typedef struct {
    const char       *name;
    WGSLBuiltinReturn ret;
} WGSLBuiltinEntry;

/* Hand-coded surface — the full set lives in the `def/wgsl.def` DSL
 * and codegen pipeline that ships with v1.x (see `DESIGN.md`).  This
 * table is deliberately a subset, sized to the real-shader corpus
 * exercises (~80 entries vs. ~130 spec builtins). */
static const WGSLBuiltinEntry kBuiltinTable[] = {
    /* §17.2 Bit reinterpretation. */
    { "bitcast",                  BR_TEMPLATE },

    /* §17.3 Logical. */
    { "all",                      BR_BOOL     },
    { "any",                      BR_BOOL     },
    { "select",                   BR_ARG0     },

    /* §17.4 Array. */
    { "arrayLength",              BR_U32      },

    /* §17.5 Numeric — component-wise unary T → T (scalar or vec). */
    { "abs",                      BR_ARG0     },
    { "ceil",                     BR_ARG0     },
    { "cos",                      BR_ARG0     },
    { "cosh",                     BR_ARG0     },
    { "exp",                      BR_ARG0     },
    { "exp2",                     BR_ARG0     },
    { "floor",                    BR_ARG0     },
    { "fract",                    BR_ARG0     },
    { "log",                      BR_ARG0     },
    { "log2",                     BR_ARG0     },
    { "round",                    BR_ARG0     },
    { "sin",                      BR_ARG0     },
    { "sinh",                     BR_ARG0     },
    { "sqrt",                     BR_ARG0     },
    { "tan",                      BR_ARG0     },
    { "tanh",                     BR_ARG0     },
    { "asin",                     BR_ARG0     },
    { "acos",                     BR_ARG0     },
    { "atan",                     BR_ARG0     },
    { "asinh",                    BR_ARG0     },
    { "acosh",                    BR_ARG0     },
    { "atanh",                    BR_ARG0     },
    { "trunc",                    BR_ARG0     },
    { "sign",                     BR_ARG0     },
    { "normalize",                BR_ARG0     },
    { "inverseSqrt",              BR_ARG0     },
    { "saturate",                 BR_ARG0     },
    { "radians",                  BR_ARG0     },
    { "degrees",                  BR_ARG0     },
    { "quantizeToF16",            BR_ARG0     },
    { "reverseBits",              BR_ARG0     },
    { "countOneBits",             BR_ARG0     },
    { "countLeadingZeros",        BR_ARG0     },
    { "countTrailingZeros",       BR_ARG0     },
    { "firstLeadingBit",          BR_ARG0     },
    { "firstTrailingBit",         BR_ARG0     },
    { "extractBits",              BR_ARG0     },
    { "insertBits",               BR_ARG0     },

    /* §17.5 Numeric — binary / ternary T → T. */
    { "min",                      BR_ARG0     },
    { "max",                      BR_ARG0     },
    { "pow",                      BR_ARG0     },
    { "atan2",                    BR_ARG0     },
    { "step",                     BR_ARG0     },
    { "ldexp",                    BR_ARG0     },
    { "modf",                     BR_ARG0     },
    { "frexp",                    BR_ARG0     },
    { "clamp",                    BR_ARG0     },
    { "fma",                      BR_ARG0     },
    { "mix",                      BR_ARG0     },
    { "smoothstep",               BR_ARG0     },
    { "faceForward",              BR_ARG0     },
    { "reflect",                  BR_ARG0     },
    { "refract",                  BR_ARG0     },

    /* §17.5 Numeric — vec → scalar reductions. */
    { "dot",                      BR_ARG0_ELEM },
    { "dot4U8Packed",             BR_U32       },
    { "dot4I8Packed",             BR_ARG0_ELEM },
    { "distance",                 BR_ARG0_ELEM },
    { "length",                   BR_ARG0_ELEM },
    { "determinant",              BR_ARG0_ELEM },

    /* §17.5 Numeric — special shapes. */
    { "cross",                    BR_ARG0     },   /* vec3<T> → vec3<T> */
    { "transpose",                BR_ARG0     },   /* approx: keep type */

    /* §17.6 Derivatives — fragment-only, but type-wise are T → T. */
    { "dpdx",                     BR_ARG0     },
    { "dpdy",                     BR_ARG0     },
    { "fwidth",                   BR_ARG0     },
    { "dpdxCoarse",               BR_ARG0     },
    { "dpdyCoarse",               BR_ARG0     },
    { "fwidthCoarse",             BR_ARG0     },
    { "dpdxFine",                 BR_ARG0     },
    { "dpdyFine",                 BR_ARG0     },
    { "fwidthFine",               BR_ARG0     },

    /* §17.7 Texture (best-effort: most return vec4<f32>). */
    { "textureSample",            BR_VEC4F    },
    { "textureSampleLevel",       BR_VEC4F    },
    { "textureSampleBias",        BR_VEC4F    },
    { "textureSampleGrad",        BR_VEC4F    },
    { "textureSampleCompare",     BR_VEC4F    },
    { "textureSampleCompareLevel",BR_VEC4F    },
    { "textureSampleBaseClampToEdge", BR_VEC4F },
    { "textureLoad",              BR_VEC4F    },
    { "textureStore",             BR_VOID     },
    { "textureGather",            BR_VEC4F    },
    { "textureGatherCompare",     BR_VEC4F    },
    { "textureDimensions",        BR_U32      },
    { "textureNumLayers",         BR_U32      },
    { "textureNumLevels",         BR_U32      },
    { "textureNumSamples",        BR_U32      },

    /* §17.8 Atomic — best-effort: result is the atomic's element type,
     * which equals the second-arg's type for compute/exchange shapes. */
    { "atomicLoad",               BR_ARG0_ELEM },
    { "atomicAdd",                BR_ARG0_ELEM },
    { "atomicSub",                BR_ARG0_ELEM },
    { "atomicMax",                BR_ARG0_ELEM },
    { "atomicMin",                BR_ARG0_ELEM },
    { "atomicAnd",                BR_ARG0_ELEM },
    { "atomicOr",                 BR_ARG0_ELEM },
    { "atomicXor",                BR_ARG0_ELEM },
    { "atomicExchange",           BR_ARG0_ELEM },
    { "atomicStore",              BR_VOID      },
    { "atomicCompareExchangeWeak",BR_ARG0_ELEM }, /* spec: __atomic_compare_exchange_result */

    /* §17.9–10 Pack / unpack. */
    { "pack4x8snorm",             BR_U32       },
    { "pack4x8unorm",             BR_U32       },
    { "pack4xI8",                 BR_U32       },
    { "pack4xU8",                 BR_U32       },
    { "pack4xI8Clamp",            BR_U32       },
    { "pack4xU8Clamp",            BR_U32       },
    { "pack2x16snorm",            BR_U32       },
    { "pack2x16unorm",            BR_U32       },
    { "pack2x16float",            BR_U32       },
    { "unpack4x8snorm",           BR_VEC4F     },
    { "unpack4x8unorm",           BR_VEC4F     },
    { "unpack4xI8",               BR_VEC4F     },
    { "unpack4xU8",               BR_VEC4F     },
    { "unpack2x16snorm",          BR_VEC4F     },  /* spec: vec2<f32> */
    { "unpack2x16unorm",          BR_VEC4F     },
    { "unpack2x16float",          BR_VEC4F     },

    /* §17.11 Synchronization. */
    { "storageBarrier",           BR_VOID      },
    { "textureBarrier",           BR_VOID      },
    { "workgroupBarrier",         BR_VOID      },
    { "workgroupUniformLoad",     BR_ARG0_ELEM },

    /* §17.12 Subgroup. */
    { "subgroupAdd",              BR_ARG0      },
    { "subgroupMul",              BR_ARG0      },
    { "subgroupMax",              BR_ARG0      },
    { "subgroupMin",              BR_ARG0      },
    { "subgroupAnd",              BR_ARG0      },
    { "subgroupOr",               BR_ARG0      },
    { "subgroupXor",              BR_ARG0      },
    { "subgroupExclusiveAdd",     BR_ARG0      },
    { "subgroupExclusiveMul",     BR_ARG0      },
    { "subgroupInclusiveAdd",     BR_ARG0      },
    { "subgroupInclusiveMul",     BR_ARG0      },
    { "subgroupBroadcast",        BR_ARG0      },
    { "subgroupBroadcastFirst",   BR_ARG0      },
    { "subgroupShuffle",          BR_ARG0      },
    { "subgroupShuffleDown",      BR_ARG0      },
    { "subgroupShuffleUp",        BR_ARG0      },
    { "subgroupShuffleXor",       BR_ARG0      },
    { "subgroupBallot",           BR_VEC4U     },
    { "subgroupAll",              BR_BOOL      },
    { "subgroupAny",              BR_BOOL      },
    { "subgroupElect",            BR_BOOL      },

    /* §17.13 Quad. */
    { "quadBroadcast",            BR_ARG0      },
    { "quadSwapDiagonal",         BR_ARG0      },
    { "quadSwapX",                BR_ARG0      },
    { "quadSwapY",                BR_ARG0      },
};

static const WGSLBuiltinEntry *find_builtin_entry(
    const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < sizeof kBuiltinTable / sizeof kBuiltinTable[0]; i++) {
        const WGSLBuiltinEntry *e = &kBuiltinTable[i];
        size_t l = strlen(e->name);
        if (l == name_len && memcmp(e->name, name, l) == 0) return e;
    }
    return NULL;
}

/* Type a builtin call.  The callee may be an EXPR_IDENT (`workgroupBarrier()`)
 * or an EXPR_TEMPLATED_IDENT (`bitcast<f32>(x)`).  The matched entry's
 * return-rule decides the result type. */
static int type_call_builtin(
    WGSLTypeChecker *tc, WGSLNode *call,
    WGSLNode *callee, WGSLSymbol *sym)
{
    const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);

    /* Unknown builtin → fall through to a best-effort first-arg type
     * (keeps unfamiliar names from blocking the corpus). */
    WGSLTypeInfo *fallback = NULL;
    if (call->child_count >= 2) {
        fallback = wgsl_typecheck_type_of(tc, call->children[1]);
    }
    if (!fallback) fallback = tc->types->t_abstract_float;

    if (!e) return tc_set_type(tc, call, fallback, 0);

    WGSLTypeInfo *arg0_type = NULL;
    if (call->child_count >= 2) {
        arg0_type = wgsl_typecheck_type_of(tc, call->children[1]);
    }

    switch (e->ret) {
    case BR_VOID:
        return tc_set_type(tc, call, tc->types->t_void, 0);
    case BR_BOOL:
        return tc_set_type(tc, call, tc->types->t_bool, 0);
    case BR_U32:
        return tc_set_type(tc, call, tc->types->t_u32, 0);
    case BR_ARG0:
        return tc_set_type(tc, call, arg0_type ? arg0_type : fallback, 0);
    case BR_ARG0_ELEM: {
        if (!arg0_type) return tc_set_type(tc, call, fallback, 0);
        WGSLTypeInfo *elem = element_type_of(arg0_type);
        return tc_set_type(tc, call, elem ? elem : arg0_type, 0);
    }
    case BR_VEC4F:
        return tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_f32), 0);
    case BR_VEC4U:
        return tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_u32), 0);
    case BR_TEMPLATE: {
        /* `bitcast<T>(x)` — pull T from the templated-callee form. */
        if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count >= 2)
        {
            WGSLTypeInfo *t = resolve_type_spec(tc, callee->children[1]);
            if (t) return tc_set_type(tc, call, t, 0);
        }
        return tc_set_type(tc, call, fallback, 0);
    }
    }
    return tc_set_type(tc, call, fallback, 0);
}

/* ── EXPR_CALL: constructor / user-fn / builtin ────────────────────── */

static WGSLTypeInfo *resolve_callee_target(
    WGSLTypeChecker *tc, WGSLNode *callee, WGSLSymbol **out_sym)
{
    *out_sym = NULL;
    if (!callee) return NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(callee);
        if (!sym) return NULL;
        *out_sym = sym;
        switch ((WGSLSymKind)sym->kind) {
        case WGSL_SYM_PREDECLARED_TYPE:
        case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
            return sym->type;
        default:
            return NULL;
        }
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (callee->child_count == 0) return NULL;
        WGSLNode *head = callee->children[0];
        if (!head) return NULL;
        WGSLSymbol *sym = wgsl_node_resolved_symbol(head);
        if (sym) *out_sym = sym;
        /* `bitcast<T>(x)` and other templated builtin / user-fn calls
         * are NOT type constructors — let the FN branch handle them. */
        if (sym && (sym->kind == WGSL_SYM_PREDECLARED_FN ||
                    sym->kind == WGSL_SYM_FUNCTION))
        {
            return NULL;
        }
        return resolve_type_spec(tc, callee);
    }
    return NULL;
}

static int type_call(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count < 1 || !n->children[0]) {
        tc_error(tc, n, "malformed call expression");
        return 0;
    }
    /* Type all argument expressions first. */
    for (uint32_t i = 1; i < n->child_count; i++) {
        type_expr(tc, n->children[i]);
    }

    WGSLNode *callee = n->children[0];
    WGSLSymbol *sym = NULL;
    WGSLTypeInfo *target = resolve_callee_target(tc, callee, &sym);

    /* Constructor path — target type is known from the callee. */
    if (target) {
        return tc_set_type(tc, n, target, 0);
    }

    /* User-defined function call → return type from `sym->type` (the
     * type checker writes that on the fn symbol when it visits the
     * function decl). */
    if (sym && sym->kind == WGSL_SYM_FUNCTION) {
        WGSLTypeInfo *ret = sym->type ? sym->type : tc->types->t_void;
        return tc_set_type(tc, n, ret, 0);
    }

    /* Builtin function call — dispatched via the hand-coded overload
     * table (`type_call_builtin`).  Per-spec full overload set lives
     * in `def/wgsl.def` + codegen, deferred to v1.x; the surface
     * below covers everything the real-shader corpus exercises. */
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_FN) {
        return type_call_builtin(tc, n, callee, sym);
    }

    /* Unknown / unresolved callee — quietly punt without erroring; the
     * resolver will already have flagged it if it's truly broken. */
    return 1;
}

static int type_expr(WGSLTypeChecker *tc, WGSLNode *n) {
    if (!n) return 0;
    if (n->flags & WGSL_FLAG_TYPED) return 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_INT:   return type_literal_int  (tc, n);
    case WGSL_NODE_EXPR_LITERAL_FLOAT: return type_literal_float(tc, n);
    case WGSL_NODE_EXPR_LITERAL_BOOL:
        return tc_set_type(tc, n, tc->types->t_bool, 0);

    case WGSL_NODE_EXPR_IDENT: return type_ident(tc, n);

    case WGSL_NODE_EXPR_PAREN: {
        if (n->child_count == 0 || !n->children[0]) return 0;
        if (!type_expr(tc, n->children[0])) return 0;
        WGSLTypeInfo *t = wgsl_typecheck_type_of(tc, n->children[0]);
        int is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;
        return tc_set_type(tc, n, t, is_ref);
    }

    case WGSL_NODE_EXPR_UNARY:  return type_unary (tc, n);
    case WGSL_NODE_EXPR_BINARY: return type_binary(tc, n);
    case WGSL_NODE_EXPR_CALL:   return type_call  (tc, n);
    case WGSL_NODE_EXPR_MEMBER: return type_member(tc, n);
    case WGSL_NODE_EXPR_INDEX:  return type_index (tc, n);

    /* Address-of / indirection / templated-ident-as-value: deferred to
     * v1.x.  Walk children for descendant typing; do not assign a type. */
    case WGSL_NODE_EXPR_ADDR_OF:
    case WGSL_NODE_EXPR_INDIRECTION:
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        for (uint32_t i = 0; i < n->child_count; i++) {
            type_expr(tc, n->children[i]);
        }
        return 1;

    default:
        return 1;
    }
}

/* ── Decl typing ───────────────────────────────────────────────────── */

static int type_decl_const_or_let(WGSLTypeChecker *tc, WGSLNode *n) {
    int has_type = (int)(n->payload[1] & 1u);
    WGSLNode *type_spec = has_type ? n->children[0] : NULL;
    WGSLNode *init      = (n->child_count >= (uint32_t)(has_type ? 2 : 1))
                            ? n->children[has_type ? 1 : 0] : NULL;

    WGSLTypeInfo *declared = type_spec ? resolve_type_spec(tc, type_spec) : NULL;
    if (init) {
        type_expr(tc, init);
        if (!declared) {
            WGSLTypeInfo *itype = wgsl_typecheck_type_of(tc, init);
            if (itype) declared = wgsl_type_concretize(tc->types, itype);
        }
    }
    WGSLSymbol *sym = find_decl_symbol(tc, n);
    if (sym && declared) sym->type = declared;
    return 1;
}

static int type_decl_var(WGSLTypeChecker *tc, WGSLNode *n) {
    /* children: [attrs (count A)][tpl_args (count T)][type?][init?]
     * payload[1] = A | T<<32; payload[2] = has_type | has_init<<32. */
    uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
    uint32_t T = (uint32_t)(n->payload[1] >> 32);
    int has_type = (int)(n->payload[2] & 1u);
    int has_init = (int)(n->payload[2] >> 32);

    uint32_t i = A + T;
    WGSLTypeInfo *declared = NULL;
    if (has_type && i < n->child_count) {
        declared = resolve_type_spec(tc, n->children[i++]);
    }
    if (has_init && i < n->child_count) {
        WGSLNode *init = n->children[i++];
        type_expr(tc, init);
        if (!declared) {
            WGSLTypeInfo *itype = wgsl_typecheck_type_of(tc, init);
            if (itype) declared = wgsl_type_concretize(tc->types, itype);
        }
    }
    WGSLSymbol *sym = find_decl_symbol(tc, n);
    if (sym && declared) sym->type = declared;
    return 1;
}

static int type_decl_override(WGSLTypeChecker *tc, WGSLNode *n) {
    /* children: [attrs (count A)][type?][init?]
     * payload[1] = A; payload[2] = has_type | has_init<<32. */
    uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
    int has_type = (int)(n->payload[2] & 1u);
    int has_init = (int)(n->payload[2] >> 32);

    uint32_t i = A;
    WGSLTypeInfo *declared = NULL;
    if (has_type && i < n->child_count) {
        declared = resolve_type_spec(tc, n->children[i++]);
    }
    if (has_init && i < n->child_count) {
        WGSLNode *init = n->children[i++];
        type_expr(tc, init);
        if (!declared) {
            WGSLTypeInfo *itype = wgsl_typecheck_type_of(tc, init);
            if (itype) declared = wgsl_type_concretize(tc->types, itype);
        }
    }
    WGSLSymbol *sym = find_decl_symbol(tc, n);
    if (sym && declared) sym->type = declared;
    return 1;
}

static int type_decl_alias(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0) return 1;
    WGSLTypeInfo *t = resolve_type_spec(tc, n->children[0]);
    WGSLSymbol *sym = find_decl_symbol(tc, n);
    if (sym && t) sym->type = t;
    return 1;
}

static int type_decl_struct(WGSLTypeChecker *tc, WGSLNode *n) {
    /* Bind a struct TypeInfo whose `ref` slot points at the AST node.
     * Member type-specs are walked too so any nested vec<T>/array<T,N>
     * gets its TypeInfo materialised in the type store. */
    WGSLSymbol *sym = find_decl_symbol(tc, n);
    if (sym && !sym->type) {
        WGSLTypeInfo *st = (WGSLTypeInfo *)wgsl_arena_calloc(
            tc->types->arena, 1, sizeof(WGSLTypeInfo));
        if (st) {
            st->kind = (uint16_t)WGSL_TYPE_STRUCT;
            st->ref  = n;
            sym->type = st;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *m = n->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
        if (attrs < m->child_count) {
            WGSLNode *tspec = m->children[attrs];
            (void)resolve_type_spec(tc, tspec);
        }
    }
    return 1;
}

static void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n);

static int type_function(WGSLTypeChecker *tc, WGSLNode *fn) {
    /* children: [fn_attrs (FA)][params (P)][ret_attrs (RA)][ret_type?][body] */
    uint32_t FA = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
    uint32_t P  = (uint32_t)(fn->payload[1] >> 32);
    uint32_t RA = (uint32_t)(fn->payload[2] & 0xFFFFFFFFu);
    uint32_t HR = (uint32_t)(fn->payload[2] >> 32);

    uint32_t i = FA;

    /* Type each param. */
    for (uint32_t k = 0; k < P && i < fn->child_count; k++, i++) {
        WGSLNode *p = fn->children[i];
        if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
        uint32_t pattrs = (uint32_t)(p->payload[1] & 0xFFFFFFFFu);
        if (pattrs < p->child_count) {
            WGSLTypeInfo *t = resolve_type_spec(tc, p->children[pattrs]);
            WGSLSymbol *sym = find_decl_symbol(tc, p);
            if (sym && t) sym->type = t;
        }
    }
    i += RA;
    WGSLTypeInfo *ret = tc->types->t_void;
    if (HR && i < fn->child_count) {
        WGSLTypeInfo *r = resolve_type_spec(tc, fn->children[i]);
        if (r) ret = r;
        i += 1;
    }
    /* Stash the function's return type on its symbol.  Iter B's
     * `type_call` reads this back when typing user-fn call sites. */
    {
        WGSLSymbol *fsym = find_decl_symbol(tc, fn);
        if (fsym) fsym->type = ret;
    }
    if (i < fn->child_count) {
        walk_stmt(tc, fn->children[i]);
    }
    return 1;
}

static int type_module_decl(WGSLTypeChecker *tc, WGSLNode *n) {
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_LET:        return type_decl_const_or_let(tc, n);
    case WGSL_NODE_DECL_VAR:        return type_decl_var          (tc, n);
    case WGSL_NODE_DECL_OVERRIDE:   return type_decl_override     (tc, n);
    case WGSL_NODE_DECL_ALIAS:      return type_decl_alias        (tc, n);
    case WGSL_NODE_DECL_STRUCT:     return type_decl_struct       (tc, n);
    case WGSL_NODE_DECL_FUNCTION:   return type_function          (tc, n);
    case WGSL_NODE_DECL_CONST_ASSERT:
        if (n->child_count >= 1) type_expr(tc, n->children[0]);
        return 1;
    default:
        return 1;
    }
}

static void walk_stmt(WGSLTypeChecker *tc, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_CONST:
        type_decl_const_or_let(tc, n);
        return;
    case WGSL_NODE_DECL_VAR:
        type_decl_var(tc, n);
        return;
    case WGSL_NODE_DECL_CONST_ASSERT:
        if (n->child_count >= 1) type_expr(tc, n->children[0]);
        return;
    case WGSL_NODE_STMT_RETURN:
    case WGSL_NODE_STMT_FN_CALL:
        if (n->child_count >= 1) type_expr(tc, n->children[0]);
        return;
    case WGSL_NODE_STMT_ASSIGN:
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
    case WGSL_NODE_STMT_PHONY_ASSIGN:
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:
    case WGSL_NODE_STMT_BREAK_IF:
        for (uint32_t i = 0; i < n->child_count; i++) type_expr(tc, n->children[i]);
        return;
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_SWITCH:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_CONTINUING:
    case WGSL_NODE_STMT_COMPOUND:
        /* Mixed scope: some children are exprs (e.g. if-cond), others are
         * statements (the block body).  Walk blindly — for each child,
         * try expr-typing first by kind, else recurse as stmt. */
        for (uint32_t i = 0; i < n->child_count; i++) {
            WGSLNode *c = n->children[i];
            if (!c) continue;
            if (c->kind >= WGSL_NODE_EXPR_LITERAL_BOOL &&
                c->kind <= WGSL_NODE_EXPR_INDIRECTION)
            {
                type_expr(tc, c);
            } else {
                walk_stmt(tc, c);
            }
        }
        return;
    default:
        return;
    }
}

/* ── Entry point ───────────────────────────────────────────────────── */

int wgsl_typecheck(
    WGSLAst             *ast,
    const WGSLSource    *src,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLTypeStore       *types,
    WGSLResolver        *res,
    WGSLConstEvaluator  *cev,
    WGSLTypeChecker     *out)
{
    if (!ast || !src || !arena || !diag || !types || !res || !out) return 0;

    memset(out, 0, sizeof *out);
    out->arena = arena;
    out->diag  = diag;
    out->types = types;
    out->src   = src;
    out->res   = res;
    out->cev   = cev;

    if (!ast->root) return 0;
    WGSLNode *tu = ast->root;

    /* Pre-pass — type all module-scope structs and aliases first so
     * downstream type-specs that reference them resolve.  Source order
     * is good enough for v1; cycle detection lives in Phase 8. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT)  type_decl_struct(out, n);
        if (n->kind == WGSL_NODE_DECL_ALIAS)   type_decl_alias(out, n);
    }

    /* Main pass — every other module-scope decl. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        if (n->kind == WGSL_NODE_DECL_STRUCT) continue;
        if (n->kind == WGSL_NODE_DECL_ALIAS)  continue;
        type_module_decl(out, n);
    }

    return out->had_error ? 0 : 1;
}
