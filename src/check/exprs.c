/**
 * @file exprs.c — Phase 7 expression typing + builtin-call
 * overload resolution.
 *
 * Covers every EXPR_* node kind: literals, identifiers, parens,
 * unary / binary operators, member access (swizzle on vec, field
 * access on struct), index (vec / mat / array), address-of, indirection,
 * function calls (type constructors / user-defined fns / builtin
 * functions through the generated `kBuiltinTable` metadata).
 *
 * The §6.1.3 overload resolution machinery (build_t_candidates +
 * resolve_overload) lives here too — it drives `type_call_builtin`'s
 * per-builtin shape inference using each entry's `WGSLParamFamily` +
 * `arity` configuration.  Builtin metadata is generated from
 * `def/wgsl.def`; mixed-shape overloads still use small local validators
 * where a compact T-pattern is not expressive enough.
 *
 * Co-extensive with WGSL §8 (Expressions) + §17 (Built-in Functions).
 */
#include "internal/check.h"
#include "internal/check_priv.h"
#include "internal/token.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Expression typing ─────────────────────────────────────────────── */

int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n);

static int type_literal_int(WGSLTypeChecker *tc, WGSLNode *n) {
    uint32_t suf = (uint32_t)n->payload[0];
    uint32_t base = suf & 0x0fu;
    WGSLTypeInfo *t;
    switch (base) {
    case WGSL_NUM_SUFFIX_I: t = tc->types->t_i32; break;
    case WGSL_NUM_SUFFIX_U: t = tc->types->t_u32; break;
    default:                t = tc->types->t_abstract_int; break;
    }
    if ((base == WGSL_NUM_SUFFIX_I || base == WGSL_NUM_SUFFIX_U) &&
        tc->cev && !(n->flags & WGSL_FLAG_CONST_EVALED))
    {
        WGSLValue v = {0};
        if (!wgsl_consteval_expr(tc->cev, n, &v)) {
            tc->had_error = 1;
        } else if (v.kind == WGSL_VAL_INT) {
            if (base == WGSL_NUM_SUFFIX_I &&
                (v.u.i < INT32_MIN || v.u.i > INT32_MAX))
            {
                wgsl_tc_error(tc, n, "value %lld out of range for i32",
                    (long long)v.u.i);
            }
            if (base == WGSL_NUM_SUFFIX_U &&
                (v.u.i < 0 || v.u.i > (int64_t)UINT32_MAX))
            {
                wgsl_tc_error(tc, n, "value %lld out of range for u32",
                    (long long)v.u.i);
            }
        }
    }
    return wgsl_tc_set_type(tc, n, t, 0);
}

static int type_literal_float(WGSLTypeChecker *tc, WGSLNode *n) {
    uint32_t suf = (uint32_t)n->payload[0];
    WGSLTypeInfo *t;
    int needs_range_check = 0;
    switch (suf & 0x0fu) {
    case WGSL_NUM_SUFFIX_F:
        t = tc->types->t_f32;
        needs_range_check = 1;
        break;
    case WGSL_NUM_SUFFIX_H:
        if (!(tc->enabled_extensions & WGSL_EXT_F16)) {
            wgsl_tc_error(tc, n, "use of 'h' suffix requires 'enable f16;'");
            return 0;
        }
        t = tc->types->t_f16;
        needs_range_check = 1;
        break;
    default:                t = tc->types->t_abstract_float; break;
    }
    /* §6.2.4 — kick the consteval-side range check for suffixed
     * literals that the module pass wouldn't otherwise visit (e.g.,
     * a `let` initializer).  `parse_float_literal` sets
     * WGSL_FLAG_CONST_EVALED on first visit so a `const` init that
     * was already materialized by the module pass isn't double-flagged. */
    if (needs_range_check && tc->cev && !(n->flags & WGSL_FLAG_CONST_EVALED)) {
        WGSLValue v = {0};
        if (!wgsl_consteval_expr(tc->cev, n, &v)) {
            tc->had_error = 1;
        }
    }
    return wgsl_tc_set_type(tc, n, t, 0);
}

static int type_ident(WGSLTypeChecker *tc, WGSLNode *n) {
    WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
    if (!sym) {
        wgsl_tc_error(tc, n, "unresolved identifier in type-checked expression");
        return 0;
    }
    /* Function names aren't first-class values in WGSL — they are only
     * usable in call position.  EXPR_CALL pulls the symbol out of the
     * callee directly, so a FUNCTION sym reaching this path is a
     * mis-use (`let f = my_fn;`). */
    if (sym->kind == WGSL_SYM_FUNCTION) {
        wgsl_tc_error(tc, n,
            "function '%.*s' is not a value",
            (int)sym->name_len, sym->name);
        return 0;
    }
    /* Builtin function names same story. */
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        wgsl_tc_error(tc, n,
            "builtin function '%.*s' is not a value",
            (int)sym->name_len, sym->name);
        return 0;
    }
    /* §6.3 — Enumerants are template parameters; they cannot appear as
     * a value, store type, or effective-value-type.  The legal positions
     * for these symbols are the template-arg slots of address-space /
     * access-mode / texture-storage shapes, which resolve through
     * dedicated paths (wgsl_tc_resolve_addr_space_arg / wgsl_tc_resolve_access_mode_arg /
     * resolve_texel_format_arg) before reaching here. */
    if (sym->kind == WGSL_SYM_PREDECLARED_ACCESS_MODE ||
        sym->kind == WGSL_SYM_PREDECLARED_ADDR_SPACE ||
        sym->kind == WGSL_SYM_PREDECLARED_TEXEL_FORMAT)
    {
        const char *what =
            sym->kind == WGSL_SYM_PREDECLARED_ACCESS_MODE ? "access mode" :
            sym->kind == WGSL_SYM_PREDECLARED_ADDR_SPACE  ? "address space" :
                                                             "texel format";
        wgsl_tc_error(tc, n,
            "%s enumerant '%.*s' can only be used as a template parameter, "
            "not as a value",
            what, (int)sym->name_len, sym->name);
        return 0;
    }
    if (!sym->type) {
        if (sym->ast && (
            sym->ast->kind == WGSL_NODE_DECL_VAR ||
            sym->ast->kind == WGSL_NODE_DECL_CONST ||
            sym->ast->kind == WGSL_NODE_DECL_OVERRIDE ||
            sym->ast->kind == WGSL_NODE_DECL_LET))
        {
            if (sym->ast->flags & WGSL_FLAG_TYPING) {
                wgsl_tc_error(tc, n, "'%.*s' has no type yet (cyclic dependency)",
                         (int)sym->name_len, sym->name);
                return 0;
            }
            sym->ast->flags |= WGSL_FLAG_TYPING;
            /* Forward-ref resolution: temporarily drop `in_function` so the
             * module decl is typed in its real (module-scope) context.
             * Otherwise `var<storage>` scope-placement gates etc. would
             * fire false-positives "must be at module scope" for a decl
             * lazily resolved while walking some fn body. */
            int saved_in_fn = tc->in_function;
            tc->in_function = 0;
            wgsl_tc_type_module_decl(tc, sym->ast);
            tc->in_function = saved_in_fn;
            sym->ast->flags &= ~WGSL_FLAG_TYPING;
        }
    }
    if (!sym->type) {
        wgsl_tc_error(tc, n, "'%.*s' has no type yet",
                 (int)sym->name_len, sym->name);
        return 0;
    }
    /* `var` bindings produce a structural ref<AS,T,AM>; const / let /
     * param produce values.  (Param-by-pointer has a ptr value type.) */
    int is_ref = (sym->kind == WGSL_SYM_VAR);
    return wgsl_tc_set_type(tc, n, sym->type, is_ref);
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

    wgsl_tc_error(tc, at,
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
        wgsl_tc_error(tc, n, "malformed unary expression");
        return 0;
    }
    if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
    WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
    if (!t) return 0;

    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
    WGSLTypeInfo *elem = t;
    if (t->kind == WGSL_TYPE_VEC) elem = (WGSLTypeInfo *)t->ref;
    switch (op) {
    case WGSL_TOK_MINUS:
        if (elem == tc->types->t_u32) {
            wgsl_tc_error(tc, n, "unary minus is not defined for u32");
            return 0;
        }
        if (!wgsl_type_is_numeric(elem)) {
            wgsl_tc_error(tc, n, "unary minus requires a numeric operand");
            return 0;
        }
        return wgsl_tc_set_type(tc, n, t, 0);
    case WGSL_TOK_BANG:
        if (elem != tc->types->t_bool) {
            wgsl_tc_error(tc, n, "logical not requires a bool operand");
            return 0;
        }
        return wgsl_tc_set_type(tc, n, t, 0);
    case WGSL_TOK_TILDE:
        if (!wgsl_type_is_integer(elem)) {
            wgsl_tc_error(tc, n, "bitwise not requires an integer operand");
            return 0;
        }
        return wgsl_tc_set_type(tc, n, t, 0);
    default:
        wgsl_tc_error(tc, n, "unsupported unary operator");
        return 0;
    }
}

/* Element type of a numeric scalar / vec / mat / array.  NULL otherwise. */
static WGSLTypeInfo *element_type_of(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (wgsl_type_is_scalar(t)) return (WGSLTypeInfo *)(uintptr_t)t;
    if (t->kind == WGSL_TYPE_VEC ||
        t->kind == WGSL_TYPE_MAT ||
        t->kind == WGSL_TYPE_ATOMIC ||
        t->kind == WGSL_TYPE_ARRAY)
    {
        return (WGSLTypeInfo *)t->ref;
    }
    return NULL;
}

static int consteval_noise_diag(const char *m) {
    return m &&
        (strstr(m, "not a constant") != NULL ||
         strstr(m, "not implemented") != NULL ||
         strstr(m, "unresolved identifier") != NULL ||
         strstr(m, "cyclic dependency") != NULL ||
         strstr(m, "is not a constant") != NULL ||
         strstr(m, "expression is not a constant") != NULL);
}

/* Returns 1 on successful const-eval, 0 when the expression is simply
 * non-constant, and -1 when const-eval surfaced a real shader error. */
static int try_consteval_expr_soft(
    WGSLTypeChecker *tc, WGSLNode *expr, WGSLValue *out)
{
    if (!tc || !tc->cev || !tc->diag || !expr || !out) return 0;
    if (tc->suppress_consteval_value_errors) return 0;
    size_t saved_count = tc->diag->count;
    int saved_errs = tc->diag->error_count;
    int saved_cev_err = tc->cev->had_error;
    memset(out, 0, sizeof *out);

    if (wgsl_consteval_expr(tc->cev, expr, out)) return 1;

    int real_error = 0;
    for (size_t k = saved_count; k < tc->diag->count; k++) {
        const char *m = tc->diag->items[k].message;
        if (!consteval_noise_diag(m)) {
            real_error = 1;
            break;
        }
    }
    if (real_error) {
        tc->had_error = 1;
        return -1;
    }

    tc->diag->count = saved_count;
    tc->diag->error_count = saved_errs;
    tc->cev->had_error = saved_cev_err;
    return 0;
}

static int const_value_contains_zero(const WGSLValue *v)
{
    if (!v) return 0;
    switch ((WGSLValKind)v->kind) {
    case WGSL_VAL_INT:
        return v->u.i == 0;
    case WGSL_VAL_FLOAT:
        return v->u.f == 0.0;
    case WGSL_VAL_VEC:
    case WGSL_VAL_MAT:
    case WGSL_VAL_ARRAY:
    case WGSL_VAL_STRUCT:
        for (uint32_t i = 0; i < v->u.agg.count; i++) {
            if (const_value_contains_zero(&v->u.agg.elems[i])) return 1;
        }
        return 0;
    default:
        return 0;
    }
}

void wgsl_tc_check_constexpr_divisor_nonzero(
    WGSLTypeChecker *tc,
    WGSLNode *rhs,
    WGSLTypeInfo *checked_type,
    const char *message)
{
    WGSLTypeInfo *elem = element_type_of(checked_type);
    if (!elem || !wgsl_type_is_integer(elem)) return;
    WGSLValue v = {0};
    if (try_consteval_expr_soft(tc, rhs, &v) == 1 &&
        const_value_contains_zero(&v))
    {
        wgsl_tc_error(tc, rhs, "%s", message);
    }
}

static int const_value_shift_amount_out_of_range(
    const WGSLValue *v, int64_t bit_width, int check_upper, int64_t *bad)
{
    if (!v) return 0;
    switch ((WGSLValKind)v->kind) {
    case WGSL_VAL_INT:
        if (v->u.i < 0 || (check_upper && v->u.i >= bit_width)) {
            if (bad) *bad = v->u.i;
            return 1;
        }
        return 0;
    case WGSL_VAL_VEC:
        for (uint32_t i = 0; i < v->u.agg.count; i++) {
            if (const_value_shift_amount_out_of_range(
                    &v->u.agg.elems[i], bit_width, check_upper, bad))
                return 1;
        }
        return 0;
    default:
        return 0;
    }
}

static int type_binary(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        wgsl_tc_error(tc, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];

    if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
        if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
        WGSLValue left_const = {0};
        int short_circuit = 0;
        if (try_consteval_expr_soft(tc, n->children[0], &left_const) == 1 &&
            left_const.kind == WGSL_VAL_BOOL)
        {
            short_circuit =
                (op == WGSL_TOK_AMP_AMP && !left_const.u.b) ||
                (op == WGSL_TOK_PIPE_PIPE && left_const.u.b);
        }
        if (short_circuit) tc->suppress_consteval_value_errors++;
        int rhs_ok = wgsl_tc_type_expr(tc, n->children[1]);
        if (short_circuit) tc->suppress_consteval_value_errors--;
        if (!rhs_ok) return 0;
    } else {
        if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
        if (!wgsl_tc_type_expr(tc, n->children[1])) return 0;
    }
    WGSLTypeInfo *a = wgsl_tc_store_type_of(tc, n->children[0]);
    WGSLTypeInfo *b = wgsl_tc_store_type_of(tc, n->children[1]);
    if (!a || !b) return 0;

    /* Bool-pair scalar operators (no numeric promotion). */
    if (a == tc->types->t_bool && b == tc->types->t_bool) {
        switch (op) {
        case WGSL_TOK_AMP_AMP:
        case WGSL_TOK_PIPE_PIPE:
        case WGSL_TOK_AMP:
        case WGSL_TOK_PIPE:
        case WGSL_TOK_EQUAL_EQUAL:
        case WGSL_TOK_BANG_EQUAL:
            return wgsl_tc_set_type(tc, n, tc->types->t_bool, 0);
        default:
            wgsl_tc_error(tc, n, "operator does not apply to bool operands");
            return 0;
        }
    }

    if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
        wgsl_tc_error(tc, n, "logical operator requires bool operands");
        return 0;
    }

    /* §8.9 shifts have asymmetric typing: LHS is i32 / u32 / AbstractInt
     * scalar or vec, RHS must be u32 (scalar/vec, matching LHS shape).
     * AbstractInt RHS is allowed (it converts to u32).  Run this gate
     * before component-wise unification so we can give a precise error
     * about the *shift amount* being the wrong type. */
    if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
        WGSLTypeInfo *ae = element_type_of(a);
        WGSLTypeInfo *be = element_type_of(b);
        int a_shape_ok = wgsl_type_is_scalar(a) || a->kind == WGSL_TYPE_VEC;
        int b_shape_ok = wgsl_type_is_scalar(b) || b->kind == WGSL_TYPE_VEC;
        if (!a_shape_ok || !b_shape_ok) {
            wgsl_tc_error(tc, n,
                "shift operands must be scalar or vector integer values");
            return 0;
        }
        if (!ae || !wgsl_type_is_integer(ae)) {
            wgsl_tc_error(tc, n, "shift requires an integer-typed left operand");
            return 0;
        }
        if (!be ||
            !(be == tc->types->t_u32 || be == tc->types->t_abstract_int))
        {
            wgsl_tc_error(tc, n,
                "shift amount must be u32 (or vec<u32>); got %s",
                be ? wgsl_type_kind_name((WGSLTypeKind)be->kind) : "non-numeric");
            return 0;
        }
        /* Vector shape: both sides must match width if both are vecs;
         * scalar LHS forbids a vec RHS and vice versa. */
        int a_vec = (a->kind == WGSL_TYPE_VEC);
        int b_vec = (b->kind == WGSL_TYPE_VEC);
        if (a_vec != b_vec ||
            (a_vec && b_vec && a->width != b->width))
        {
            wgsl_tc_error(tc, n,
                "shift operand shapes must match (got %s and %s)",
                wgsl_type_kind_name((WGSLTypeKind)a->kind),
                wgsl_type_kind_name((WGSLTypeKind)b->kind));
            return 0;
        }
        WGSLValue shift = {0};
        int64_t bad_shift = 0;
        int64_t bit_width =
            (ae == tc->types->t_i32 || ae == tc->types->t_u32) ? 32 : 64;
        int check_upper =
            !(op == WGSL_TOK_GREATER_GREATER &&
              ae == tc->types->t_abstract_int);
        if (try_consteval_expr_soft(tc, n->children[1], &shift) == 1 &&
            const_value_shift_amount_out_of_range(
                &shift, bit_width, check_upper, &bad_shift))
        {
            wgsl_tc_error(tc, n->children[1],
                "shift amount %lld is out of range for %s",
                (long long)bad_shift,
                wgsl_type_kind_name((WGSLTypeKind)ae->kind));
            return 0;
        }
        /* Result type = LHS type (concretized later if it carries
         * AbstractInt; the consteval / materialization path closes the
         * loop). */
        return wgsl_tc_set_type(tc, n, a, 0);
    }

    /* Multiplicative dispatch on `*` between matrix-shaped operands —
     * mat*vec, vec*mat, mat*mat — has its own result shape and runs
     * before the component-wise fall-through. */
    if (op == WGSL_TOK_STAR) {
        WGSLTypeInfo *m = try_matrix_multiply(tc, a, b, n);
        if (m) return wgsl_tc_set_type(tc, n, m, 0);
        int matrix_involved =
            (a->kind == WGSL_TYPE_MAT) || (b->kind == WGSL_TYPE_MAT);
        int matrix_scalar =
            (a->kind == WGSL_TYPE_MAT && wgsl_type_is_scalar(b)) ||
            (b->kind == WGSL_TYPE_MAT && wgsl_type_is_scalar(a));
        if (matrix_scalar) {
            WGSLTypeInfo *scalar = a->kind == WGSL_TYPE_MAT ? b : a;
            if (scalar == tc->types->t_i32 || scalar == tc->types->t_u32) {
                wgsl_tc_error(tc, n,
                    "matrix-scalar multiplication requires a floating-point scalar");
                return 0;
            }
        }
        if (matrix_involved && !matrix_scalar) {
            wgsl_tc_error(tc, n,
                "matrix multiplication operands have incompatible shapes");
            return 0;
        }
    }
    if ((a->kind == WGSL_TYPE_MAT || b->kind == WGSL_TYPE_MAT) &&
        op != WGSL_TOK_PLUS && op != WGSL_TOK_MINUS && op != WGSL_TOK_STAR)
    {
        wgsl_tc_error(tc, n, "operator does not apply to matrix operands");
        return 0;
    }
    if ((op == WGSL_TOK_PLUS || op == WGSL_TOK_MINUS) &&
        ((a->kind == WGSL_TYPE_MAT) != (b->kind == WGSL_TYPE_MAT)))
    {
        wgsl_tc_error(tc, n,
            "matrix addition and subtraction require two matrices");
        return 0;
    }
    /* Component-wise (incl. broadcasts).  Picks the common shape. */
    WGSLTypeInfo *common = common_componentwise(tc, a, b, n);
    if (!common) {
        wgsl_tc_error(tc, n,
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
            wgsl_tc_error(tc, n, "operator requires numeric operands");
            return 0;
        }
        int eval_state = try_consteval_expr_soft(tc, n, &(WGSLValue){0});
        if (eval_state == 0 &&
            (op == WGSL_TOK_SLASH || op == WGSL_TOK_PERCENT))
        {
            wgsl_tc_check_constexpr_divisor_nonzero(
                tc, n->children[1], common,
                op == WGSL_TOK_SLASH ? "division by zero" : "modulo by zero");
        }
        return wgsl_tc_set_type(tc, n, common, 0);

    case WGSL_TOK_AMP: case WGSL_TOK_PIPE: case WGSL_TOK_CARET:
        /* §8.6 allows `&` / `|` on vec<bool> as component-wise
         * logical (the scalar bool case is handled by the bool-pair
         * branch above; we only land here when one side is a vec).
         * Shifts (`<<`, `>>`) remain integer-only. */
        if (elem == tc->types->t_bool) {
            if (op == WGSL_TOK_CARET) {
                wgsl_tc_error(tc, n,
                    "operator '^' does not apply to bool operands");
                return 0;
            }
            if ((a->kind == WGSL_TYPE_VEC) != (b->kind == WGSL_TYPE_VEC)) {
                wgsl_tc_error(tc, n,
                    "bool vector operators require matching shapes");
                return 0;
            }
            return wgsl_tc_set_type(tc, n, common, 0);
        }
        if (!elem || !wgsl_type_is_integer(elem)) {
            wgsl_tc_error(tc, n, "bitwise operator requires integer operands");
            return 0;
        }
        if ((a->kind == WGSL_TYPE_VEC) != (b->kind == WGSL_TYPE_VEC)) {
            wgsl_tc_error(tc, n,
                "bitwise vector operators require matching shapes");
            return 0;
        }
        return wgsl_tc_set_type(tc, n, common, 0);
    /* Shifts handled before unification (above); this branch is dead but
     * kept for the switch's exhaustiveness. */
    case WGSL_TOK_LESS_LESS: case WGSL_TOK_GREATER_GREATER:
        return wgsl_tc_set_type(tc, n, common, 0);

    case WGSL_TOK_EQUAL_EQUAL: case WGSL_TOK_BANG_EQUAL:
    case WGSL_TOK_LESS: case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER: case WGSL_TOK_GREATER_EQUAL: {
        if ((a->kind == WGSL_TYPE_VEC) != (b->kind == WGSL_TYPE_VEC)) {
            wgsl_tc_error(tc, n,
                "comparison operands must have matching shapes");
            return 0;
        }
        /* For scalar operands → bool; for vec operands → vec<width, bool>. */
        if (common->kind == WGSL_TYPE_VEC) {
            return wgsl_tc_set_type(tc, n,
                wgsl_type_vec(tc->types, common->width, tc->types->t_bool), 0);
        }
        return wgsl_tc_set_type(tc, n, tc->types->t_bool, 0);
    }
    default:
        wgsl_tc_error(tc, n, "unsupported binary operator");
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

/* §8.5.1 — `xyzw` and `rgba` letterings must not be mixed within a
 * swizzle.  Returns 1 if every character in `name[0..len-1]` belongs to
 * the same set, 0 otherwise. */
static int swizzle_lettering_consistent(const char *name, uint32_t len) {
    int saw_xyzw = 0, saw_rgba = 0;
    for (uint32_t i = 0; i < len; i++) {
        char c = name[i];
        if (c == 'x' || c == 'y' || c == 'z' || c == 'w') saw_xyzw = 1;
        else if (c == 'r' || c == 'g' || c == 'b' || c == 'a') saw_rgba = 1;
    }
    return !(saw_xyzw && saw_rgba);
}

static int type_member(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count < 2) return 0;
    WGSLNode *obj    = n->children[0];
    WGSLNode *member = n->children[1];
    if (!obj || !member) return 0;
    if (!wgsl_tc_type_expr(tc, obj)) return 0;
    WGSLTypeInfo *ot = wgsl_tc_store_type_of(tc, obj);
    if (!ot) return 0;
    int obj_is_ref = (obj->flags & WGSL_FLAG_IS_REF) != 0;
    if (ot->kind == WGSL_TYPE_PTR) {
        ot = (WGSLTypeInfo *)ot->ref;
        obj_is_ref = 1;
    }

    uint32_t off = (uint32_t)(member->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(member->payload[0] >> 32);
    const char *name = tc->src->bytes + off;

    /* Vector swizzle. */
    if (ot->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)ot->ref;
        if (len < 1 || len > 4) {
            wgsl_tc_error(tc, member, "swizzle must be 1-4 characters");
            return 0;
        }
        for (uint32_t i = 0; i < len; i++) {
            int idx = swizzle_index(name[i]);
            if (idx < 0) {
                wgsl_tc_error(tc, member,
                    "invalid swizzle character '%c'", name[i]);
                return 0;
            }
            if (idx >= ot->width) {
                wgsl_tc_error(tc, member,
                    "swizzle component '%c' out of range for vec%u",
                    name[i], (unsigned)ot->width);
                return 0;
            }
        }
        /* §8.5.1 — `xyzw` and `rgba` letterings must not be mixed. */
        if (!swizzle_lettering_consistent(name, len)) {
            wgsl_tc_error(tc, member,
                "mixed swizzle letterings: components must all be from "
                "the {x,y,z,w} set or the {r,g,b,a} set, not both");
            return 0;
        }
        if (len == 1) {
            /* A single-component swizzle on a Ref<vec> is itself a Ref;
             * a multi-component swizzle is always a value (not LHS). */
            return wgsl_tc_set_type(tc, n, elem, obj_is_ref);
        }
        WGSLTypeInfo *vt = wgsl_type_vec(tc->types, (uint8_t)len, elem);
        return wgsl_tc_set_type(tc, n, vt, 0);
    }

    /* Struct field access. */
    if (ot->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)ot->ref;
        if (!sd) {
            wgsl_tc_error(tc, n, "struct has no field information");
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
                WGSLTypeInfo *ft = wgsl_tc_resolve_type_spec(tc, m->children[attrs]);
                if (!ft) return 0;
                return wgsl_tc_set_type(tc, n, ft, obj_is_ref);
            }
        }
        wgsl_tc_error(tc, member,
            "struct has no field '%.*s'", (int)len, name);
        return 0;
    }

    /* Synthetic atomic CX result struct. */
    if (ot->kind == WGSL_TYPE_ATOMIC_CX_RESULT) {
        if (len == 9 && memcmp(name, "old_value", 9) == 0) {
            return wgsl_tc_set_type(tc, n, (WGSLTypeInfo *)ot->ref, obj_is_ref);
        }
        if (len == 9 && memcmp(name, "exchanged", 9) == 0) {
            return wgsl_tc_set_type(tc, n, tc->types->t_bool, obj_is_ref);
        }
    }

    /* §17 — `__frexp_result<T>` { fract: T, exp: I } where I is i32 if T
     * is scalar, or vecN<i32> matching T's vec width. */
    if (ot->kind == WGSL_TYPE_FREXP_RESULT) {
        WGSLTypeInfo *T = (WGSLTypeInfo *)ot->ref;
        if (len == 5 && memcmp(name, "fract", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        if (len == 3 && memcmp(name, "exp", 3) == 0) {
            WGSLTypeInfo *I = tc->types->t_i32;
            if (T && T->kind == WGSL_TYPE_VEC) {
                I = wgsl_type_vec(tc->types, T->width, tc->types->t_i32);
            }
            return wgsl_tc_set_type(tc, n, I, obj_is_ref);
        }
        wgsl_tc_error(tc, n, "__frexp_result has no field '%.*s' "
                 "(expected 'fract' or 'exp')", (int)len, name);
        return 0;
    }

    /* §17 — `__modf_result<T>` { fract: T, whole: T }. */
    if (ot->kind == WGSL_TYPE_MODF_RESULT) {
        WGSLTypeInfo *T = (WGSLTypeInfo *)ot->ref;
        if (len == 5 && memcmp(name, "fract", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        if (len == 5 && memcmp(name, "whole", 5) == 0) {
            return wgsl_tc_set_type(tc, n, T, obj_is_ref);
        }
        wgsl_tc_error(tc, n, "__modf_result has no field '%.*s' "
                 "(expected 'fract' or 'whole')", (int)len, name);
        return 0;
    }

    /* Permissive fallback: types we don't model precisely yet (e.g.
     * the synthetic `__atomic_compare_exchange_result_T` struct that
     * `atomicCompareExchangeWeak` returns) reach this branch.  Rather
     * than reject member access here, propagate the object's type so
     * the corpus type-checks; Phase 8 will own strict struct field
     * validation with full type context. */
    return wgsl_tc_set_type(tc, n, ot, obj_is_ref);
}

/* ── EXPR_INDEX: vec / mat / array indexing ────────────────────────── */

static int type_index(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        wgsl_tc_error(tc, n, "malformed index expression");
        return 0;
    }
    if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
    if (!wgsl_tc_type_expr(tc, n->children[1])) return 0;
    WGSLTypeInfo *base = wgsl_tc_store_type_of(tc, n->children[0]);
    if (!base) return 0;
    WGSLTypeInfo *index = wgsl_tc_store_type_of(tc, n->children[1]);
    if (!index) return 0;
    int base_is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;
    if (base->kind == WGSL_TYPE_PTR) {
        base = (WGSLTypeInfo *)base->ref;
        base_is_ref = 1;
    }

    WGSLTypeInfo *result = NULL;
    uint32_t bound = 0;          /* 0 = no static bound (runtime array)  */
    const char *kind_name = "";
    switch ((WGSLTypeKind)base->kind) {
    case WGSL_TYPE_VEC:
        result = (WGSLTypeInfo *)base->ref;
        bound = base->width;
        kind_name = "vec";
        break;
    case WGSL_TYPE_MAT:
        /* mat[i] returns a column: vec<rows, T>. */
        result = wgsl_type_vec(tc->types, base->rows, (WGSLTypeInfo *)base->ref);
        bound = base->width;
        kind_name = "matrix";
        break;
    case WGSL_TYPE_ARRAY:
        result = (WGSLTypeInfo *)base->ref;
        bound = base->array_len;  /* 0 → runtime-sized; skip the check.  */
        kind_name = "array";
        break;
    default:
        wgsl_tc_error(tc, n,
            "type %s does not support indexing",
            wgsl_type_kind_name((WGSLTypeKind)base->kind));
        return 0;
    }

    if (!wgsl_type_is_scalar(index) || !wgsl_type_is_integer(index)) {
        char got[96];
        wgsl_type_format(index, got, sizeof got);
        wgsl_tc_error(tc, n->children[1],
            "%s index has type %s; expected integer scalar",
            kind_name, got);
        return 0;
    }

    /* §8.5.{1,2,3} — const-expression index must be in range.  Drive
     * `wgsl_consteval_expr` over the index so we silently roll back any
     * "not a constant" diagnostics for runtime indices, but keep the
     * "out of range" / "negative" diagnostic the materializer emits. */
    if (tc->cev && tc->diag) {
        size_t saved_count = tc->diag->count;
        int    saved_errs  = tc->diag->error_count;
        int    saved_cev   = tc->cev->had_error;
        WGSLValue v = {0};
        if (wgsl_consteval_expr(tc->cev, n->children[1], &v) &&
            v.kind == WGSL_VAL_INT)
        {
            int64_t iv = v.u.i;
            if (iv < 0) {
                wgsl_tc_error(tc, n->children[1],
                    "negative const-expression index '%lld' for %s",
                    (long long)iv, kind_name);
            } else if (bound > 0 && (uint64_t)iv >= bound) {
                wgsl_tc_error(tc, n->children[1],
                    "const-expression index %lld out of range for %s of "
                    "length %u", (long long)iv, kind_name, (unsigned)bound);
            }
        } else {
            /* Index isn't a const-expression — drop any consteval
             * diagnostics emitted along the way; the runtime bound
             * check belongs to the executor, not the type checker. */
            tc->diag->count       = saved_count;
            tc->diag->error_count = saved_errs;
            tc->cev->had_error    = saved_cev;
        }
    }

    /* Indexing into a Ref<…> yields a Ref<elem>; into a value yields a value. */
    return wgsl_tc_set_type(tc, n, result, base_is_ref);
}

/* ── §6.1.2 / §6.1.3 — overload resolution machinery ──────────────── */

static int builtin_call_is_const(const WGSLNode *n);
static int builtin_call_is_override(const WGSLNode *n);

/* A const-expression predicate (§7.2.1 / §15).  Walks `n` and returns 1
 * iff the value is fixed at shader-generation time.  Conservative: any
 * shape we do not explicitly recognize is treated as non-const, which
 * matches the step-2 elimination intent (prefer concrete when in doubt). */
static int is_const_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && is_const_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               is_const_expr(n->children[0]) &&
               is_const_expr(n->children[1]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && s->kind == WGSL_SYM_CONST;
    }
    case WGSL_NODE_EXPR_CALL:
        return builtin_call_is_const(n);
    default:
        return 0;
    }
}

/* §7.2.2 / §15: an override-expression is a superset of const-expression
 * that also accepts references to module-scope `override` symbols.
 * Used to validate `override X = init;` where init may reference other
 * overrides (`override Y = X * 2;` is legal) but not non-const vars or
 * fn params. */
int wgsl_tc_is_override_expr(const WGSLNode *n) {
    if (!n) return 0;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return 1;
    case WGSL_NODE_EXPR_PAREN:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_UNARY:
        return n->child_count == 1 && wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_BINARY:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_INDEX:
        return n->child_count == 2 &&
               wgsl_tc_is_override_expr(n->children[0]) &&
               wgsl_tc_is_override_expr(n->children[1]);
    case WGSL_NODE_EXPR_MEMBER:
        return n->child_count >= 1 &&
               wgsl_tc_is_override_expr(n->children[0]);
    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode *)n);
        return s && (s->kind == WGSL_SYM_CONST || s->kind == WGSL_SYM_OVERRIDE);
    }
    case WGSL_NODE_EXPR_CALL:
        return builtin_call_is_override(n);
    default:
        return 0;
    }
}

int wgsl_typecheck_is_override_expr(const WGSLNode *n) {
    return wgsl_tc_is_override_expr(n);
}

/* Per-arg parameter family.  Encodes the set of scalar element types T
 * that a builtin's parametric overload set ranges over.  When the field
 * is `PF_NONE`, the entry skips full overload resolution and uses the
 * legacy BR_* return rule directly. */
typedef enum {
    PF_NONE = 0,
    PF_FLOAT_T,         /* T ∈ {AbstractFloat, f32, f16}                */
    PF_NUMERIC_T,       /* T ∈ {AbstractInt, AbstractFloat, i32, u32,   *
                         *      f32, f16}                               */
    PF_INTEGER_T,       /* T ∈ {AbstractInt, i32, u32}                  */
    PF_SIGN_T,          /* T ∈ {AbstractInt, AbstractFloat, i32, f32, f16} */
    PF_F32_T,           /* T ∈ {f32}; abstract args may materialize to f32 */
} WGSLParamFamily;

/* Maximum arity / candidate count tracked by the resolver.  Wide enough
 * for every numeric builtin in the WGSL spec; bumping the candidate cap
 * is cheap (stack-only) if a future family needs it. */
#define WGSL_OL_MAX_PARAMS 4
#define WGSL_OL_MAX_CANDS  8

typedef struct {
    WGSLTypeInfo *params[WGSL_OL_MAX_PARAMS];
    WGSLTypeInfo *result;   /* shape used to derive the return type     */
} WGSLOLCandidate;

/* §6.1.3 — overload resolution.
 *
 *  Step 1: drop candidates with an infeasible conversion on any arg.
 *  Step 2: drop candidates where one parameter is abstract while
 *          another argument is not a const-expression.
 *  Step 3: rank candidates pairwise — C1 is preferred over C2 iff
 *          R_C1[i] ≤ R_C2[i] for every i, with strict inequality on
 *          at least one i.
 *  Step 4: if exactly one candidate is preferred over every other
 *          survivor, succeed.  Otherwise fail.
 *
 * Returns the chosen index or -1 on failure.  Emits a diagnostic on `at`
 * for "no matching overload" and "ambiguous overload". */
static int resolve_overload(
    WGSLTypeChecker *tc,
    const WGSLOLCandidate *cands, int cand_count,
    WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count,
    const char *what, const WGSLNode *at)
{
    if (cand_count <= 0 || arg_count <= 0 || arg_count > WGSL_OL_MAX_PARAMS) {
        return -1;
    }

    int ranks[WGSL_OL_MAX_CANDS][WGSL_OL_MAX_PARAMS];
    int alive[WGSL_OL_MAX_CANDS] = {0};
    int alive_count = 0;

    /* Step 1: conversion-rank feasibility. */
    for (int c = 0; c < cand_count; c++) {
        alive[c] = 1;
        for (int i = 0; i < arg_count; i++) {
            int r = wgsl_type_conversion_rank(arg_types[i], cands[c].params[i]);
            if (r < 0) { alive[c] = 0; break; }
            ranks[c][i] = r;
        }
        if (alive[c]) alive_count += 1;
    }
    if (alive_count == 0) {
        wgsl_tc_error(tc, at, "no matching overload for '%s'", what);
        return -1;
    }

    /* Step 2: eliminate abstract-after-conversion + non-const-elsewhere. */
    if (arg_count > 1) {
        int has_nonconst = 0;
        for (int j = 0; j < arg_count; j++) {
            if (!arg_is_const[j]) { has_nonconst = 1; break; }
        }
        if (has_nonconst) {
            for (int c = 0; c < cand_count; c++) {
                if (!alive[c]) continue;
                int any_abstract_param = 0;
                for (int i = 0; i < arg_count; i++) {
                    if (wgsl_type_is_abstract(cands[c].params[i])) {
                        any_abstract_param = 1;
                        break;
                    }
                }
                if (any_abstract_param) {
                    alive[c] = 0;
                    alive_count -= 1;
                }
            }
        }
        if (alive_count == 0) {
            wgsl_tc_error(tc, at, "no matching overload for '%s'", what);
            return -1;
        }
    }

    /* Step 3+4: pairwise preference, pick unique winner. */
    int best = -1;
    for (int c = 0; c < cand_count; c++) {
        if (!alive[c]) continue;
        int preferred_over_all = 1;
        for (int o = 0; o < cand_count; o++) {
            if (o == c || !alive[o]) continue;
            int all_le = 1, some_lt = 0;
            for (int i = 0; i < arg_count; i++) {
                if (ranks[c][i] > ranks[o][i]) { all_le = 0; break; }
                if (ranks[c][i] < ranks[o][i]) some_lt = 1;
            }
            if (!(all_le && some_lt)) { preferred_over_all = 0; break; }
        }
        if (preferred_over_all) {
            if (best == -1) {
                best = c;
            } else {
                /* Multiple candidates each preferred over all others —
                 * impossible by the partial-order definition, but treat
                 * as ambiguous to be safe. */
                best = -1;
                break;
            }
        }
    }
    if (best == -1) {
        wgsl_tc_error(tc, at, "ambiguous overload for '%s'", what);
        return -1;
    }
    return best;
}

/* Lift a scalar T into the shape carried by `shape` (scalar / vec / mat). */
static WGSLTypeInfo *lift_to_shape(
    WGSLTypeStore *ts, WGSLTypeInfo *T, const WGSLTypeInfo *shape)
{
    if (!shape || !T) return T;
    if (shape->kind == WGSL_TYPE_VEC) {
        return wgsl_type_vec(ts, shape->width, T);
    }
    if (shape->kind == WGSL_TYPE_MAT) {
        return wgsl_type_mat(ts, shape->width, shape->rows, T);
    }
    return T;
}

/* Pick a "shape source" arg — prefer vec/mat over scalar so a mixed
 * scalar/composite call still finds the composite shape. */
static const WGSLTypeInfo *pick_shape_arg(
    WGSLTypeInfo *const *args, int n)
{
    for (int i = 0; i < n; i++) {
        if (!args[i]) continue;
        if (args[i]->kind == WGSL_TYPE_VEC || args[i]->kind == WGSL_TYPE_MAT) {
            return args[i];
        }
    }
    return n > 0 ? args[0] : NULL;
}

/* Build the candidate set for a "T-pattern" builtin: all `arity` params
 * share the same shape (derived from the args), with the element ranging
 * over the family's scalar set.  Returns the number of candidates. */
static int build_t_candidates(
    WGSLTypeChecker *tc, WGSLParamFamily fam,
    WGSLTypeInfo *const *args, int arity,
    WGSLOLCandidate *out_cands)
{
    const WGSLTypeInfo *shape = pick_shape_arg(args, arity);
    if (!shape) return 0;

    WGSLTypeInfo *scalars[8];
    int sc = 0;
    switch (fam) {
    case PF_FLOAT_T:
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_NUMERIC_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_INTEGER_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_u32;
        break;
    case PF_SIGN_T:
        scalars[sc++] = tc->types->t_abstract_int;
        scalars[sc++] = tc->types->t_abstract_float;
        scalars[sc++] = tc->types->t_i32;
        scalars[sc++] = tc->types->t_f32;
        if (tc->enabled_extensions & WGSL_EXT_F16) {
            scalars[sc++] = tc->types->t_f16;
        }
        break;
    case PF_F32_T:
        scalars[sc++] = tc->types->t_f32;
        break;
    default:
        return 0;
    }

    int n = 0;
    for (int i = 0; i < sc && n < WGSL_OL_MAX_CANDS; i++) {
        WGSLTypeInfo *lifted = lift_to_shape(tc->types, scalars[i], shape);
        WGSLOLCandidate *cnd = &out_cands[n];
        for (int a = 0; a < arity; a++) cnd->params[a] = lifted;
        cnd->result = lifted;
        n += 1;
    }
    return n;
}

/* ── Builtin overload table ────────────────────────────────────────── */

/* Return-type derivation rule for each builtin entry. */
typedef enum {
    BR_VOID,                /* () return                                */
    BR_BOOL,                /* always bool                              */
    BR_I32,                 /* always i32                               */
    BR_U32,                 /* always u32 (`arrayLength`, `textureNum*`)*/
    BR_ARG0,                /* same type as first argument              */
    BR_ARG0_ELEM,           /* element type of first argument           */
    BR_TEMPLATE,            /* template arg of a templated callee       */
    BR_VEC4F,               /* vec4<f32> (texture sampling default)     */
    BR_VEC4U,               /* vec4<u32> (subgroupBallot, unpack4xU8)   */
    BR_VEC4I,               /* vec4<i32> (unpack4xI8)                   */
    BR_VEC2F,               /* vec2<f32> (unpack2x16{s,u}norm/float)    */
    BR_ATOMIC_CX_RESULT,    /* __atomic_compare_exchange_result<T>      */
    BR_FREXP_RESULT,        /* __frexp_result<T> for T = arg0's type    */
    BR_MODF_RESULT,         /* __modf_result<T>  for T = arg0's type    */
} WGSLBuiltinReturn;

typedef enum {
    WGSL_BIF_CONST    = 1u << 0,
    WGSL_BIF_MUST_USE = 1u << 1,
} WGSLBuiltinFlags;

typedef enum {
    BS_NONE = 0,
    BS_LDEXP,
    BS_MIX,
    BS_TEXTURE,
} WGSLBuiltinSpecial;

typedef struct {
    const char       *name;
    WGSLBuiltinReturn ret;
    WGSLParamFamily   fam;     /* PF_NONE → skip full overload resolution */
    uint8_t           arity;   /* number of leading args that share the T */
    uint16_t          flags;   /* WGSL_BIF_* metadata from def/wgsl.def */
    uint8_t           special; /* WGSLBuiltinSpecial validator hook */
} WGSLBuiltinEntry;

static void check_const_builtin_call(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinEntry *entry)
{
    if (!entry || !(entry->flags & WGSL_BIF_CONST)) return;
    if (!is_const_expr(call)) return;
    WGSLValue ignored = {0};
    (void)try_consteval_expr_soft(tc, call, &ignored);
}

typedef enum {
    BOP_VOID_0 = 0,
    BOP_ALL_ANY,
    BOP_SELECT,
    BOP_ARRAY_LENGTH,
    BOP_WORKGROUP_UNIFORM_LOAD,
    BOP_T_FLOAT_1,
    BOP_T_FLOAT_2,
    BOP_T_FLOAT_3,
    BOP_T_NUMERIC_1,
    BOP_T_NUMERIC_2,
    BOP_T_NUMERIC_3,
    BOP_T_SIGN_1,
    BOP_T_INTEGER_1,
    BOP_EXTRACT_BITS,
    BOP_INSERT_BITS,
    BOP_LDEXP,
    BOP_MIX,
    BOP_DOT,
    BOP_DOT4_U32,
    BOP_LENGTH,
    BOP_DETERMINANT,
    BOP_CROSS,
    BOP_TRANSPOSE,
    BOP_VEC_FLOAT_1,
    BOP_VEC_FLOAT_2,
    BOP_VEC_FLOAT_3,
    BOP_REFRACT,
    BOP_T_F32_1,
    BOP_ARG_U32_1,
    BOP_ARG_VEC4F_1,
    BOP_ARG_VEC4I_1,
    BOP_ARG_VEC4U_1,
    BOP_ARG_VEC2F_1,
    BOP_ATOMIC_LOAD,
    BOP_ATOMIC_STORE,
    BOP_ATOMIC_RMW,
    BOP_ATOMIC_CX,
    BOP_BOOL_1,
    BOP_T_NUMERIC_U32_2,
} WGSLBuiltinOverloadPattern;

typedef struct {
    const char                 *name;
    WGSLBuiltinOverloadPattern  pattern;
    WGSLBuiltinReturn           ret;
    uint16_t                    flags;
    uint8_t                     special;
} WGSLBuiltinOverload;

typedef enum {
    BTOV_SAMPLE = 0,
    BTOV_SAMPLE_OFFSET,
    BTOV_SAMPLE_LEVEL,
    BTOV_SAMPLE_LEVEL_OFFSET,
    BTOV_SAMPLE_BIAS,
    BTOV_SAMPLE_BIAS_OFFSET,
    BTOV_SAMPLE_GRAD,
    BTOV_SAMPLE_GRAD_OFFSET,
    BTOV_SAMPLE_COMPARE,
    BTOV_SAMPLE_COMPARE_OFFSET,
    BTOV_SAMPLE_COMPARE_LEVEL,
    BTOV_SAMPLE_COMPARE_LEVEL_OFFSET,
    BTOV_SAMPLE_BASE_CLAMP,
    BTOV_LOAD,
    BTOV_STORE,
    BTOV_GATHER_COMPONENT,
    BTOV_GATHER_COMPONENT_OFFSET,
    BTOV_GATHER_DEPTH,
    BTOV_GATHER_DEPTH_OFFSET,
    BTOV_GATHER_COMPARE,
    BTOV_GATHER_COMPARE_OFFSET,
    BTOV_DIMENSIONS,
    BTOV_DIMENSIONS_LEVEL,
    BTOV_NUM_LAYERS,
    BTOV_NUM_LEVELS,
    BTOV_NUM_SAMPLES,
} WGSLTextureOverloadKind;

typedef enum {
    BTEXEL_ANY = 0,
    BTEXEL_F32,
    BTEXEL_I32,
    BTEXEL_U32,
    BTEXEL_DEPTH,
    BTEXEL_EXTERNAL,
    BTEXEL_STORAGE_ANY,
    BTEXEL_STORAGE_F32,
    BTEXEL_STORAGE_I32,
    BTEXEL_STORAGE_U32,
} WGSLTextureOverloadElem;

typedef struct {
    const char              *name;
    WGSLTextureOverloadKind  kind;
    uint8_t                  dim;       /* WGSLTextureDim */
    uint8_t                  elem;      /* WGSLTextureOverloadElem */
    uint8_t                  access;    /* WGSLAccessMode; NONE = don't care */
} WGSLTextureOverload;

/* Generated from def/wgsl.def.  The fam / arity fields drive §6.1.3
 * overload resolution; the flags field carries explicit @const / @must_use
 * metadata, and special hooks cover mixed-shape overloads that do not fit
 * the compact T-pattern. */
#include "check/builtins.gen.h"

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

int wgsl_tc_builtin_must_use(const char *name, uint32_t name_len) {
    const WGSLBuiltinEntry *e = find_builtin_entry(name, name_len);
    return e && (e->flags & WGSL_BIF_MUST_USE);
}

static int sym_is_constructor_callable(const WGSLSymbol *sym) {
    if (!sym) return 0;
    switch ((WGSLSymKind)sym->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPEGEN:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_TYPE_ALIAS:
    case WGSL_SYM_STRUCT:
        return 1;
    default:
        return 0;
    }
}

static int builtin_call_is_const(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!is_const_expr(n->children[i])) return 0;
    }
    return 1;
}

static int builtin_call_is_override(const WGSLNode *n) {
    if (!n || n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 1) return 0;
    const WGSLNode *callee = n->children[0];
    if (!callee) return 0;

    WGSLSymbol *sym = NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        sym = wgsl_node_resolved_symbol((WGSLNode *)callee);
    } else if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
               callee->child_count >= 1 &&
               callee->children[0])
    {
        sym = wgsl_node_resolved_symbol(callee->children[0]);
    }
    if (!sym) return 0;
    if (sym->kind == WGSL_SYM_PREDECLARED_FN) {
        const WGSLBuiltinEntry *e = find_builtin_entry(sym->name, sym->name_len);
        if (!e || !(e->flags & WGSL_BIF_CONST)) return 0;
    } else if (!sym_is_constructor_callable(sym)) {
        return 0;
    }
    for (uint32_t i = 1; i < n->child_count; i++) {
        if (!wgsl_tc_is_override_expr(n->children[i])) return 0;
    }
    return 1;
}

/* §17.8 — atomic builtins take a pointer to `atomic<T>` as the first
 * argument; subsequent value args must match `T`.  Shape table:
 *   atomicLoad(ptr)                              → T
 *   atomicStore(ptr, value:T)                    → ()
 *   atomicAdd / Sub / Max / Min / And / Or / Xor (ptr, value:T) → T
 *   atomicExchange(ptr, value:T)                 → T
 *   atomicCompareExchangeWeak(ptr, cmp:T, val:T) → __atomic_cx_result<T>
 *
 * The pointer's address space is restricted to `storage` or `workgroup`
 * by §6.2.8 + §14.3 (already enforced at the decl site).  Here we only
 * verify (1) pointee is `atomic<T>`, (2) value args have type `T`. */
typedef enum {
    AT_KIND_LOAD,             /* (ptr) → T */
    AT_KIND_STORE,            /* (ptr, T) → () */
    AT_KIND_RMW,              /* (ptr, T) → T  (Add/Sub/Max/Min/.../Exchange) */
    AT_KIND_CX,               /* (ptr, T, T) → __atomic_cx_result<T> */
} WGSLAtomicKind;

static int classify_atomic(const char *name, uint32_t len, WGSLAtomicKind *out) {
    if (len < 6 || memcmp(name, "atomic", 6) != 0) return 0;
    if (len == 10 && memcmp(name, "atomicLoad", 10) == 0)               { *out = AT_KIND_LOAD;  return 1; }
    if (len == 11 && memcmp(name, "atomicStore", 11) == 0)              { *out = AT_KIND_STORE; return 1; }
    if (len == 25 && memcmp(name, "atomicCompareExchangeWeak", 25) == 0){ *out = AT_KIND_CX;    return 1; }
    /* atomicAdd / Sub / Max / Min / And / Or / Xor / Exchange */
    *out = AT_KIND_RMW;
    return 1;
}

/* Pull the atomic<T>'s element T plus pointer qualifiers from the first arg.
 * Atomic builtins require a pointer value, not an atomic<T> value. */
static int atomic_ptr_info(
    WGSLTypeInfo *arg0,
    WGSLTypeInfo **out_T,
    WGSLAddressSpace *out_as,
    WGSLAccessMode *out_am)
{
    if (!arg0 || arg0->kind != WGSL_TYPE_PTR) return 0;
    WGSLTypeInfo *pointee = (WGSLTypeInfo *)arg0->ref;
    if (!pointee || pointee->kind != WGSL_TYPE_ATOMIC) return 0;
    if (out_T)  *out_T  = (WGSLTypeInfo *)pointee->ref;
    if (out_as) *out_as = (WGSLAddressSpace)(arg0->flags & 0xFFu);
    if (out_am) *out_am = (WGSLAccessMode)((arg0->flags >> 8) & 0xFFu);
    return 1;
}

static int atomic_ptr_access_ok(WGSLAddressSpace as, WGSLAccessMode am) {
    return (as == WGSL_AS_WORKGROUP || as == WGSL_AS_STORAGE) &&
           am == WGSL_ACCESS_READ_WRITE;
}

static WGSLTypeInfo *atomic_pointee_T(WGSLTypeInfo *arg0) {
    WGSLTypeInfo *T = NULL;
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode am = WGSL_ACCESS_NONE;
    if (!atomic_ptr_info(arg0, &T, &as, &am)) return NULL;
    return atomic_ptr_access_ok(as, am) ? T : NULL;
}

static const char *type_label(WGSLTypeInfo *t, char *buf, size_t cap) {
    if (!t) return "(no type)";
    if (!buf || cap == 0) return wgsl_type_kind_name((WGSLTypeKind)t->kind);
    size_t n = wgsl_type_format(t, buf, cap);
    if (n >= cap) n = cap - 1;
    buf[n] = '\0';
    return buf;
}

static int bitcast_scalar_bits(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
        return 32;
    case WGSL_TYPE_F16:
        return 16;
    default:
        return 0;
    }
}

static int bitcast_type_bits(const WGSLTypeInfo *t) {
    if (!t) return 0;
    int scalar_bits = 0;
    if (wgsl_type_is_scalar(t)) {
        scalar_bits = bitcast_scalar_bits(t);
        return (scalar_bits % 32) == 0 ? scalar_bits : 0;
    }
    if (t->kind == WGSL_TYPE_VEC) {
        scalar_bits = bitcast_scalar_bits((const WGSLTypeInfo *)t->ref);
        int total = scalar_bits * (int)t->width;
        return (total % 32) == 0 ? total : 0;
    }
    return 0;
}

static void check_atomic_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0_type)
{
    WGSLAtomicKind kind;
    if (!classify_atomic(name, name_len, &kind)) return;

    /* Expected arg count: 1 for Load; 2 for Store/RMW; 3 for CX. */
    int want_args =
        kind == AT_KIND_LOAD  ? 1 :
        kind == AT_KIND_STORE ? 2 :
        kind == AT_KIND_CX    ? 3 : 2;
    int got_args = (int)call->child_count - 1;   /* minus callee */
    if (got_args != want_args) {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d argument%s; got %d",
            (int)name_len, name, want_args, want_args == 1 ? "" : "s", got_args);
        return;
    }

    /* Arg 0 must be a read_write pointer to atomic<T> in a shared address
     * space. */
    WGSLTypeInfo *T = NULL;
    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode am = WGSL_ACCESS_NONE;
    if (!atomic_ptr_info(arg0_type, &T, &as, &am)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a pointer to atomic<T> as the first "
            "argument; got %s",
            (int)name_len, name,
            type_label(arg0_type, got, sizeof got));
        return;
    }
    if (!atomic_ptr_access_ok(as, am)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' first argument must be ptr<workgroup, atomic<T>> "
            "or ptr<storage, atomic<T>, read_write>; got %s",
            (int)name_len, name,
            type_label(arg0_type, got, sizeof got));
        return;
    }

    /* Validate value args (positions depend on kind).  For each, the
     * arg's type must be feasibly convertible to T. */
    int n_values =
        kind == AT_KIND_LOAD ? 0 :
        kind == AT_KIND_CX   ? 2 : 1;
    for (int i = 0; i < n_values; i++) {
        WGSLNode *arg = call->children[2 + i];
        WGSLTypeInfo *vt = wgsl_tc_store_type_of(tc, arg);
        if (!vt) continue;
        if (wgsl_tc_init_type_mismatch(vt, T)) {
            char got[96], want[96];
            wgsl_tc_error(tc, arg,
                "'%.*s' value argument has type %s; expected %s "
                "(matching the atomic's element type)",
                (int)name_len, name,
                type_label(vt, got, sizeof got),
                type_label(T, want, sizeof want));
        }
    }
}

/* §17.9 / §17.10 — pack* / unpack* builtins take a fixed-shape arg.
 * Return 1 if the call's first arg matches the expected shape, 0
 * otherwise (after emitting a diagnostic with the expected shape).
 * Shapes:
 *   pack4x8snorm/unorm           → vec4<f32>
 *   pack4xI8 / pack4xI8Clamp     → vec4<i32>
 *   pack4xU8 / pack4xU8Clamp     → vec4<u32>
 *   pack2x16snorm/unorm/float    → vec2<f32>
 *   unpack* (all 7)              → u32                                */
static int check_pack_unpack_arg(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0)
{
    if (!arg0) return 1;   /* arity-mismatch handled elsewhere */

    /* Determine expected shape from name. */
    const WGSLTypeInfo *want_vec_elem = NULL;
    int want_vec_w = 0;
    int want_u32   = 0;

    /* All unpack* take u32. */
    if (name_len >= 6 && memcmp(name, "unpack", 6) == 0) {
        want_u32 = 1;
    }
    /* pack4x8snorm / pack4x8unorm → vec4<f32> */
    else if (name_len == 12 &&
             (memcmp(name, "pack4x8snorm", 12) == 0 ||
              memcmp(name, "pack4x8unorm", 12) == 0))
    {
        want_vec_elem = tc->types->t_f32;
        want_vec_w    = 4;
    }
    /* pack4xI8 / pack4xI8Clamp → vec4<i32> */
    else if ((name_len == 8  && memcmp(name, "pack4xI8", 8) == 0) ||
             (name_len == 13 && memcmp(name, "pack4xI8Clamp", 13) == 0))
    {
        want_vec_elem = tc->types->t_i32;
        want_vec_w    = 4;
    }
    /* pack4xU8 / pack4xU8Clamp → vec4<u32> */
    else if ((name_len == 8  && memcmp(name, "pack4xU8", 8) == 0) ||
             (name_len == 13 && memcmp(name, "pack4xU8Clamp", 13) == 0))
    {
        want_vec_elem = tc->types->t_u32;
        want_vec_w    = 4;
    }
    /* pack2x16{s,u}norm / pack2x16float → vec2<f32> */
    else if ((name_len == 13 && memcmp(name, "pack2x16snorm", 13) == 0) ||
             (name_len == 13 && memcmp(name, "pack2x16unorm", 13) == 0) ||
             (name_len == 13 && memcmp(name, "pack2x16float", 13) == 0))
    {
        want_vec_elem = tc->types->t_f32;
        want_vec_w    = 2;
    } else {
        return 1;   /* not a pack/unpack name */
    }

    /* Validate. */
    if (want_u32) {
        if (arg0 != tc->types->t_u32 &&
            arg0 != tc->types->t_abstract_int)
        {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a u32 argument; got %s",
                (int)name_len, name,
                wgsl_type_kind_name((WGSLTypeKind)arg0->kind));
            return 0;
        }
        return 1;
    }

    if (want_vec_w > 0 && want_vec_elem) {
        WGSLTypeInfo *want = wgsl_type_vec(
            tc->types, (uint8_t)want_vec_w, (WGSLTypeInfo *)want_vec_elem);
        int ok = arg0 && want && wgsl_type_conversion_rank(arg0, want) >= 0;
        if (!ok) {
            const char *elem_name =
                want_vec_elem == tc->types->t_f32 ? "f32" :
                want_vec_elem == tc->types->t_i32 ? "i32" :
                want_vec_elem == tc->types->t_u32 ? "u32" : "?";
            wgsl_tc_error(tc, call,
                "'%.*s' requires a vec%d<%s> argument; got %s",
                (int)name_len, name, want_vec_w, elem_name,
                wgsl_type_kind_name((WGSLTypeKind)arg0->kind));
            return 0;
        }
    }
    return 1;
}

static int builtin_name_eq(const char *name, uint32_t len, const char *lit) {
    size_t n = strlen(lit);
    return len == n && memcmp(name, lit, n) == 0;
}

static int builtin_arg_count(const WGSLNode *call) {
    return call && call->child_count > 0 ? (int)call->child_count - 1 : 0;
}

static int builtin_arg_count_is(WGSLNode *call, int n) {
    return builtin_arg_count(call) == n;
}

static WGSLTypeInfo *builtin_arg_type(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index)
{
    if (!call || arg_index < 0) return NULL;
    uint32_t child_index = (uint32_t)arg_index + 1u;
    if (child_index >= call->child_count) return NULL;
    return wgsl_tc_store_type_of(tc, call->children[child_index]);
}

static int shape_width(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_VEC) return (int)t->width;
    if (wgsl_type_is_scalar(t)) return 1;
    return 0;
}

static WGSLTypeInfo *shape_elem(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (t->kind == WGSL_TYPE_VEC) return (WGSLTypeInfo *)t->ref;
    if (wgsl_type_is_scalar(t)) return (WGSLTypeInfo *)(uintptr_t)t;
    return NULL;
}

static int type_converts_to(WGSLTypeInfo *from, WGSLTypeInfo *to) {
    return from && to && wgsl_type_conversion_rank(from, to) >= 0;
}

static int expect_arg_count(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, int min_args, int max_args)
{
    int got = builtin_arg_count(call);
    if (got >= min_args && got <= max_args) return 1;
    if (min_args == max_args) {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d argument%s; got %d",
            (int)name_len, name, min_args, min_args == 1 ? "" : "s", got);
    } else {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d-%d arguments; got %d",
            (int)name_len, name, min_args, max_args, got);
    }
    return 0;
}

static int expect_sampler_arg(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int comparison)
{
    if (t && t->kind == WGSL_TYPE_SAMPLER && ((int)t->width != 0) == comparison) {
        return 1;
    }
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' requires %s as the sampler argument; got %s",
        (int)name_len, name,
        comparison ? "sampler_comparison" : "sampler",
        type_label(t, got, sizeof got));
    return 0;
}

static int expect_float_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *want = width == 1
        ? tc->types->t_f32
        : wgsl_type_vec(tc->types, (uint8_t)width, tc->types->t_f32);
    if (type_converts_to(t, want)) return 1;
    char got[96], exp[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        type_label(want, exp, sizeof exp));
    return 0;
}

static int expect_integer_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem && wgsl_type_is_integer(elem) && shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %sinteger %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "an " : "a ",
        width == 1 ? "scalar" : "vector with matching texture dimensionality");
    return 0;
}

static int expect_i32_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem &&
             (elem == tc->types->t_abstract_int || elem == tc->types->t_i32) &&
             shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "i32 or AbstractInt"
                   : "vecN<i32> or vecN<AbstractInt> with matching offset dimensionality");
    return 0;
}

static int expect_value_convertible(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *got_t, WGSLTypeInfo *want_t, const char *role)
{
    if (type_converts_to(got_t, want_t)) return 1;
    char got[96], want[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(got_t, got, sizeof got),
        type_label(want_t, want, sizeof want));
    return 0;
}

static int eval_const_int_silent(
    WGSLTypeChecker *tc, WGSLNode *n, int64_t *out)
{
    if (!tc || !tc->cev || !tc->diag || !n || !out) return 0;
    size_t saved_count = tc->diag->count;
    int saved_errors = tc->diag->error_count;
    int saved_cev = tc->cev->had_error;
    WGSLValue v = {0};
    int ok = wgsl_consteval_expr(tc->cev, n, &v) && v.kind == WGSL_VAL_INT;
    tc->diag->count = saved_count;
    tc->diag->error_count = saved_errors;
    tc->cev->had_error = saved_cev;
    if (!ok) return 0;
    *out = v.u.i;
    return 1;
}

static int collect_const_int_components(
    WGSLTypeChecker *tc, WGSLNode *n, int width, int64_t *out)
{
    if (!n || !out || width < 1 || width > 4) return 0;
    if (width == 1) return eval_const_int_silent(tc, n, &out[0]);
    if (n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 2) return 0;

    int arg_count = (int)n->child_count - 1;
    if (arg_count == 1) {
        int64_t v = 0;
        if (!eval_const_int_silent(tc, n->children[1], &v)) return 0;
        for (int i = 0; i < width; i++) out[i] = v;
        return 1;
    }
    if (arg_count != width) return 0;
    for (int i = 0; i < width; i++) {
        if (!eval_const_int_silent(tc, n->children[1 + i], &out[i])) {
            return 0;
        }
    }
    return 1;
}

static int check_const_int_range(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    int64_t lo, int64_t hi, const char *role)
{
    int64_t v = 0;
    if (!eval_const_int_silent(tc, at, &v)) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s must be a const-expression",
            (int)name_len, name, role);
        return 0;
    }
    if (v < lo || v > hi) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s value %lld is outside [%lld, %lld]",
            (int)name_len, name, role,
            (long long)v, (long long)lo, (long long)hi);
        return 0;
    }
    return 1;
}

static int texture_is_depth(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_DEPTH_2D &&
           d <= WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

static int texture_is_storage(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_STORAGE_1D && d <= WGSL_TEX_DIM_STORAGE_3D;
}

static int texture_is_sampled(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_1D && d <= WGSL_TEX_DIM_MULTISAMPLED_2D;
}

static int texture_is_multisampled(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_MULTISAMPLED_2D ||
           d == WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

static int texture_is_arrayed(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_STORAGE_2D_ARRAY;
}

static int texture_is_gather_dim(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY;
}

static int texture_allows_offset(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_3D ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY;
}

static int texture_offset_width(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_3D ? 3 : 2;
}

static int texture_coord_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_CUBE:
    case WGSL_TEX_DIM_CUBE_ARRAY:
    case WGSL_TEX_DIM_DEPTH_CUBE:
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

static int texture_dimensions_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

static WGSLTypeInfo *storage_texture_elem(
    WGSLTypeChecker *tc, WGSLTexelFormat f)
{
    switch (f) {
    case WGSL_TEX_FORMAT_R8UINT:
    case WGSL_TEX_FORMAT_R16UINT:
    case WGSL_TEX_FORMAT_RG8UINT:
    case WGSL_TEX_FORMAT_R32UINT:
    case WGSL_TEX_FORMAT_RG16UINT:
    case WGSL_TEX_FORMAT_RGBA8UINT:
    case WGSL_TEX_FORMAT_RG32UINT:
    case WGSL_TEX_FORMAT_RGBA16UINT:
    case WGSL_TEX_FORMAT_RGBA32UINT:
    case WGSL_TEX_FORMAT_RGB10A2UINT:
        return tc->types->t_u32;
    case WGSL_TEX_FORMAT_R8SINT:
    case WGSL_TEX_FORMAT_R16SINT:
    case WGSL_TEX_FORMAT_RG8SINT:
    case WGSL_TEX_FORMAT_R32SINT:
    case WGSL_TEX_FORMAT_RG16SINT:
    case WGSL_TEX_FORMAT_RGBA8SINT:
    case WGSL_TEX_FORMAT_RG32SINT:
    case WGSL_TEX_FORMAT_RGBA16SINT:
    case WGSL_TEX_FORMAT_RGBA32SINT:
        return tc->types->t_i32;
    default:
        return tc->types->t_f32;
    }
}

static WGSLTypeInfo *texture_value_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureSampleCompare") ||
        builtin_name_eq(name, name_len, "textureSampleCompareLevel"))
    {
        return tc->types->t_f32;
    }
    if (builtin_name_eq(name, name_len, "textureGatherCompare"))
    {
        return wgsl_type_vec(tc->types, 4, tc->types->t_f32);
    }
    if (builtin_name_eq(name, name_len, "textureGather")) {
        WGSLTypeInfo *elem = texture_is_storage(dim)
            ? storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len)
            : (WGSLTypeInfo *)tex->ref;
        if (!elem || texture_is_depth(dim) || dim == WGSL_TEX_DIM_EXTERNAL) {
            elem = tc->types->t_f32;
        }
        return wgsl_type_vec(tc->types, 4, elem);
    }
    if (texture_is_depth(dim)) {
        return tc->types->t_f32;
    }
    WGSLTypeInfo *elem = NULL;
    if (texture_is_storage(dim)) {
        elem = storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len);
    } else if (dim == WGSL_TEX_DIM_EXTERNAL) {
        elem = tc->types->t_f32;
    } else {
        elem = (WGSLTypeInfo *)tex->ref;
        if (!elem) elem = tc->types->t_f32;
    }
    return wgsl_type_vec(tc->types, 4, elem);
}

static WGSLTypeInfo *texture_builtin_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (builtin_name_eq(name, name_len, "textureStore")) {
        return tc->types->t_void;
    }
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureDimensions")) {
        int w = texture_dimensions_width(dim);
        return w == 1 ? tc->types->t_u32
                      : wgsl_type_vec(tc->types, (uint8_t)w, tc->types->t_u32);
    }
    if (builtin_name_eq(name, name_len, "textureNumLayers") ||
        builtin_name_eq(name, name_len, "textureNumLevels") ||
        builtin_name_eq(name, name_len, "textureNumSamples"))
    {
        return tc->types->t_u32;
    }
    return texture_value_result(tc, name, name_len, tex);
}

static WGSLTypeInfo *texture_arg_type_for_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type)
{
    if (builtin_name_eq(name, name_len, "textureGather") &&
        (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE))
    {
        return builtin_arg_type(tc, call, 1);
    }
    return arg0_type;
}

static void check_ldexp_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type)
{
    if (!expect_arg_count(tc, call, name, name_len, 2, 2)) return;
    WGSLTypeInfo *x = arg0_type;
    WGSLTypeInfo *e = builtin_arg_type(tc, call, 1);
    WGSLTypeInfo *xe = shape_elem(x);
    WGSLTypeInfo *ee = shape_elem(e);
    int xw = shape_width(x);
    int ew = shape_width(e);
    if (!xe || !wgsl_type_is_float(xe) || xw == 0) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' first argument has type %s; expected a floating-point "
            "scalar or vector",
            (int)name_len, name, type_label(x, got, sizeof got));
        return;
    }
    if (!ee || !(ee == tc->types->t_abstract_int || ee == tc->types->t_i32) ||
        ew != xw)
    {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' exponent argument has type %s; expected %s",
            (int)name_len, name, type_label(e, got, sizeof got),
            xw == 1 ? "AbstractInt or i32"
                    : "vecN<AbstractInt> or vecN<i32> matching the first argument");
        return;
    }

    int64_t exps[4] = {0};
    int max_exp = xe == tc->types->t_f16 ? 16 :
                  xe == tc->types->t_f32 ? 128 : 1024;
    if (collect_const_int_components(tc, call->children[2], xw, exps)) {
        for (int i = 0; i < xw; i++) {
            if (exps[i] > max_exp) {
                wgsl_tc_error(tc, call,
                    "builtin 'ldexp' domain error: exponent too large");
                return;
            }
        }
    }
}

static void check_mix_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *resolved_shape)
{
    if (!expect_arg_count(tc, call, name, name_len, 3, 3)) return;
    WGSLTypeInfo *shape = resolved_shape ? resolved_shape : builtin_arg_type(tc, call, 0);
    WGSLTypeInfo *factor = builtin_arg_type(tc, call, 2);
    WGSLTypeInfo *elem = shape_elem(shape);
    if (!elem || !wgsl_type_is_float(elem)) return;

    WGSLTypeInfo *same_shape = shape;
    WGSLTypeInfo *scalar_shape = elem;
    if (type_converts_to(factor, same_shape)) return;
    if (shape && shape->kind == WGSL_TYPE_VEC && type_converts_to(factor, scalar_shape)) {
        return;
    }

    char got[96], want[96];
    wgsl_tc_error(tc, call,
        "'%.*s' mix factor has type %s; expected %s%s",
        (int)name_len, name,
        type_label(factor, got, sizeof got),
        type_label(same_shape, want, sizeof want),
        shape && shape->kind == WGSL_TYPE_VEC
            ? " or the matching floating-point scalar" : "");
}

static uint32_t const_value_width_tc(const WGSLValue *v) {
    return v && v->kind == WGSL_VAL_VEC ? v->u.agg.count : 1;
}

static const WGSLValue *const_value_component_tc(const WGSLValue *v, uint32_t i) {
    if (!v) return NULL;
    if (v->kind == WGSL_VAL_VEC) {
        return i < v->u.agg.count ? &v->u.agg.elems[i] : NULL;
    }
    return i == 0 ? v : NULL;
}

static int const_value_gt_tc(
    WGSLTypeChecker *tc, const WGSLValue *a, const WGSLValue *b, int *out)
{
    if (!a || !b || !out) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        if (a->type == tc->types->t_u32 || b->type == tc->types->t_u32) {
            *out = (uint32_t)a->u.i > (uint32_t)b->u.i;
        } else {
            *out = a->u.i > b->u.i;
        }
        return 1;
    }
    if (a->kind == WGSL_VAL_FLOAT && b->kind == WGSL_VAL_FLOAT) {
        *out = a->u.f > b->u.f;
        return 1;
    }
    return 0;
}

static int const_value_eq_tc(const WGSLValue *a, const WGSLValue *b, int *out)
{
    if (!a || !b || !out) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        *out = a->u.i == b->u.i;
        return 1;
    }
    if (a->kind == WGSL_VAL_FLOAT && b->kind == WGSL_VAL_FLOAT) {
        *out = a->u.f == b->u.f;
        return 1;
    }
    return 0;
}

static void check_clamp_low_high_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len)
{
    if (!builtin_name_eq(name, name_len, "clamp") || builtin_arg_count(call) != 3) {
        return;
    }
    WGSLValue lo = {0};
    WGSLValue hi = {0};
    if (try_consteval_expr_soft(tc, call->children[2], &lo) != 1 ||
        try_consteval_expr_soft(tc, call->children[3], &hi) != 1)
    {
        return;
    }
    uint32_t width = const_value_width_tc(&lo);
    if (const_value_width_tc(&hi) != width) return;
    for (uint32_t i = 0; i < width; i++) {
        int gt = 0;
        if (const_value_gt_tc(
                tc, const_value_component_tc(&lo, i),
                const_value_component_tc(&hi, i), &gt) && gt)
        {
            wgsl_tc_error(tc, call,
                "builtin 'clamp' domain error: low must be <= high");
            return;
        }
    }
}

static void check_smoothstep_edges_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len)
{
    if (!builtin_name_eq(name, name_len, "smoothstep") ||
        builtin_arg_count(call) != 3)
    {
        return;
    }
    WGSLValue edge0 = {0};
    WGSLValue edge1 = {0};
    if (try_consteval_expr_soft(tc, call->children[1], &edge0) != 1 ||
        try_consteval_expr_soft(tc, call->children[2], &edge1) != 1)
    {
        return;
    }
    uint32_t width = const_value_width_tc(&edge0);
    if (const_value_width_tc(&edge1) != width) return;
    for (uint32_t i = 0; i < width; i++) {
        int eq = 0;
        if (const_value_eq_tc(
                const_value_component_tc(&edge0, i),
                const_value_component_tc(&edge1, i), &eq) && eq)
        {
            wgsl_tc_error(tc, call,
                "builtin 'smoothstep' domain error: edge0 and edge1 must differ");
            return;
        }
    }
}

static void check_texture_offset_arg(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim, int arg_index)
{
    if (arg_index < 0 || (uint32_t)(1 + arg_index) >= call->child_count) return;
    WGSLNode *arg = call->children[1 + arg_index];
    if (!texture_allows_offset(dim)) {
        wgsl_tc_error(tc, arg,
            "'%.*s' does not accept an offset for this texture shape",
            (int)name_len, name);
        return;
    }

    int width = texture_offset_width(dim);
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    expect_i32_shape(tc, arg, name, name_len, t, width, "offset");
    if (!is_const_expr(arg)) {
        wgsl_tc_error(tc, arg,
            "'%.*s' offset must be a const-expression",
            (int)name_len, name);
        return;
    }

    int64_t vals[4] = {0};
    if (collect_const_int_components(tc, arg, width, vals)) {
        for (int i = 0; i < width; i++) {
            if (vals[i] < -8 || vals[i] > 7) {
                wgsl_tc_error(tc, arg,
                    "'%.*s' offset component %d value %lld is outside [-8, 7]",
                    (int)name_len, name, i, (long long)vals[i]);
                return;
            }
        }
    }
}

static void check_texture_sample_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim)
{
    int compare =
        builtin_name_eq(name, name_len, "textureSampleCompare") ||
        builtin_name_eq(name, name_len, "textureSampleCompareLevel");
    int base_clamp = builtin_name_eq(name, name_len, "textureSampleBaseClampToEdge");
    int sample_bias = builtin_name_eq(name, name_len, "textureSampleBias");
    int sample_grad = builtin_name_eq(name, name_len, "textureSampleGrad");
    int sample_level = builtin_name_eq(name, name_len, "textureSampleLevel");
    int has_array = texture_is_arrayed(dim);

    if (base_clamp) {
        if (!expect_arg_count(tc, call, name, name_len, 3, 3)) return;
        if (!(dim == WGSL_TEX_DIM_2D || dim == WGSL_TEX_DIM_EXTERNAL)) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires texture_2d<f32> or texture_external; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        if (dim == WGSL_TEX_DIM_2D && tex->ref != tc->types->t_f32) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires texture_2d<f32> or texture_external; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else if (compare) {
        int min_args = has_array ? 5 : 4;
        int max_args = min_args + 1;
        if (!expect_arg_count(tc, call, name, name_len, min_args, max_args)) return;
        if (!texture_is_depth(dim) || texture_is_multisampled(dim)) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-multisampled depth texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else {
        int min_args = has_array ? 4 : 3;
        int max_args = min_args + 1;
        if (sample_grad) {
            min_args += 2;
            max_args += 2;
        } else if (sample_bias || sample_level)
        {
            min_args += 1;
            max_args += 1;
        }
        if (!expect_arg_count(tc, call, name, name_len, min_args, max_args)) return;
        if (!(texture_is_sampled(dim) || texture_is_depth(dim)) ||
            texture_is_multisampled(dim) ||
            dim == WGSL_TEX_DIM_EXTERNAL)
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a sampled or depth texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        if ((sample_bias || sample_grad) &&
            (!texture_is_sampled(dim) || texture_is_depth(dim) ||
             dim == WGSL_TEX_DIM_1D || texture_is_multisampled(dim)))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-depth sampled 2d/2d-array/3d/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    }

    WGSLTypeInfo *sampler = builtin_arg_type(tc, call, 1);
    expect_sampler_arg(tc, call->children[2], name, name_len, sampler, compare);
    expect_float_shape(tc, call->children[3], name, name_len,
                       builtin_arg_type(tc, call, 2),
                       texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[4], name, name_len,
                             builtin_arg_type(tc, call, 3), 1,
                             "array index");
    }
    int next_arg = has_array ? 4 : 3;
    if (compare) {
        expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg), 1,
                           "depth reference");
        if (builtin_arg_count(call) > next_arg + 1) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 1);
        }
    } else if (sample_bias || sample_level) {
        if (sample_level && texture_is_depth(dim)) {
            expect_integer_shape(tc, call->children[1 + next_arg], name, name_len,
                                 builtin_arg_type(tc, call, next_arg), 1,
                                 "level");
        } else {
            expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                               builtin_arg_type(tc, call, next_arg), 1,
                               sample_bias ? "bias" : "level");
        }
        if (builtin_arg_count(call) > next_arg + 1) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 1);
        }
    } else if (sample_grad) {
        expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg),
                           texture_coord_width(dim), "ddx");
        expect_float_shape(tc, call->children[2 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg + 1),
                           texture_coord_width(dim), "ddy");
        if (builtin_arg_count(call) > next_arg + 2) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 2);
        }
    } else if (builtin_arg_count(call) > next_arg) {
        check_texture_offset_arg(tc, call, name, name_len, dim, next_arg);
    }
}

static void check_texture_load_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim)
{
    int has_array = texture_is_arrayed(dim);
    int base = has_array ? 3 : 2; /* texture + coords [+ array_index] */
    int want = base;
    if (texture_is_multisampled(dim)) want += 1;       /* sample_index */
    else if (dim != WGSL_TEX_DIM_EXTERNAL &&
             !texture_is_storage(dim)) want += 1;      /* mip level */
    if (!expect_arg_count(tc, call, name, name_len, want, want)) return;

    expect_integer_shape(tc, call->children[2], name, name_len,
                         builtin_arg_type(tc, call, 1),
                         texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[3], name, name_len,
                             builtin_arg_type(tc, call, 2), 1,
                             "array index");
    }
}

static void check_texture_store_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim)
{
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 4 : 3;
    if (!expect_arg_count(tc, call, name, name_len, want, want)) return;
    if (!texture_is_storage(dim)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a storage texture; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
        return;
    }
    if ((WGSLAccessMode)tex->rows == WGSL_ACCESS_READ) {
        wgsl_tc_error(tc, call,
            "'%.*s' requires a write or read_write storage texture",
            (int)name_len, name);
    }
    expect_integer_shape(tc, call->children[2], name, name_len,
                         builtin_arg_type(tc, call, 1),
                         texture_coord_width(dim), "coordinates");
    int value_arg = has_array ? 3 : 2;
    if (has_array) {
        expect_integer_shape(tc, call->children[3], name, name_len,
                             builtin_arg_type(tc, call, 2), 1,
                             "array index");
    }
    WGSLTypeInfo *want_value = wgsl_type_vec(
        tc->types, 4, storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len));
    expect_value_convertible(tc, call->children[1 + value_arg], name, name_len,
                             builtin_arg_type(tc, call, value_arg),
                             want_value, "value argument");
}

static void check_texture_query_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim)
{
    if (builtin_name_eq(name, name_len, "textureNumLayers")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (!texture_is_arrayed(dim)) {
            wgsl_tc_error(tc, call,
                "'%.*s' requires an array texture", (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureNumLevels")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (texture_is_storage(dim) || texture_is_multisampled(dim) ||
            dim == WGSL_TEX_DIM_EXTERNAL)
        {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a sampled or depth texture with mip levels",
                (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureNumSamples")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (!texture_is_multisampled(dim)) {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a multisampled texture",
                (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureDimensions")) {
        int want_min = 1;
        int want_max = (texture_is_storage(dim) ||
                        texture_is_multisampled(dim) ||
                        dim == WGSL_TEX_DIM_EXTERNAL) ? 1 : 2;
        expect_arg_count(tc, call, name, name_len, want_min, want_max);
    }
}

static void check_texture_gather_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim, int component_form)
{
    int compare = builtin_name_eq(name, name_len, "textureGatherCompare");
    int has_array = texture_is_arrayed(dim);
    int tex_arg = component_form ? 1 : 0;
    int sampler_arg = tex_arg + 1;
    int coords_arg = tex_arg + 2;
    int array_arg = coords_arg + 1;

    if (compare) {
        int depth_ref_arg = has_array ? 4 : 3;
        int min_args = depth_ref_arg + 1;
        if (!expect_arg_count(tc, call, name, name_len, min_args, min_args + 1)) {
            return;
        }
        if (!texture_is_depth(dim) || texture_is_multisampled(dim) ||
            !texture_is_gather_dim(dim))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-multisampled depth 2d/2d-array/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        expect_sampler_arg(tc, call->children[1 + sampler_arg], name, name_len,
                           builtin_arg_type(tc, call, sampler_arg), 1);
        expect_float_shape(tc, call->children[1 + coords_arg], name, name_len,
                           builtin_arg_type(tc, call, coords_arg),
                           texture_coord_width(dim), "coordinates");
        if (has_array) {
            expect_integer_shape(tc, call->children[1 + array_arg], name, name_len,
                                 builtin_arg_type(tc, call, array_arg), 1,
                                 "array index");
        }
        expect_float_shape(tc, call->children[1 + depth_ref_arg], name, name_len,
                           builtin_arg_type(tc, call, depth_ref_arg), 1,
                           "depth reference");
        if (builtin_arg_count(call) > min_args) {
            check_texture_offset_arg(tc, call, name, name_len, dim, min_args);
        }
        return;
    }

    int min_args = (component_form ? 4 : 3) + (has_array ? 1 : 0);
    if (!expect_arg_count(tc, call, name, name_len, min_args, min_args + 1)) {
        return;
    }

    if (component_form) {
        WGSLTypeInfo *ct = builtin_arg_type(tc, call, 0);
        expect_integer_shape(tc, call->children[1], name, name_len,
                             ct, 1, "component");
        check_const_int_range(tc, call->children[1], name, name_len,
                              0, 3, "component");
        if (texture_is_depth(dim) || !texture_is_sampled(dim) ||
            texture_is_multisampled(dim) || dim == WGSL_TEX_DIM_EXTERNAL ||
            !texture_is_gather_dim(dim))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' component form requires a non-depth sampled 2d/2d-array/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else if (!texture_is_depth(dim) || texture_is_multisampled(dim) ||
               !texture_is_gather_dim(dim))
    {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' without a component argument requires a non-multisampled depth texture; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
    }

    expect_sampler_arg(tc, call->children[1 + sampler_arg], name, name_len,
                       builtin_arg_type(tc, call, sampler_arg), 0);
    expect_float_shape(tc, call->children[1 + coords_arg], name, name_len,
                       builtin_arg_type(tc, call, coords_arg),
                       texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[1 + array_arg], name, name_len,
                             builtin_arg_type(tc, call, array_arg), 1,
                             "array index");
    }
    if (builtin_arg_count(call) > min_args) {
        check_texture_offset_arg(tc, call, name, name_len, dim, min_args);
    }
}

static void check_texture_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *tex,
    int gather_component_form)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a texture argument; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
        return;
    }
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;

    if (builtin_name_eq(name, name_len, "textureLoad")) {
        check_texture_load_call(tc, call, name, name_len, dim);
    } else if (builtin_name_eq(name, name_len, "textureStore")) {
        check_texture_store_call(tc, call, name, name_len, tex, dim);
    } else if (builtin_name_eq(name, name_len, "textureDimensions") ||
               builtin_name_eq(name, name_len, "textureNumLayers") ||
               builtin_name_eq(name, name_len, "textureNumLevels") ||
               builtin_name_eq(name, name_len, "textureNumSamples"))
    {
        check_texture_query_call(tc, call, name, name_len, dim);
    } else if (builtin_name_eq(name, name_len, "textureGather") ||
               builtin_name_eq(name, name_len, "textureGatherCompare"))
    {
        check_texture_gather_call(
            tc, call, name, name_len, tex, dim, gather_component_form);
    } else {
        check_texture_sample_call(tc, call, name, name_len, tex, dim);
    }
}

static int texture_silent_float_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *want = width == 1
        ? tc->types->t_f32
        : wgsl_type_vec(tc->types, (uint8_t)width, tc->types->t_f32);
    return type_converts_to(t, want);
}

static int texture_silent_integer_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *elem = shape_elem(t);
    return elem && wgsl_type_is_integer(elem) && shape_width(t) == width;
}

static int texture_silent_i32_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *elem = shape_elem(t);
    return elem &&
           (elem == tc->types->t_abstract_int || elem == tc->types->t_i32) &&
           shape_width(t) == width;
}

static int texture_silent_sampler(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int comparison)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    return t && t->kind == WGSL_TYPE_SAMPLER &&
           (((int)t->width != 0) == comparison);
}

static int texture_silent_storage_value(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, WGSLTypeInfo *tex)
{
    if (!tex || !texture_is_storage((WGSLTextureDim)tex->width)) return 0;
    WGSLTypeInfo *elem = storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len);
    WGSLTypeInfo *want = wgsl_type_vec(tc->types, 4, elem);
    return type_converts_to(builtin_arg_type(tc, call, arg_index), want);
}

static int texture_row_elem_matches(
    WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return 0;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    switch ((WGSLTextureOverloadElem)row->elem) {
    case BTEXEL_ANY:
        return 1;
    case BTEXEL_F32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_f32;
    case BTEXEL_I32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_i32;
    case BTEXEL_U32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_u32;
    case BTEXEL_DEPTH:
        return texture_is_depth(dim);
    case BTEXEL_EXTERNAL:
        return dim == WGSL_TEX_DIM_EXTERNAL;
    case BTEXEL_STORAGE_ANY:
        return texture_is_storage(dim);
    case BTEXEL_STORAGE_F32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_f32;
    case BTEXEL_STORAGE_I32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_i32;
    case BTEXEL_STORAGE_U32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_u32;
    }
    return 0;
}

static int texture_row_texture_matches(
    WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return 0;
    if (tex->width != row->dim) return 0;
    if (row->access != WGSL_ACCESS_NONE && tex->rows != row->access) return 0;
    return texture_row_elem_matches(tc, row, tex);
}

static int texture_match_array_index_if_needed(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index)
{
    return !texture_is_arrayed(dim) ||
           texture_silent_integer_shape(tc, call, arg_index, 1);
}

static int texture_match_offset(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index)
{
    return texture_allows_offset(dim) &&
           texture_silent_i32_shape(tc, call, arg_index, texture_offset_width(dim));
}

static int texture_match_sample_like(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row,
    WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int base_args = has_array ? 4 : 3;
    int next = has_array ? 4 : 3;

    if (!texture_silent_sampler(tc, call, 1,
            row->kind == BTOV_SAMPLE_COMPARE ||
            row->kind == BTOV_SAMPLE_COMPARE_OFFSET ||
            row->kind == BTOV_SAMPLE_COMPARE_LEVEL ||
            row->kind == BTOV_SAMPLE_COMPARE_LEVEL_OFFSET))
        return 0;
    if (!texture_silent_float_shape(tc, call, 2, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 3)) return 0;

    switch ((WGSLTextureOverloadKind)row->kind) {
    case BTOV_SAMPLE:
        return builtin_arg_count_is(call, base_args);
    case BTOV_SAMPLE_OFFSET:
        return builtin_arg_count_is(call, base_args + 1) &&
               texture_match_offset(tc, call, dim, next);
    case BTOV_SAMPLE_LEVEL:
    case BTOV_SAMPLE_BIAS:
        return builtin_arg_count_is(call, base_args + 1) &&
               (row->kind == BTOV_SAMPLE_LEVEL && texture_is_depth(dim)
                    ? texture_silent_integer_shape(tc, call, next, 1)
                    : texture_silent_float_shape(tc, call, next, 1));
    case BTOV_SAMPLE_LEVEL_OFFSET:
    case BTOV_SAMPLE_BIAS_OFFSET:
        return builtin_arg_count_is(call, base_args + 2) &&
               (row->kind == BTOV_SAMPLE_LEVEL_OFFSET && texture_is_depth(dim)
                    ? texture_silent_integer_shape(tc, call, next, 1)
                    : texture_silent_float_shape(tc, call, next, 1)) &&
               texture_match_offset(tc, call, dim, next + 1);
    case BTOV_SAMPLE_GRAD:
        return builtin_arg_count_is(call, base_args + 2) &&
               texture_silent_float_shape(tc, call, next, texture_coord_width(dim)) &&
               texture_silent_float_shape(tc, call, next + 1, texture_coord_width(dim));
    case BTOV_SAMPLE_GRAD_OFFSET:
        return builtin_arg_count_is(call, base_args + 3) &&
               texture_silent_float_shape(tc, call, next, texture_coord_width(dim)) &&
               texture_silent_float_shape(tc, call, next + 1, texture_coord_width(dim)) &&
               texture_match_offset(tc, call, dim, next + 2);
    case BTOV_SAMPLE_COMPARE:
    case BTOV_SAMPLE_COMPARE_LEVEL:
        return builtin_arg_count_is(call, base_args + 1) &&
               texture_silent_float_shape(tc, call, next, 1);
    case BTOV_SAMPLE_COMPARE_OFFSET:
    case BTOV_SAMPLE_COMPARE_LEVEL_OFFSET:
        return builtin_arg_count_is(call, base_args + 2) &&
               texture_silent_float_shape(tc, call, next, 1) &&
               texture_match_offset(tc, call, dim, next + 1);
    default:
        return 0;
    }
}

static int texture_match_load(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 3 : 2;
    int extra = -1;
    if (texture_is_multisampled(dim)) {
        extra = want;
        want += 1;
    } else if (dim != WGSL_TEX_DIM_EXTERNAL && !texture_is_storage(dim)) {
        extra = want;
        want += 1;
    }
    if (!builtin_arg_count_is(call, want)) return 0;
    if (!texture_silent_integer_shape(tc, call, 1, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 2)) return 0;
    return extra < 0 || texture_silent_integer_shape(tc, call, extra, 1);
}

static int texture_match_store(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 4 : 3;
    int value_arg = has_array ? 3 : 2;
    if (!builtin_arg_count_is(call, want)) return 0;
    if (!texture_silent_integer_shape(tc, call, 1, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 2)) return 0;
    return texture_silent_storage_value(tc, call, value_arg, tex);
}

static int texture_match_gather(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row)
{
    WGSLTextureOverloadKind kind = (WGSLTextureOverloadKind)row->kind;
    int component = kind == BTOV_GATHER_COMPONENT ||
                    kind == BTOV_GATHER_COMPONENT_OFFSET;
    int tex_arg = component ? 1 : 0;
    WGSLTypeInfo *tex = builtin_arg_type(tc, call, tex_arg);
    if (!texture_row_texture_matches(tc, row, tex)) return 0;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int sampler_arg = component ? 2 : 1;
    int coords_arg = component ? 3 : 2;
    int array_arg = coords_arg + 1;
    int depth_ref_arg = has_array ? 4 : 3;
    int base_args =
        kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET
            ? depth_ref_arg + 1
            : (component ? 4 : 3) + (has_array ? 1 : 0);

    if (!builtin_arg_count_is(call,
            base_args + ((kind == BTOV_GATHER_COMPONENT_OFFSET ||
                          kind == BTOV_GATHER_DEPTH_OFFSET ||
                          kind == BTOV_GATHER_COMPARE_OFFSET) ? 1 : 0)))
        return 0;
    if (component && !texture_silent_integer_shape(tc, call, 0, 1)) return 0;
    if (!texture_silent_sampler(
            tc, call, sampler_arg,
            kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET))
        return 0;
    if (!texture_silent_float_shape(tc, call, coords_arg, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, array_arg)) return 0;
    if ((kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET) &&
        !texture_silent_float_shape(tc, call, depth_ref_arg, 1))
    {
        return 0;
    }
    if (kind == BTOV_GATHER_COMPONENT_OFFSET ||
        kind == BTOV_GATHER_DEPTH_OFFSET ||
        kind == BTOV_GATHER_COMPARE_OFFSET)
    {
        return texture_match_offset(tc, call, dim, base_args);
    }
    return 1;
}

static int match_texture_overload_row(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row,
    WGSLTypeInfo **out_type)
{
    WGSLTextureOverloadKind kind = (WGSLTextureOverloadKind)row->kind;
    int tex_arg =
        (kind == BTOV_GATHER_COMPONENT || kind == BTOV_GATHER_COMPONENT_OFFSET)
            ? 1 : 0;
    WGSLTypeInfo *tex = builtin_arg_type(tc, call, tex_arg);
    if (kind != BTOV_GATHER_COMPONENT &&
        kind != BTOV_GATHER_COMPONENT_OFFSET &&
        kind != BTOV_GATHER_DEPTH &&
        kind != BTOV_GATHER_DEPTH_OFFSET &&
        kind != BTOV_GATHER_COMPARE &&
        kind != BTOV_GATHER_COMPARE_OFFSET &&
        !texture_row_texture_matches(tc, row, tex))
    {
        return 0;
    }

    WGSLTextureDim dim = tex && tex->kind == WGSL_TYPE_TEXTURE
        ? (WGSLTextureDim)tex->width : WGSL_TEX_DIM_1D;
    int ok = 0;
    switch (kind) {
    case BTOV_SAMPLE:
    case BTOV_SAMPLE_OFFSET:
    case BTOV_SAMPLE_LEVEL:
    case BTOV_SAMPLE_LEVEL_OFFSET:
    case BTOV_SAMPLE_BIAS:
    case BTOV_SAMPLE_BIAS_OFFSET:
    case BTOV_SAMPLE_GRAD:
    case BTOV_SAMPLE_GRAD_OFFSET:
    case BTOV_SAMPLE_COMPARE:
    case BTOV_SAMPLE_COMPARE_OFFSET:
    case BTOV_SAMPLE_COMPARE_LEVEL:
    case BTOV_SAMPLE_COMPARE_LEVEL_OFFSET:
        ok = texture_match_sample_like(tc, call, row, tex);
        break;
    case BTOV_SAMPLE_BASE_CLAMP:
        ok = builtin_arg_count_is(call, 3) &&
             texture_silent_sampler(tc, call, 1, 0) &&
             texture_silent_float_shape(tc, call, 2, texture_coord_width(dim));
        break;
    case BTOV_LOAD:
        ok = texture_match_load(tc, call, tex);
        break;
    case BTOV_STORE:
        ok = texture_match_store(tc, call, tex);
        break;
    case BTOV_GATHER_COMPONENT:
    case BTOV_GATHER_COMPONENT_OFFSET:
    case BTOV_GATHER_DEPTH:
    case BTOV_GATHER_DEPTH_OFFSET:
    case BTOV_GATHER_COMPARE:
    case BTOV_GATHER_COMPARE_OFFSET:
        ok = texture_match_gather(tc, call, row);
        break;
    case BTOV_DIMENSIONS:
        ok = builtin_arg_count_is(call, 1);
        break;
    case BTOV_DIMENSIONS_LEVEL:
        ok = builtin_arg_count_is(call, 2) &&
             texture_silent_integer_shape(tc, call, 1, 1);
        break;
    case BTOV_NUM_LAYERS:
    case BTOV_NUM_LEVELS:
    case BTOV_NUM_SAMPLES:
        ok = builtin_arg_count_is(call, 1);
        break;
    }
    if (!ok) return 0;
    *out_type = texture_builtin_result(tc, row->name, (uint32_t)strlen(row->name), tex);
    return *out_type != NULL;
}

static int resolve_generated_texture_overload(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo **out_type)
{
    int saw_row = 0;
    for (size_t i = 0; i < sizeof kTextureOverloads / sizeof kTextureOverloads[0]; i++) {
        const WGSLTextureOverload *row = &kTextureOverloads[i];
        size_t l = strlen(row->name);
        if (l != name_len || memcmp(row->name, name, l) != 0) continue;
        saw_row = 1;
        WGSLTypeInfo *resolved = NULL;
        if (!match_texture_overload_row(tc, call, row, &resolved)) continue;
        *out_type = resolved;
        return 1;
    }
    return saw_row ? -1 : 0;
}

/* ── Exact §17 overload rows generated from def/wgsl.def ──────────── */

typedef enum {
    BSHAPE_SCALAR_OR_VEC,
    BSHAPE_VEC,
    BSHAPE_VEC3,
    BSHAPE_MAT,
    BSHAPE_MAT_SQUARE,
} WGSLBuiltinShapeReq;

static int resolve_overload_silent(
    WGSLTypeChecker *tc,
    const WGSLOLCandidate *cands, int cand_count,
    WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count,
    const char *what, const WGSLNode *at)
{
    size_t saved_count = tc->diag ? tc->diag->count : 0;
    int saved_errors = tc->diag ? tc->diag->error_count : 0;
    int saved_had_error = tc->had_error;
    int picked = resolve_overload(
        tc, cands, cand_count, arg_types, arg_is_const, arg_count, what, at);
    if (picked < 0 && tc->diag) {
        tc->diag->count = saved_count;
        tc->diag->error_count = saved_errors;
        tc->had_error = saved_had_error;
    }
    return picked;
}

static int shape_satisfies_req(const WGSLTypeInfo *shape, WGSLBuiltinShapeReq req) {
    if (!shape) return 0;
    switch (req) {
    case BSHAPE_SCALAR_OR_VEC:
        return wgsl_type_is_scalar(shape) || shape->kind == WGSL_TYPE_VEC;
    case BSHAPE_VEC:
        return shape->kind == WGSL_TYPE_VEC;
    case BSHAPE_VEC3:
        return shape->kind == WGSL_TYPE_VEC && shape->width == 3;
    case BSHAPE_MAT:
        return shape->kind == WGSL_TYPE_MAT;
    case BSHAPE_MAT_SQUARE:
        return shape->kind == WGSL_TYPE_MAT && shape->width == shape->rows;
    }
    return 0;
}

static WGSLTypeInfo *resolve_t_family_prefix(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name,
    WGSLParamFamily fam, int arity, int total_args, WGSLBuiltinShapeReq req)
{
    if (!builtin_arg_count_is(call, total_args) || arity <= 0 ||
        arity > WGSL_OL_MAX_PARAMS)
    {
        return NULL;
    }

    WGSLTypeInfo *args[WGSL_OL_MAX_PARAMS] = {0};
    int arg_const[WGSL_OL_MAX_PARAMS] = {0};
    for (int i = 0; i < arity; i++) {
        WGSLNode *arg = call->children[1 + i];
        WGSLTypeInfo *t = builtin_arg_type(tc, call, i);
        if (!t) return NULL;
        args[i] = t;
        arg_const[i] = is_const_expr(arg);
    }

    const WGSLTypeInfo *shape = pick_shape_arg(args, arity);
    if (!shape_satisfies_req(shape, req)) return NULL;

    WGSLOLCandidate cands[WGSL_OL_MAX_CANDS];
    int nc = build_t_candidates(tc, fam, args, arity, cands);
    if (nc <= 0) return NULL;
    int picked = resolve_overload_silent(
        tc, cands, nc, args, arg_const, arity, name, call);
    return picked >= 0 ? cands[picked].result : NULL;
}

static WGSLTypeInfo *resolve_t_family_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name,
    WGSLParamFamily fam, int arity, WGSLBuiltinShapeReq req)
{
    return resolve_t_family_prefix(tc, call, name, fam, arity, arity, req);
}

static int match_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    return type_converts_to(builtin_arg_type(tc, call, index), tc->types->t_u32);
}

static int match_i32_or_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    WGSLTypeInfo *t = builtin_arg_type(tc, call, index);
    return type_converts_to(t, tc->types->t_i32) ||
           type_converts_to(t, tc->types->t_u32);
}

static int subgroup_shuffle_second_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, 1);
    int type_ok = type_converts_to(t, tc->types->t_u32);
    if (!type_ok && strcmp(name, "subgroupShuffle") == 0) {
        type_ok = type_converts_to(t, tc->types->t_i32);
    }
    if (!type_ok) return 0;

    if (call->child_count < 3 || !is_const_expr(call->children[2])) return 1;
    int64_t value = 0;
    if (!eval_const_int_silent(tc, call->children[2], &value)) return 1;
    if (value < 0 || value >= 128) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be in [0, 128)",
            name);
        return 0;
    }
    return 1;
}

static int broadcast_second_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!match_i32_or_u32_arg(tc, call, 1)) return 0;

    if (call->child_count < 3 || !is_const_expr(call->children[2])) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be a const-expression",
            name);
        return 0;
    }

    int64_t value = 0;
    if (!eval_const_int_silent(tc, call->children[2], &value)) return 0;
    int64_t hi = strcmp(name, "quadBroadcast") == 0 ? 4 : 128;
    if (value < 0 || value >= hi) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be in [0, %lld)",
            name, (long long)hi);
        return 0;
    }
    return 1;
}

static int match_bool_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    return builtin_arg_type(tc, call, index) == tc->types->t_bool;
}

static int match_vec_arg(WGSLTypeChecker *tc, WGSLNode *call, int index,
                         uint8_t width, WGSLTypeInfo *elem)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, index);
    WGSLTypeInfo *want = wgsl_type_vec(tc->types, width, elem);
    return t && want && wgsl_type_conversion_rank(t, want) >= 0;
}

static int match_all_any(WGSLTypeChecker *tc, WGSLNode *call) {
    if (!builtin_arg_count_is(call, 1)) return 0;
    WGSLTypeInfo *t = builtin_arg_type(tc, call, 0);
    if (t == tc->types->t_bool) return 1;
    return t && t->kind == WGSL_TYPE_VEC &&
           (WGSLTypeInfo *)t->ref == tc->types->t_bool;
}

static WGSLTypeInfo *resolve_select_result(WGSLTypeChecker *tc, WGSLNode *call) {
    if (!builtin_arg_count_is(call, 3)) return NULL;

    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, "select", PF_NUMERIC_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
    if (!value) {
        WGSLTypeInfo *a = builtin_arg_type(tc, call, 0);
        WGSLTypeInfo *b = builtin_arg_type(tc, call, 1);
        int scalar_bool = a == tc->types->t_bool && b == tc->types->t_bool;
        int vec_bool = a && b && a->kind == WGSL_TYPE_VEC &&
                       b->kind == WGSL_TYPE_VEC &&
                       a->width == b->width &&
                       (WGSLTypeInfo *)a->ref == tc->types->t_bool &&
                       (WGSLTypeInfo *)b->ref == tc->types->t_bool;
        if (!scalar_bool && !vec_bool) return NULL;
        value = a;
    }

    WGSLTypeInfo *cond = builtin_arg_type(tc, call, 2);
    if (cond == tc->types->t_bool) return value;
    if (value && value->kind == WGSL_TYPE_VEC &&
        cond && cond->kind == WGSL_TYPE_VEC &&
        cond->width == value->width &&
        (WGSLTypeInfo *)cond->ref == tc->types->t_bool)
    {
        return value;
    }
    return NULL;
}

static int bitfield_const_range_ok(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, int offset_arg, int count_arg)
{
    if (!call || offset_arg < 0 || count_arg < 0) return 0;
    uint32_t off_child = (uint32_t)offset_arg + 1u;
    uint32_t cnt_child = (uint32_t)count_arg + 1u;
    if (off_child >= call->child_count || cnt_child >= call->child_count) {
        return 0;
    }
    WGSLNode *offset_node = call->children[off_child];
    WGSLNode *count_node = call->children[cnt_child];
    if (!is_const_expr(offset_node) || !is_const_expr(count_node)) return 1;

    int64_t offset = 0;
    int64_t count = 0;
    if (!eval_const_int_silent(tc, offset_node, &offset) ||
        !eval_const_int_silent(tc, count_node, &count))
    {
        return 1;
    }
    if (offset < 0 || count < 0 || offset + count > 32) {
        wgsl_tc_error(tc, call,
            "builtin '%s' domain error: offset + count must be <= 32",
            name);
        return 0;
    }
    return 1;
}

static WGSLTypeInfo *resolve_extract_bits(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, name, PF_INTEGER_T, 1, 3, BSHAPE_SCALAR_OR_VEC);
    if (!value || !match_u32_arg(tc, call, 1) || !match_u32_arg(tc, call, 2)) {
        return NULL;
    }
    if (!bitfield_const_range_ok(tc, call, name, 1, 2)) return NULL;
    return value;
}

static WGSLTypeInfo *resolve_insert_bits(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 4)) return NULL;
    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, name, PF_INTEGER_T, 2, 4, BSHAPE_SCALAR_OR_VEC);
    if (!value || !match_u32_arg(tc, call, 2) || !match_u32_arg(tc, call, 3)) {
        return NULL;
    }
    if (!bitfield_const_range_ok(tc, call, name, 2, 3)) return NULL;
    return value;
}

static WGSLTypeInfo *resolve_ldexp_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 2)) return NULL;
    WGSLTypeInfo *x = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 1, 2, BSHAPE_SCALAR_OR_VEC);
    if (!x) return NULL;
    WGSLTypeInfo *e = builtin_arg_type(tc, call, 1);
    WGSLTypeInfo *ee = shape_elem(e);
    int xw = shape_width(x);
    int ew = shape_width(e);
    if (!ee || ew != xw ||
        !(ee == tc->types->t_abstract_int || ee == tc->types->t_i32))
    {
        return NULL;
    }
    return x;
}

static WGSLTypeInfo *resolve_mix_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *shape = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
    if (!shape) return NULL;
    WGSLTypeInfo *factor = builtin_arg_type(tc, call, 2);
    WGSLTypeInfo *elem = shape_elem(shape);
    if (type_converts_to(factor, shape)) return shape;
    if (shape->kind == WGSL_TYPE_VEC && type_converts_to(factor, elem)) {
        return shape;
    }
    return NULL;
}

static WGSLTypeInfo *resolve_refract_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *shape = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 2, 3, BSHAPE_VEC);
    if (!shape) return NULL;
    WGSLTypeInfo *elem = shape_elem(shape);
    if (!type_converts_to(builtin_arg_type(tc, call, 2), elem)) return NULL;
    return shape;
}

static WGSLTypeInfo *resolve_transpose_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    WGSLTypeInfo *mat = resolve_t_family_exact(
        tc, call, name, PF_FLOAT_T, 1, BSHAPE_MAT);
    if (!mat || mat->kind != WGSL_TYPE_MAT) return NULL;
    return wgsl_type_mat(
        tc->types, mat->rows, mat->width, (WGSLTypeInfo *)mat->ref);
}

static WGSLTypeInfo *resolve_atomic_exact(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type,
    WGSLBuiltinOverloadPattern pattern)
{
    WGSLTypeInfo *T = atomic_pointee_T(arg0_type);
    if (!T) return NULL;
    int argc = builtin_arg_count(call);
    switch (pattern) {
    case BOP_ATOMIC_LOAD:
        return argc == 1 ? T : NULL;
    case BOP_ATOMIC_STORE:
        return argc == 2 && type_converts_to(builtin_arg_type(tc, call, 1), T)
            ? T : NULL;
    case BOP_ATOMIC_RMW:
        return argc == 2 && type_converts_to(builtin_arg_type(tc, call, 1), T)
            ? T : NULL;
    case BOP_ATOMIC_CX:
        return argc == 3 &&
               type_converts_to(builtin_arg_type(tc, call, 1), T) &&
               type_converts_to(builtin_arg_type(tc, call, 2), T)
            ? T : NULL;
    default:
        return NULL;
    }
}

static int array_length_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type)
{
    if (!builtin_arg_count_is(call, 1)) return 0;
    if (arg0_type && arg0_type->kind == WGSL_TYPE_PTR) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)arg0_type->ref;
        uint8_t as = (uint8_t)(arg0_type->flags & 0xFFu);
        return elem && elem->kind == WGSL_TYPE_ARRAY &&
               elem->array_len == 0 && as == WGSL_AS_STORAGE;
    }
    if (arg0_type && arg0_type->kind == WGSL_TYPE_ARRAY &&
        call->child_count >= 2 &&
        call->children[1] &&
        call->children[1]->kind == WGSL_NODE_EXPR_ADDR_OF &&
        call->children[1]->child_count >= 1)
    {
        WGSLAddressSpace as = WGSL_AS_NONE;
        WGSLAccessMode am = WGSL_ACCESS_NONE;
        wgsl_tc_get_ref_space_and_mode(tc, call->children[1]->children[0], &as, &am);
        (void)am;
        return as == WGSL_AS_STORAGE && arg0_type->array_len == 0;
    }
    return 0;
}

static void check_array_length_call(WGSLTypeChecker *tc, WGSLNode *call,
                                    WGSLTypeInfo *arg0_type)
{
    if (!array_length_arg_ok(tc, call, arg0_type)) {
        wgsl_tc_error(tc, call,
            "arrayLength requires a pointer to a runtime-sized "
            "array in the storage address space");
    }
}

static WGSLTypeInfo *resolve_workgroup_uniform_load(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type)
{
    if (!builtin_arg_count_is(call, 1) || !arg0_type) return NULL;
    if (arg0_type->kind == WGSL_TYPE_PTR) {
        uint8_t as = (uint8_t)(arg0_type->flags & 0xFFu);
        uint8_t am = (uint8_t)((arg0_type->flags >> 8) & 0xFFu);
        if (as == WGSL_AS_WORKGROUP && am == WGSL_ACCESS_READ_WRITE) {
            WGSLTypeInfo *store = (WGSLTypeInfo *)arg0_type->ref;
            if (store && store->kind == WGSL_TYPE_ATOMIC) {
                return (WGSLTypeInfo *)store->ref;
            }
            if (!wgsl_tc_is_constructible_deep(tc, store)) return NULL;
            return store;
        }
        return NULL;
    }
    if (call->child_count >= 2 &&
        call->children[1] &&
        call->children[1]->kind == WGSL_NODE_EXPR_ADDR_OF &&
        call->children[1]->child_count >= 1)
    {
        WGSLNode *base = call->children[1]->children[0];
        if (base && base->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *bs = wgsl_node_resolved_symbol(base);
            if (bs && bs->as == WGSL_AS_WORKGROUP &&
                bs->am == WGSL_ACCESS_READ_WRITE)
            {
                if (!wgsl_tc_is_constructible_deep(tc, arg0_type)) return NULL;
                return arg0_type;
            }
        }
    }
    return NULL;
}

static WGSLTypeInfo *result_from_builtin_return(
    WGSLTypeChecker *tc, WGSLBuiltinReturn ret,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo *resolved)
{
    WGSLTypeInfo *src = resolved ? resolved : (arg0_type ? arg0_type : fallback);
    switch (ret) {
    case BR_VOID:
        return tc->types->t_void;
    case BR_BOOL:
        return tc->types->t_bool;
    case BR_I32:
        return tc->types->t_i32;
    case BR_U32:
        return tc->types->t_u32;
    case BR_ARG0:
        return src;
    case BR_ARG0_ELEM: {
        WGSLTypeInfo *elem = element_type_of(src);
        return elem ? elem : src;
    }
    case BR_VEC4F:
        return wgsl_type_vec(tc->types, 4, tc->types->t_f32);
    case BR_VEC4U:
        return wgsl_type_vec(tc->types, 4, tc->types->t_u32);
    case BR_VEC4I:
        return wgsl_type_vec(tc->types, 4, tc->types->t_i32);
    case BR_VEC2F:
        return wgsl_type_vec(tc->types, 2, tc->types->t_f32);
    case BR_ATOMIC_CX_RESULT:
        return wgsl_type_atomic_cx_result(tc->types, src);
    case BR_FREXP_RESULT:
        return wgsl_type_frexp_result(tc->types, src);
    case BR_MODF_RESULT:
        return wgsl_type_modf_result(tc->types, src);
    case BR_TEMPLATE:
        return src;
    }
    return fallback;
}

static int builtin_requires_concrete_result(const char *name, uint32_t name_len) {
    return (name_len >= 8 && memcmp(name, "subgroup", 8) == 0) ||
           (name_len >= 4 && memcmp(name, "quad", 4) == 0);
}

static int match_builtin_overload_row(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinOverload *row,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo **out_resolved)
{
    WGSLTypeInfo *r = NULL;
    switch (row->pattern) {
    case BOP_VOID_0:
        r = builtin_arg_count_is(call, 0) ? tc->types->t_void : NULL;
        break;
    case BOP_ALL_ANY:
        r = match_all_any(tc, call) ? tc->types->t_bool : NULL;
        break;
    case BOP_SELECT:
        r = resolve_select_result(tc, call);
        break;
    case BOP_ARRAY_LENGTH:
        r = array_length_arg_ok(tc, call, arg0_type) ? tc->types->t_u32 : NULL;
        break;
    case BOP_WORKGROUP_UNIFORM_LOAD:
        r = resolve_workgroup_uniform_load(tc, call, arg0_type);
        break;
    case BOP_T_FLOAT_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_FLOAT_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_FLOAT_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 3, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 2, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 3, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_SIGN_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_SIGN_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_INTEGER_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_INTEGER_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_EXTRACT_BITS:
        r = resolve_extract_bits(tc, call, row->name);
        break;
    case BOP_INSERT_BITS:
        r = resolve_insert_bits(tc, call, row->name);
        break;
    case BOP_LDEXP:
        r = resolve_ldexp_exact(tc, call, row->name);
        break;
    case BOP_MIX:
        r = resolve_mix_exact(tc, call, row->name);
        break;
    case BOP_DOT:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 2, BSHAPE_VEC);
        break;
    case BOP_DOT4_U32:
        r = builtin_arg_count_is(call, 2) &&
            match_u32_arg(tc, call, 0) && match_u32_arg(tc, call, 1)
            ? tc->types->t_u32 : NULL;
        break;
    case BOP_LENGTH:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_DETERMINANT:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_MAT_SQUARE);
        break;
    case BOP_CROSS:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_VEC3);
        break;
    case BOP_TRANSPOSE:
        r = resolve_transpose_exact(tc, call, row->name);
        break;
    case BOP_VEC_FLOAT_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_VEC);
        break;
    case BOP_VEC_FLOAT_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_VEC);
        break;
    case BOP_VEC_FLOAT_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 3, BSHAPE_VEC);
        break;
    case BOP_REFRACT:
        r = resolve_refract_exact(tc, call, row->name);
        break;
    case BOP_T_F32_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_F32_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_ARG_U32_1:
        r = builtin_arg_count_is(call, 1) && match_u32_arg(tc, call, 0)
            ? tc->types->t_u32 : NULL;
        break;
    case BOP_ARG_VEC4F_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_f32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC4I_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_i32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC4U_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_u32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC2F_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 2, tc->types->t_f32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ATOMIC_LOAD:
    case BOP_ATOMIC_STORE:
    case BOP_ATOMIC_RMW:
    case BOP_ATOMIC_CX:
        r = resolve_atomic_exact(tc, call, arg0_type, row->pattern);
        break;
    case BOP_BOOL_1:
        r = builtin_arg_count_is(call, 1) && match_bool_arg(tc, call, 0)
            ? tc->types->t_bool : NULL;
        break;
    case BOP_T_NUMERIC_U32_2:
        if (!builtin_arg_count_is(call, 2)) break;
        if (strcmp(row->name, "subgroupBroadcast") == 0 ||
            strcmp(row->name, "quadBroadcast") == 0)
        {
            if (!broadcast_second_arg_ok(tc, call, row->name)) break;
        } else if (strncmp(row->name, "subgroupShuffle", 15) == 0) {
            if (!subgroup_shuffle_second_arg_ok(tc, call, row->name)) break;
        } else if (!match_i32_or_u32_arg(tc, call, 1)) {
            break;
        }
        {
            r = resolve_t_family_prefix(tc, call, row->name, PF_NUMERIC_T, 1,
                                        2, BSHAPE_SCALAR_OR_VEC);
        }
        break;
    }
    if (!r) return 0;
    *out_resolved = r;
    return 1;
}

static int resolve_generated_builtin_overload(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo **out_type)
{
    int saw_row = 0;
    for (size_t i = 0; i < sizeof kBuiltinOverloads / sizeof kBuiltinOverloads[0]; i++) {
        const WGSLBuiltinOverload *row = &kBuiltinOverloads[i];
        size_t l = strlen(row->name);
        if (l != name_len || memcmp(row->name, name, l) != 0) continue;
        saw_row = 1;
        WGSLTypeInfo *resolved = NULL;
        if (!match_builtin_overload_row(tc, call, row, arg0_type, &resolved)) {
            continue;
        }
        WGSLTypeInfo *result = result_from_builtin_return(
            tc, row->ret, arg0_type, fallback, resolved);
        if (builtin_requires_concrete_result(name, name_len)) {
            result = wgsl_type_concretize(tc->types, result);
        }
        *out_type = result;
        return 1;
    }
    if (!saw_row) return 0;
    if (builtin_name_eq(name, name_len, "mix")) {
        WGSLTypeInfo *shape = resolve_t_family_prefix(
            tc, call, "mix", PF_FLOAT_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
        check_mix_call(tc, call, name, name_len, shape ? shape : arg0_type);
    } else if (builtin_name_eq(name, name_len, "ldexp")) {
        WGSLTypeInfo *shape = resolve_t_family_prefix(
            tc, call, "ldexp", PF_FLOAT_T, 1, 2, BSHAPE_SCALAR_OR_VEC);
        check_ldexp_call(tc, call, name, name_len, shape ? shape : arg0_type);
    } else if (builtin_name_eq(name, name_len, "arrayLength")) {
        check_array_length_call(tc, call, arg0_type);
    }
    wgsl_tc_error(tc, call, "no matching overload for '%.*s'", (int)name_len, name);
    *out_type = fallback;
    return 1;
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
        fallback = wgsl_tc_store_type_of(tc, call->children[1]);
    }
    if (!fallback) fallback = tc->types->t_abstract_float;

    if (!e) return wgsl_tc_set_type(tc, call, fallback, 0);

    /* §4.1.1 / §17.12 / §17.13: subgroup and quad builtin functions
     * require `enable subgroups;`.  Gate them here so the diagnostic
     * fires at the call site (same pattern as @builtin gating in
     * validate.c). */
    if (sym->name_len >= 8 && memcmp(sym->name, "subgroup", 8) == 0) {
        if (!(tc->enabled_extensions & WGSL_EXT_SUBGROUPS)) {
            wgsl_tc_error(tc, call,
                "use of '%.*s' requires 'enable subgroups;'",
                (int)sym->name_len, sym->name);
        }
    }
    if (sym->name_len >= 4 && memcmp(sym->name, "quad", 4) == 0) {
        if (!(tc->enabled_extensions & WGSL_EXT_SUBGROUPS)) {
            wgsl_tc_error(tc, call,
                "use of '%.*s' requires 'enable subgroups;'",
                (int)sym->name_len, sym->name);
        }
    }

    WGSLTypeInfo *arg0_type = NULL;
    if (call->child_count >= 2) {
        WGSLNode *arg0 = call->children[1];
        arg0_type = wgsl_tc_store_type_of(tc, arg0);
    }

    if ((WGSLBuiltinSpecial)e->special == BS_TEXTURE) {
        WGSLTypeInfo *texture_result = NULL;
        int tex_match = resolve_generated_texture_overload(
            tc, call, sym->name, sym->name_len, &texture_result);
        int gather_component_form =
            builtin_name_eq(sym->name, sym->name_len, "textureGather") &&
            (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE);
        WGSLTypeInfo *tex_arg = texture_arg_type_for_call(
            tc, call, sym->name, sym->name_len, arg0_type);
        check_texture_call(
            tc, call, sym->name, sym->name_len, tex_arg,
            gather_component_form);
        if (tex_match < 0) {
            wgsl_tc_error(tc, call, "no matching overload for '%.*s'",
                          (int)sym->name_len, sym->name);
            WGSLTypeInfo *diagnostic_result = texture_builtin_result(
                tc, sym->name, sym->name_len, tex_arg);
            return wgsl_tc_set_type(
                tc, call, diagnostic_result ? diagnostic_result : fallback, 0);
        }
        return wgsl_tc_set_type(
            tc, call, texture_result ? texture_result : fallback, 0);
    }

    /* §17.9 / §17.10 — pack* / unpack* arg shape (vec4<f32> /
     * vec4<i32> / vec4<u32> / vec2<f32> / u32). */
    if ((sym->name_len >= 4 && memcmp(sym->name, "pack", 4) == 0) ||
        (sym->name_len >= 6 && memcmp(sym->name, "unpack", 6) == 0))
    {
        check_pack_unpack_arg(tc, call, sym->name, sym->name_len, arg0_type);
    }

    /* §17.8 — atomic* arg shapes: ptr<…, atomic<T>, …> + value args of T. */
    if (sym->name_len >= 6 && memcmp(sym->name, "atomic", 6) == 0) {
        check_atomic_call(tc, call, sym->name, sym->name_len, arg0_type);
    }

    WGSLTypeInfo *generated_result = NULL;
    if (resolve_generated_builtin_overload(
            tc, call, sym->name, sym->name_len, arg0_type, fallback,
            &generated_result))
    {
        check_clamp_low_high_call(tc, call, sym->name, sym->name_len);
        check_smoothstep_edges_call(tc, call, sym->name, sym->name_len);
        if (builtin_name_eq(sym->name, sym->name_len, "ldexp")) {
            check_ldexp_call(tc, call, sym->name, sym->name_len,
                             generated_result ? generated_result : arg0_type);
        }
        check_const_builtin_call(tc, call, e);
        return wgsl_tc_set_type(tc, call,
            generated_result ? generated_result : fallback, 0);
    }

    /* §6.1.3 — overload resolution.  When this entry advertises a param
     * family, build the candidate set parametrically and run the §6.1.3
     * algorithm.  The chosen candidate's `result` shape supplies the
     * builtin's static type (BR_ARG0 / BR_ARG0_ELEM read from it instead
     * of from the raw arg type, so e.g. `log2(32)` yields AbstractFloat
     * rather than AbstractInt). */
    WGSLTypeInfo *resolved_shape = NULL;
    if (e->fam != PF_NONE && e->arity > 0 && call->child_count >= 1 + e->arity) {
        WGSLTypeInfo *args[WGSL_OL_MAX_PARAMS] = {0};
        int arg_const[WGSL_OL_MAX_PARAMS] = {0};
        int arity = (int)e->arity;
        if (arity > WGSL_OL_MAX_PARAMS) arity = WGSL_OL_MAX_PARAMS;

        int ok = 1;
        for (int i = 0; i < arity; i++) {
            WGSLNode *arg = call->children[1 + i];
            WGSLTypeInfo *t;
            if (arg->kind == WGSL_NODE_EXPR_ADDR_OF && arg->child_count >= 1) {
                t = wgsl_tc_store_type_of(tc, arg->children[0]);
            } else {
                t = wgsl_tc_store_type_of(tc, arg);
            }
            if (!t) { ok = 0; break; }
            args[i] = t;
            arg_const[i] = is_const_expr(arg);
        }

        if (ok) {
            WGSLOLCandidate cands[WGSL_OL_MAX_CANDS];
            int nc = build_t_candidates(tc, e->fam, args, arity, cands);
            if (nc > 0) {
                int picked = resolve_overload(
                    tc, cands, nc, args, arg_const, arity, e->name, call);
                if (picked >= 0) {
                    resolved_shape = cands[picked].result;
                }
            }
        }
    }

    WGSLTypeInfo *special_result = NULL;
    switch ((WGSLBuiltinSpecial)e->special) {
    case BS_LDEXP:
        check_ldexp_call(tc, call, sym->name, sym->name_len,
                         resolved_shape ? resolved_shape : arg0_type);
        break;
    case BS_MIX:
        check_mix_call(tc, call, sym->name, sym->name_len,
                       resolved_shape ? resolved_shape : arg0_type);
        break;
    case BS_TEXTURE: {
        int gather_component_form =
            builtin_name_eq(sym->name, sym->name_len, "textureGather") &&
            (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE);
        WGSLTypeInfo *tex_arg = texture_arg_type_for_call(
            tc, call, sym->name, sym->name_len, arg0_type);
        check_texture_call(
            tc, call, sym->name, sym->name_len, tex_arg,
            gather_component_form);
        special_result = texture_builtin_result(
            tc, sym->name, sym->name_len, tex_arg);
        break;
    }
    case BS_NONE:
    default:
        break;
    }

    switch (e->ret) {
    case BR_VOID:
        return wgsl_tc_set_type(tc, call, tc->types->t_void, 0);
    case BR_BOOL:
        return wgsl_tc_set_type(tc, call, tc->types->t_bool, 0);
    case BR_I32:
        return wgsl_tc_set_type(tc, call, tc->types->t_i32, 0);
    case BR_U32:
        if (special_result) return wgsl_tc_set_type(tc, call, special_result, 0);
        /* §17.4 — `arrayLength(p)` requires p to be a pointer to a
         * runtime-sized array in the storage address space.  We've
         * already collected `arg0_type` above; check the shape here
         * so a `let arr: array<f32,3>; arrayLength(&arr);` gets a
         * crisp diagnostic rather than silently returning u32. */
        if (sym->name_len == 11 &&
            memcmp(sym->name, "arrayLength", 11) == 0)
        {
            if (!array_length_arg_ok(tc, call, arg0_type)) {
                wgsl_tc_error(tc, call,
                    "arrayLength requires a pointer to a runtime-sized "
                    "array in the storage address space");
            }
        }
        return wgsl_tc_set_type(tc, call, tc->types->t_u32, 0);
    case BR_ARG0: {
        WGSLTypeInfo *t = resolved_shape ? resolved_shape :
                          (arg0_type ? arg0_type : fallback);
        return wgsl_tc_set_type(tc, call, t, 0);
    }
    case BR_ARG0_ELEM: {
        WGSLTypeInfo *src = resolved_shape ? resolved_shape :
                            (arg0_type ? arg0_type : fallback);
        WGSLTypeInfo *elem = element_type_of(src);
        return wgsl_tc_set_type(tc, call, elem ? elem : src, 0);
    }
    case BR_VEC4F:
        if (special_result) return wgsl_tc_set_type(tc, call, special_result, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_f32), 0);
    case BR_VEC4U:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_u32), 0);
    case BR_VEC4I:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 4, tc->types->t_i32), 0);
    case BR_VEC2F:
        return wgsl_tc_set_type(tc, call,
            wgsl_type_vec(tc->types, 2, tc->types->t_f32), 0);
    case BR_TEMPLATE: {
        /* `bitcast<T>(x)` — pull T from the templated-callee form.
         * §17.2 restricts T to a concrete numeric scalar (i32 / u32 /
         * f32) or a concrete numeric vector; reject anything else. */
        if (callee && callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count >= 2)
        {
            WGSLTypeInfo *t = wgsl_tc_resolve_type_spec(tc, callee->children[1]);
            if (t) {
                const WGSLTypeInfo *scalar = (t->kind == WGSL_TYPE_VEC)
                    ? (const WGSLTypeInfo *)t->ref : t;
                int ok =
                    scalar == tc->types->t_i32 ||
                    scalar == tc->types->t_u32 ||
                    scalar == tc->types->t_f32 ||
                    scalar == tc->types->t_f16;
                if (!ok) {
                    wgsl_tc_error(tc, callee->children[1],
                        "bitcast<T> requires T to be a concrete numeric "
                        "scalar or vector (i32 / u32 / f32 / f16, "
                        "scalar or vecN); got %s",
                        wgsl_type_kind_name((WGSLTypeKind)t->kind));
                }
                int target_bits = bitcast_type_bits(t);
                int source_bits = bitcast_type_bits(arg0_type);
                if (target_bits == 0) {
                    char got[96];
                    wgsl_tc_error(tc, callee->children[1],
                        "bitcast<T> target type %s has no valid 32-bit "
                        "word representation",
                        type_label(t, got, sizeof got));
                }
                if (source_bits == 0) {
                    char got[96];
                    wgsl_tc_error(tc, call->child_count >= 2 ? call->children[1] : call,
                        "bitcast source type must be a concrete numeric "
                        "scalar or vector with a 32-bit word representation; got %s",
                        type_label(arg0_type, got, sizeof got));
                } else if (target_bits != 0 && target_bits != source_bits) {
                    char from[96], to[96];
                    wgsl_tc_error(tc, call,
                        "bitcast source type %s (%d bits) and target type "
                        "%s (%d bits) must have the same bit width",
                        type_label(arg0_type, from, sizeof from), source_bits,
                        type_label(t, to, sizeof to), target_bits);
                }
                return wgsl_tc_set_type(tc, call, t, 0);
            }
        }
        return wgsl_tc_set_type(tc, call, fallback, 0);
    }
    case BR_ATOMIC_CX_RESULT: {
        if (!arg0_type) return wgsl_tc_set_type(tc, call, fallback, 0);
        WGSLTypeInfo *elem = element_type_of(arg0_type);
        if (!elem) elem = arg0_type;
        return wgsl_tc_set_type(tc, call, wgsl_type_atomic_cx_result(tc->types, elem), 0);
    }
    case BR_FREXP_RESULT: {
        /* §17 — `frexp(T)` → `__frexp_result<T>` where T is the arg's
         * post-concretization type (scalar f / vecN<f>).  `resolved_shape`
         * carries the post-overload-resolution shape; fall back to the
         * raw arg type if the overload resolver couldn't pick. */
        WGSLTypeInfo *T = resolved_shape ? resolved_shape : arg0_type;
        if (!T) return wgsl_tc_set_type(tc, call, fallback, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_frexp_result(tc->types, T), 0);
    }
    case BR_MODF_RESULT: {
        WGSLTypeInfo *T = resolved_shape ? resolved_shape : arg0_type;
        if (!T) return wgsl_tc_set_type(tc, call, fallback, 0);
        return wgsl_tc_set_type(tc, call,
            wgsl_type_modf_result(tc->types, T), 0);
    }
    }
    return wgsl_tc_set_type(tc, call, fallback, 0);
}

/* ── EXPR_CALL: constructor / user-fn / builtin ────────────────────── */

static int ctor_vector_width_from_name(const char *nm, uint32_t len) {
    if (len == 4 && memcmp(nm, "vec", 3) == 0 &&
        nm[3] >= '2' && nm[3] <= '4')
    {
        return nm[3] - '0';
    }
    return 0;
}

static int ctor_matrix_dims_from_name(
    const char *nm, uint32_t len, uint8_t *cols, uint8_t *rows)
{
    if (len == 6 && memcmp(nm, "mat", 3) == 0 && nm[4] == 'x' &&
        nm[3] >= '2' && nm[3] <= '4' &&
        nm[5] >= '2' && nm[5] <= '4')
    {
        *cols = (uint8_t)(nm[3] - '0');
        *rows = (uint8_t)(nm[5] - '0');
        return 1;
    }
    return 0;
}

static int append_ctor_component_type(
    WGSLTypeChecker *tc, WGSLNode *at,
    WGSLTypeInfo **items, uint32_t cap, uint32_t *count, WGSLTypeInfo *t)
{
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_VEC) {
        for (uint32_t i = 0; i < t->width; i++) {
            if (*count >= cap) {
                wgsl_tc_error(tc, at, "too many constructor components");
                return 0;
            }
            items[(*count)++] = (WGSLTypeInfo *)t->ref;
        }
        return 1;
    }
    if (wgsl_type_is_scalar(t)) {
        if (*count >= cap) {
            wgsl_tc_error(tc, at, "too many constructor components");
            return 0;
        }
        items[(*count)++] = t;
        return 1;
    }
    wgsl_tc_error(tc, at, "constructor requires scalar or vector components");
    return 0;
}

static WGSLTypeInfo *common_constructor_type(
    WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo **items, uint32_t count)
{
    if (count == 0) return tc->types->t_abstract_int;
    WGSLTypeInfo *common = items[0];
    for (uint32_t i = 1; i < count; i++) {
        WGSLTypeInfo *t = items[i];
        if (!t) return NULL;
        if (common == t) continue;
        int ab = wgsl_type_conversion_rank(common, t);
        int ba = wgsl_type_conversion_rank(t, common);
        if (ab >= 0 && (ba < 0 || ab <= ba)) {
            common = t;
        } else if (ba >= 0) {
            /* Keep existing common type. */
        } else {
            wgsl_tc_error(tc, at, "constructor arguments have incompatible types");
            return NULL;
        }
    }
    return common;
}

static WGSLTypeInfo *constructor_matrix_elem(
    WGSLTypeChecker *tc, WGSLNode *at, WGSLTypeInfo *elem)
{
    if (elem == tc->types->t_abstract_int) elem = tc->types->t_abstract_float;
    if (!wgsl_type_is_float(elem)) {
        wgsl_tc_error(tc, at,
            "matrix constructor requires floating-point components");
        return NULL;
    }
    return elem;
}

static void check_constructor_arg_range(
    WGSLTypeChecker *tc, WGSLNode *arg, WGSLTypeInfo *target)
{
    if (!tc || !arg || !target) return;
    wgsl_tc_check_init_concretization_range(tc, arg, target);
}

static void check_vector_constructor_arg_ranges(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *elem)
{
    if (!elem) return;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, arg);
        if (t && t->kind == WGSL_TYPE_VEC) {
            check_constructor_arg_range(
                tc, arg, wgsl_type_vec(tc->types, t->width, elem));
        } else {
            check_constructor_arg_range(tc, arg, elem);
        }
    }
}

static WGSLTypeInfo *infer_typegen_constructor_type(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLSymbol *sym, int *recognized)
{
    *recognized = 0;
    if (!sym || sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) return NULL;
    const char *nm = sym->name;
    uint32_t len = sym->name_len;
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;

    int width = ctor_vector_width_from_name(nm, len);
    if (width) {
        *recognized = 1;
        if (argc == 0) {
            return wgsl_type_vec(tc->types, (uint8_t)width,
                                 tc->types->t_abstract_int);
        }
        if (argc == 1) {
            WGSLTypeInfo *arg = wgsl_tc_store_type_of(tc, call->children[1]);
            if (!arg) return NULL;
            if (wgsl_type_is_scalar(arg)) {
                return wgsl_type_vec(tc->types, (uint8_t)width, arg);
            }
        }
        WGSLTypeInfo *items[4] = {0};
        uint32_t count = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (!append_ctor_component_type(
                    tc, call->children[i], items, (uint32_t)width, &count, t))
                return NULL;
        }
        if (count != (uint32_t)width) {
            wgsl_tc_error(tc, call,
                "vector constructor expected %u component(s), got %u",
                (unsigned)width, (unsigned)count);
            return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, count);
        if (elem) check_vector_constructor_arg_ranges(tc, call, elem);
        return elem ? wgsl_type_vec(tc->types, (uint8_t)width, elem) : NULL;
    }

    uint8_t cols = 0, rows = 0;
    if (ctor_matrix_dims_from_name(nm, len, &cols, &rows)) {
        *recognized = 1;
        uint32_t total = (uint32_t)cols * (uint32_t)rows;
        if (argc == 0) {
            return wgsl_type_mat(tc->types, cols, rows,
                                 tc->types->t_abstract_int);
        }
        if (argc == 1) {
            WGSLTypeInfo *arg = wgsl_tc_store_type_of(tc, call->children[1]);
            if (!arg) return NULL;
            if (arg->kind == WGSL_TYPE_MAT &&
                arg->width == cols && arg->rows == rows)
            {
                return arg;
            }
            if (wgsl_type_is_scalar(arg)) {
                if (!wgsl_type_is_numeric(arg)) {
                    wgsl_tc_error(tc, call->children[1],
                        "matrix constructor requires numeric components");
                    return NULL;
                }
                return wgsl_type_mat(tc->types, cols, rows, arg);
            }
        }
        int has_vector_arg = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (t && t->kind == WGSL_TYPE_VEC) {
                has_vector_arg = 1;
                break;
            }
        }
        if (has_vector_arg) {
            if (argc != cols) {
                wgsl_tc_error(tc, call,
                    "matrix constructor expected %u column vector(s), got %u",
                    (unsigned)cols, (unsigned)argc);
                return NULL;
            }
            WGSLTypeInfo *items[4] = {0};
            for (uint32_t i = 0; i < cols; i++) {
                WGSLNode *argn = call->children[1 + i];
                WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, argn);
                if (!t || t->kind != WGSL_TYPE_VEC || t->width != rows) {
                    wgsl_tc_error(tc, argn,
                        "matrix constructor column vector must have width %u",
                        (unsigned)rows);
                    return NULL;
                }
                items[i] = (WGSLTypeInfo *)t->ref;
            }
            WGSLTypeInfo *elem = common_constructor_type(tc, call, items, cols);
            if (!elem) return NULL;
            elem = constructor_matrix_elem(tc, call, elem);
            if (!elem) return NULL;
            return wgsl_type_mat(tc->types, cols, rows, elem);
        }
        WGSLTypeInfo *items[16] = {0};
        uint32_t count = 0;
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, call->children[i]);
            if (!append_ctor_component_type(
                    tc, call->children[i], items, total, &count, t))
                return NULL;
        }
        if (count != total) {
            wgsl_tc_error(tc, call,
                "matrix constructor expected %u component(s), got %u",
                (unsigned)total, (unsigned)count);
            return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, count);
        if (!elem) return NULL;
        elem = constructor_matrix_elem(tc, call, elem);
        if (!elem) return NULL;
        return wgsl_type_mat(tc->types, cols, rows, elem);
    }

    if (len == 5 && memcmp(nm, "array", 5) == 0) {
        *recognized = 1;
        if (argc == 0) {
            wgsl_tc_error(tc, call, "array constructor requires at least one element");
            return NULL;
        }
        WGSLTypeInfo **items = (WGSLTypeInfo **)wgsl_arena_alloc(
            tc->arena, sizeof *items * argc);
        if (!items) return NULL;
        for (uint32_t i = 0; i < argc; i++) {
            items[i] = wgsl_tc_store_type_of(tc, call->children[1 + i]);
            if (!items[i]) return NULL;
        }
        WGSLTypeInfo *elem = common_constructor_type(tc, call, items, argc);
        if (elem) {
            for (uint32_t i = 0; i < argc; i++) {
                check_constructor_arg_range(
                    tc, call->children[1 + i], elem);
            }
        }
        return elem ? wgsl_type_array(tc->types, elem, argc) : NULL;
    }

    return NULL;
}

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
        case WGSL_SYM_TYPE_ALIAS:
        case WGSL_SYM_STRUCT:
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
        return wgsl_tc_resolve_type_spec(tc, callee);
    }
    return NULL;
}

static int constructor_elem_compatible(
    WGSLTypeChecker *tc, WGSLTypeInfo *target, WGSLTypeInfo *got)
{
    if (!target || !got) return 1;
    if (target == got) return 1;
    if (target == tc->types->t_bool) return got == tc->types->t_bool;
    if (target == tc->types->t_i32) {
        return got == tc->types->t_i32 || got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_u32) {
        return got == tc->types->t_u32 || got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_f32) {
        return got == tc->types->t_f32 ||
               got == tc->types->t_abstract_float ||
               got == tc->types->t_abstract_int;
    }
    if (target == tc->types->t_f16) {
        return got == tc->types->t_f16 ||
               got == tc->types->t_abstract_float ||
               got == tc->types->t_abstract_int;
    }
    return 0;
}

static int check_constructor_elem(
    WGSLTypeChecker *tc, WGSLNode *arg,
    WGSLTypeInfo *target_elem, WGSLTypeInfo *got_elem)
{
    if (constructor_elem_compatible(tc, target_elem, got_elem)) return 1;
    char got[96], want[96];
    wgsl_tc_error(tc, arg,
        "constructor component has type %s; expected %s",
        type_label(got_elem, got, sizeof got),
        type_label(target_elem, want, sizeof want));
    return 0;
}

static int check_explicit_vector_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    if (argc == 0) return 1;
    if (argc == 1) {
        WGSLNode *arg = call->children[1];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) return 1;
        if (at->kind == WGSL_TYPE_VEC &&
            at->width == target->width)
        {
            return 1;
        }
        if (wgsl_type_is_scalar(at)) {
            return check_constructor_elem(tc, arg, elem, at);
        }
    }

    uint32_t count = 0;
    int ok = 1;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) continue;
        if (at->kind == WGSL_TYPE_VEC) {
            count += at->width;
            ok = check_constructor_elem(
                    tc, arg, elem, (WGSLTypeInfo *)at->ref) && ok;
        } else if (wgsl_type_is_scalar(at)) {
            count += 1;
            ok = check_constructor_elem(tc, arg, elem, at) && ok;
        } else {
            wgsl_tc_error(tc, arg,
                "vector constructor requires scalar or vector components");
            ok = 0;
        }
    }
    if (count != target->width) {
        wgsl_tc_error(tc, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)target->width, (unsigned)count);
        ok = 0;
    }
    return ok;
}

static int check_explicit_matrix_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    uint32_t total = (uint32_t)target->width * (uint32_t)target->rows;
    if (argc == 0) return 1;
    if (!wgsl_type_is_float(elem)) {
        wgsl_tc_error(tc, call, "matrix constructor requires floating-point components");
        return 0;
    }
    if (argc == 1) {
        WGSLNode *arg = call->children[1];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) return 1;
        if (at->kind == WGSL_TYPE_MAT &&
            at->width == target->width && at->rows == target->rows)
        {
            WGSLTypeInfo *got_elem = (WGSLTypeInfo *)at->ref;
            if (wgsl_type_is_numeric(elem) && wgsl_type_is_numeric(got_elem)) {
                return 1;
            }
            return check_constructor_elem(
                tc, arg, elem, got_elem);
        }
        if (wgsl_type_is_scalar(at) && wgsl_type_is_numeric(at)) {
            return 1;
        }
    }

    int all_columns = argc == target->width;
    int ok = 1;
    if (all_columns) {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLNode *arg = call->children[i];
            WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
            if (!at || at->kind != WGSL_TYPE_VEC ||
                at->width != target->rows)
            {
                all_columns = 0;
                break;
            }
        }
    }
    if (all_columns) {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLNode *arg = call->children[i];
            WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
            ok = check_constructor_elem(
                    tc, arg, elem, (WGSLTypeInfo *)at->ref) && ok;
        }
        return ok;
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLNode *arg = call->children[i];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (!at) continue;
        if (!wgsl_type_is_scalar(at)) {
            wgsl_tc_error(tc, arg,
                "matrix constructor requires scalar components or column vectors");
            ok = 0;
            continue;
        }
        count += 1;
        ok = check_constructor_elem(tc, arg, elem, at) && ok;
    }
    if (count != total) {
        wgsl_tc_error(tc, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        ok = 0;
    }
    return ok;
}

static int check_explicit_struct_constructor(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLNode *sd = target ? (WGSLNode *)target->ref : NULL;
    if (!sd) return 1;

    uint32_t member_count = 0;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (m && m->kind == WGSL_NODE_DECL_STRUCT_MEMBER) member_count++;
    }
    if (argc == 0) return 1;

    int ok = 1;
    if (argc != member_count) {
        wgsl_tc_error(tc, call,
            "struct constructor expected %u member value(s), got %u",
            (unsigned)member_count, (unsigned)argc);
        ok = 0;
    }

    uint32_t arg_index = 0;
    for (uint32_t i = 0; i < sd->child_count && arg_index < argc; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
        if (attrs >= m->child_count) {
            arg_index++;
            continue;
        }
        WGSLTypeInfo *mt = wgsl_tc_store_type_of(tc, m->children[attrs]);
        WGSLNode *arg = call->children[1 + arg_index++];
        WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
        if (mt && at && wgsl_tc_init_type_mismatch(at, mt)) {
            char got[96], want[96];
            wgsl_tc_error(tc, arg,
                "struct constructor member has type %s; expected %s",
                type_label(at, got, sizeof got),
                type_label(mt, want, sizeof want));
            ok = 0;
        }
        check_constructor_arg_range(tc, arg, mt);
    }
    return ok;
}

static int check_explicit_constructor_args(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *target)
{
    if (!target) return 1;
    if (target->kind == WGSL_TYPE_ARRAY &&
        target->array_len == 0 &&
        (target->flags != 0 || target->pad_ != 0))
    {
        wgsl_tc_error(tc, call,
            "override-sized array type cannot be constructed");
        return 0;
    }
    if (target->kind == WGSL_TYPE_VEC) {
        return check_explicit_vector_constructor(tc, call, target);
    }
    if (target->kind == WGSL_TYPE_MAT) {
        return check_explicit_matrix_constructor(tc, call, target);
    }
    if (target->kind == WGSL_TYPE_STRUCT) {
        return check_explicit_struct_constructor(tc, call, target);
    }
    return 1;
}

static int type_call(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count < 1 || !n->children[0]) {
        wgsl_tc_error(tc, n, "malformed call expression");
        return 0;
    }
    /* Type all argument expressions first. */
    for (uint32_t i = 1; i < n->child_count; i++) {
        wgsl_tc_type_expr(tc, n->children[i]);
    }

    WGSLNode *callee = n->children[0];
    WGSLSymbol *sym = NULL;
    WGSLTypeInfo *target = resolve_callee_target(tc, callee, &sym);

    /* Constructor path — target type is known from the callee. */
    if (target) {
        check_explicit_constructor_args(tc, n, target);
        return wgsl_tc_set_type(tc, n, target, 0);
    }

    /* Inferred value constructors: `vec3(1)`, `mat2x2(1,2,3,4)`,
     * `array(1,2)`.  Templated forms are handled by the target path
     * above; this branch covers the predeclared type-generator head
     * without explicit template arguments. */
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
        callee->kind == WGSL_NODE_EXPR_IDENT)
    {
        int recognized = 0;
        WGSLTypeInfo *inferred =
            infer_typegen_constructor_type(tc, n, sym, &recognized);
        if (inferred) return wgsl_tc_set_type(tc, n, inferred, 0);
        return recognized ? 0 : 1;
    }

    /* User-defined function call → return type from `sym->type` (the
     * type checker writes that on the fn symbol when it visits the
     * function decl). */
    if (sym && sym->kind == WGSL_SYM_FUNCTION) {
        /* §11.4 — entry points (@vertex / @fragment / @compute) cannot
         * be call targets.  Reject the call site directly. */
        uint16_t stage_flags = (uint16_t)(
            WGSL_SYM_FLAG_VERTEX | WGSL_SYM_FLAG_FRAGMENT |
            WGSL_SYM_FLAG_COMPUTE);
        if (sym->flags & stage_flags) {
            const char *stage =
                (sym->flags & WGSL_SYM_FLAG_VERTEX)   ? "vertex" :
                (sym->flags & WGSL_SYM_FLAG_FRAGMENT) ? "fragment" :
                                                        "compute";
            wgsl_tc_error(tc, callee,
                "cannot call entry point '%.*s' (@%s); entry points "
                "must never be call targets",
                (int)sym->name_len, sym->name, stage);
        }
        /* §8.10 — argument count and per-arg type checking.  Walk the
         * function decl's param children to recover param types; each
         * param's tspec was resolved during type_function, so the type
         * lives on the side-table. */
        WGSLNode *fn = sym->ast;
        if (fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
            uint32_t FA = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
            uint32_t P  = (uint32_t)(fn->payload[1] >> 32);
            uint32_t call_argc = (n->child_count >= 1) ? n->child_count - 1 : 0;
            if (call_argc != P) {
                wgsl_tc_error(tc, n,
                    "argument count mismatch for '%.*s': expected %u, got %u",
                    (int)sym->name_len, sym->name,
                    (unsigned)P, (unsigned)call_argc);
            } else {
                /* Per-arg type compatibility — same predicate the
                 * decl-init mismatch check uses. */
                for (uint32_t k = 0; k < P && (FA + k) < fn->child_count; k++) {
                    WGSLNode *param = fn->children[FA + k];
                    if (!param || param->kind != WGSL_NODE_DECL_PARAM) continue;
                    uint32_t pattrs = (uint32_t)(param->payload[1] & 0xFFFFFFFFu);
                    if (pattrs >= param->child_count) continue;
                    WGSLTypeInfo *pt = wgsl_tc_store_type_of(
                        tc, param->children[pattrs]);
                    WGSLNode *arg = n->children[1 + k];
                    WGSLTypeInfo *at = wgsl_tc_store_type_of(tc, arg);
                    if (!pt || !at) continue;
                    if (wgsl_tc_init_type_mismatch(at, pt)) {
                        char got[96], want[96];
                        wgsl_tc_error(tc, arg,
                            "argument %u of '%.*s' has type %s; "
                            "expected %s",
                            (unsigned)(k + 1),
                            (int)sym->name_len, sym->name,
                            type_label(at, got, sizeof got),
                            type_label(pt, want, sizeof want));
                    }
                }
            }
        }
        WGSLTypeInfo *ret = sym->type;
        if (!ret && fn && fn->kind == WGSL_NODE_DECL_FUNCTION) {
            uint32_t FA = (uint32_t)(fn->payload[1] & 0xFFFFFFFFu);
            uint32_t P  = (uint32_t)(fn->payload[1] >> 32);
            uint32_t RA = (uint32_t)(fn->payload[2] & 0xFFFFFFFFu);
            uint32_t has_ret = (uint32_t)(fn->payload[2] >> 32);
            uint32_t ri = FA + P + RA;
            if (has_ret && ri < fn->child_count) {
                ret = wgsl_tc_resolve_type_spec(tc, fn->children[ri]);
            }
        }
        if (!ret) ret = tc->types->t_void;
        return wgsl_tc_set_type(tc, n, ret, 0);
    }

    /* Builtin function call — dispatched via the hand-coded overload
     * table (`type_call_builtin`).  Per-spec full overload set lives
     * in `def/wgsl.def` + codegen, deferred to v1.x; the surface
     * below covers everything the real-shader corpus exercises. */
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_FN) {
        return type_call_builtin(tc, n, callee, sym);
    }

    if (sym) {
        wgsl_tc_error(tc, callee,
            "'%.*s' is not callable",
            (int)sym->name_len, sym->name);
        return 0;
    }

    /* Unknown / unresolved callee — quietly punt without erroring; the
     * resolver will already have flagged it if it's truly broken. */
    return 1;
}

void wgsl_tc_get_ref_space_and_mode(WGSLTypeChecker *tc, const WGSLNode *n, WGSLAddressSpace *as, WGSLAccessMode *am) {
    if (!n) return;
    WGSLTypeInfo *typed = wgsl_typecheck_type_of(tc, n);
    if (typed &&
        (typed->kind == WGSL_TYPE_PTR || typed->kind == WGSL_TYPE_REF))
    {
        *as = (WGSLAddressSpace)(typed->flags & 0xFFu);
        *am = (WGSLAccessMode)((typed->flags >> 8) & 0xFFu);
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_PAREN) {
        if (n->child_count > 0) {
            wgsl_tc_get_ref_space_and_mode(tc, n->children[0], as, am);
        }
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        WGSLSymbol *s = wgsl_node_resolved_symbol((WGSLNode*)n);
        if (s) { *as = (WGSLAddressSpace)s->as; *am = (WGSLAccessMode)s->am; }
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_MEMBER || n->kind == WGSL_NODE_EXPR_INDEX) {
        if (n->child_count > 0) wgsl_tc_get_ref_space_and_mode(tc, n->children[0], as, am);
        return;
    }
    if (n->kind == WGSL_NODE_EXPR_INDIRECTION) {
        if (n->child_count > 0) {
            WGSLTypeInfo *pt = wgsl_tc_store_type_of(tc, n->children[0]);
            if (pt && pt->kind == WGSL_TYPE_PTR) {
                *as = (WGSLAddressSpace)(pt->flags & 0xFF);
                *am = (WGSLAccessMode)((pt->flags >> 8) & 0xFF);
            }
        }
        return;
    }
}

static int type_addr_of(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0 || !n->children[0]) return 0;
    WGSLNode *expr = n->children[0];
    if (!wgsl_tc_type_expr(tc, expr)) return 0;

    if (!wgsl_typecheck_is_ref(tc, expr)) {
        wgsl_tc_error(tc, expr, "cannot take address of value; operand must be a reference");
        return 0;
    }

    /* §8.13 — `&` cannot take the address of a single vector component
     * (`&v.x`).  `v.x` is a ref<vec elem> per the single-letter swizzle
     * rule, which would otherwise satisfy the ref check above.  Detect
     * the EXPR_MEMBER-on-a-vec pattern with a 1-character member name. */
    if (expr->kind == WGSL_NODE_EXPR_MEMBER && expr->child_count >= 2) {
        WGSLNode *base = expr->children[0];
        WGSLNode *mem  = expr->children[1];
        WGSLTypeInfo *bt = base ? wgsl_tc_store_type_of(tc, base) : NULL;
        if (bt && bt->kind == WGSL_TYPE_VEC && mem) {
            uint32_t mlen = (uint32_t)(mem->payload[0] >> 32);
            if (mlen == 1) {
                wgsl_tc_error(tc, n,
                    "cannot take address of a vector component "
                    "(spec §8.13)");
                return 0;
            }
        }
    }

    WGSLTypeInfo *elem = wgsl_tc_store_type_of(tc, expr);
    if (!elem) return 0;

    /* §8.13 — `&t` on a texture / sampler is rejected.  Handle-AS
     * resources cannot have their address taken; only var-allocated
     * memory in function/private/workgroup/uniform/storage can. */
    if (elem->kind == WGSL_TYPE_TEXTURE || elem->kind == WGSL_TYPE_SAMPLER) {
        wgsl_tc_error(tc, n,
            "cannot take address of a handle-address-space variable "
            "(spec §8.13)");
        return 0;
    }

    WGSLAddressSpace as = WGSL_AS_NONE;
    WGSLAccessMode   am = WGSL_ACCESS_NONE;
    wgsl_tc_get_ref_space_and_mode(tc, expr, &as, &am);
    if (as == WGSL_AS_HANDLE) {
        wgsl_tc_error(tc, n,
            "cannot take address of a handle-address-space variable "
            "(spec §8.13)");
        return 0;
    }

    WGSLTypeInfo *ptr = wgsl_type_ptr(tc->types, as, elem, am);
    return wgsl_tc_set_type(tc, n, ptr, 0);
}

static int type_indirection(WGSLTypeChecker *tc, WGSLNode *n) {
    if (n->child_count == 0 || !n->children[0]) return 0;
    WGSLNode *expr = n->children[0];
    if (!wgsl_tc_type_expr(tc, expr)) return 0;
    
    WGSLTypeInfo *ptr = wgsl_tc_store_type_of(tc, expr);
    if (!ptr || ptr->kind != WGSL_TYPE_PTR) {
        wgsl_tc_error(tc, expr, "cannot dereference non-pointer type");
        return 0;
    }
    
    WGSLTypeInfo *elem = (WGSLTypeInfo*)ptr->ref;
    return wgsl_tc_set_type(tc, n, elem, 1);
}

int wgsl_tc_type_expr(WGSLTypeChecker *tc, WGSLNode *n) {
    if (!n) return 0;
    if (n->flags & WGSL_FLAG_TYPED) return 1;

    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_EXPR_LITERAL_INT:   return type_literal_int  (tc, n);
    case WGSL_NODE_EXPR_LITERAL_FLOAT: return type_literal_float(tc, n);
    case WGSL_NODE_EXPR_LITERAL_BOOL:
        return wgsl_tc_set_type(tc, n, tc->types->t_bool, 0);

    case WGSL_NODE_EXPR_IDENT: return type_ident(tc, n);

    case WGSL_NODE_EXPR_PAREN: {
        if (n->child_count == 0 || !n->children[0]) return 0;
        if (!wgsl_tc_type_expr(tc, n->children[0])) return 0;
        WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
        int is_ref = (n->children[0]->flags & WGSL_FLAG_IS_REF) != 0;
        return wgsl_tc_set_type(tc, n, t, is_ref);
    }

    case WGSL_NODE_EXPR_UNARY:  return type_unary (tc, n);
    case WGSL_NODE_EXPR_BINARY: return type_binary(tc, n);
    case WGSL_NODE_EXPR_CALL:   return type_call  (tc, n);
    case WGSL_NODE_EXPR_MEMBER: return type_member(tc, n);
    case WGSL_NODE_EXPR_INDEX:  return type_index (tc, n);

    case WGSL_NODE_EXPR_ADDR_OF:        return type_addr_of(tc, n);
    case WGSL_NODE_EXPR_INDIRECTION:    return type_indirection(tc, n);

    /* templated-ident-as-value: deferred to
     * v1.x.  Walk children for descendant typing; do not assign a type. */
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        for (uint32_t i = 0; i < n->child_count; i++) {
            wgsl_tc_type_expr(tc, n->children[i]);
        }
        return 1;

    default:
        return 1;
    }
}
