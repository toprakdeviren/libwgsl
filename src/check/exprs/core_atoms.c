/**
 * @file core_atoms.c — literals, idents, common types, unary, consteval helpers
 */
#include "internal/exprs_priv.h"


WGSLTypeInfo *contextual_type(
    WGSLTypeInfo *got, WGSLTypeInfo *expected)
{
    if (!got || !expected || got == expected) return got;
    if (!wgsl_type_is_abstract(got)) return got;
    if (wgsl_type_conversion_rank(got, expected) < 0) return got;
    return expected;
}

int type_literal_int(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    uint32_t suf = (uint32_t)wgsl_lit_raw(n);
    uint32_t base = suf & 0x0fu;
    WGSLTypeInfo *t;
    switch (base) {
    case WGSL_NUM_SUFFIX_I: t = tc->types->t_i32; break;
    case WGSL_NUM_SUFFIX_U: t = tc->types->t_u32; break;
    default:                t = tc->types->t_abstract_int; break;
    }
    /* Bidirectional: unsuffixed abstract literal materializes to the
     * contextual concrete (or abstract-float) type when conversion is
     * feasible.  Suffixed literals keep their explicit type. */
    if (base == WGSL_NUM_ABSTRACT) {
        WGSLTypeInfo *ctx = contextual_type(t, expected);
        if (ctx && ctx != t) t = ctx;
    }
    if ((t == tc->types->t_i32 || t == tc->types->t_u32) &&
        tc->cev && !(n->flags & WGSL_FLAG_CONST_EVALED))
    {
        WGSLValue v = {0};
        if (!wgsl_consteval_expr(tc->cev, n, &v)) {
            tc->had_error = 1;
        } else if (v.kind == WGSL_VAL_INT) {
            if (t == tc->types->t_i32 &&
                (v.u.i < INT32_MIN || v.u.i > INT32_MAX))
            {
                wgsl_tc_error(tc, n, "value %lld out of range for i32",
                    (long long)v.u.i);
            }
            if (t == tc->types->t_u32 &&
                (v.u.i < 0 || v.u.i > (int64_t)UINT32_MAX))
            {
                wgsl_tc_error(tc, n, "value %lld out of range for u32",
                    (long long)v.u.i);
            }
        }
    }
    return wgsl_tc_set_type(tc, n, t, 0);
}

int type_literal_float(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    uint32_t suf = (uint32_t)wgsl_lit_raw(n);
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
    if ((suf & 0x0fu) == WGSL_NUM_ABSTRACT) {
        WGSLTypeInfo *ctx = contextual_type(t, expected);
        if (ctx && ctx != t) {
            t = ctx;
            needs_range_check =
                (t == tc->types->t_f32 || t == tc->types->t_f16);
        }
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

int type_ident(WGSLTypeChecker *tc, WGSLNode *n) {
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
WGSLTypeInfo *common_scalar_type(
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
WGSLTypeInfo *common_componentwise(
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
WGSLTypeInfo *try_matrix_multiply(
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

int type_unary(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    if (n->child_count != 1 || !n->children[0]) {
        wgsl_tc_error(tc, n, "malformed unary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
    /* §4.3 — `~` preserves operand type: push expected so
     * `let x: u32 = ~0` materializes 0 as u32 before complement.
     * `-` must NOT push a concrete expected: (1) unary minus is
     * undefined for u32, (2) `let x: i32 = -2147483648` needs the
     * magnitude to stay AbstractInt until after the sign is applied.
     * `!` expects bool. */
    WGSLTypeInfo *child_exp = NULL;
    if (op == WGSL_TOK_BANG) {
        child_exp = tc->types->t_bool;
    } else if (op == WGSL_TOK_TILDE) {
        child_exp = expected;
    }
    if (!wgsl_tc_type_expr_exp(tc, n->children[0], child_exp)) return 0;
    WGSLTypeInfo *t = wgsl_tc_store_type_of(tc, n->children[0]);
    if (!t) return 0;

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
        /* Keep abstract; use-site materialize does the range check
         * (`let x: u32 = -1` must fail). */
        (void)expected;
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
        /* Child already saw `expected` (so `~0` under u32 is u32). */
        return wgsl_tc_set_type(tc, n, t, 0);
    default:
        wgsl_tc_error(tc, n, "unsupported unary operator");
        return 0;
    }
}

/* Element type of a numeric scalar / vec / mat / array.  NULL otherwise. */
WGSLTypeInfo *element_type_of(const WGSLTypeInfo *t) {
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

/* Returns 1 on successful const-eval, 0 when the expression is simply
 * non-constant (noise), and -1 when const-eval surfaced a real value error.
 * Branches on `cev->store.last_status`, never on diagnostic message text. */
int try_consteval_expr_soft(
    WGSLTypeChecker *tc, WGSLNode *expr, WGSLValue *out)
{
    if (!tc || !tc->cev || !tc->diag || !expr || !out) return 0;
    if (tc->suppress_consteval_value_errors) return 0;
    size_t saved_count = tc->diag->count;
    int saved_errs = tc->diag->error_count;
    int saved_cev_err = tc->cev->store.had_error;
    memset(out, 0, sizeof *out);

    if (wgsl_consteval_expr(tc->cev, expr, out)) return 1;

    if (tc->cev->store.last_status == WGSL_CEV_VALUE_ERROR) {
        tc->had_error = 1;
        return -1;
    }

    /* Trial const-eval missed; restore diagnostics and keep checking. */
    tc->diag->count = saved_count;
    tc->diag->error_count = saved_errs;
    tc->cev->store.had_error = saved_cev_err;
    tc->cev->store.last_status = WGSL_CEV_OK;
    return 0;
}

int const_value_contains_zero(const WGSLValue *v)
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

int const_value_shift_amount_out_of_range(
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
