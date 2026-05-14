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

