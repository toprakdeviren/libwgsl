static int int_fits_target(int64_t v, WGSLTypeKind k) {
    switch (k) {
    case WGSL_TYPE_I32:
        return v >= INT32_MIN && v <= INT32_MAX;
    case WGSL_TYPE_U32:
        /* WGSL §6.2.1: AbstractInt → u32 only if value ≥ 0. */
        return v >= 0 && v <= (int64_t)UINT32_MAX;
    case WGSL_TYPE_ABSTRACT_INT:
        return 1;
    default:
        return 0;
    }
}

static int float_fits_target(double v, WGSLTypeKind k) {
    /* Spec defers to IEEE-754 finite range; we accept any finite + 0. */
    switch (k) {
    case WGSL_TYPE_F32:
        if (!isfinite(v)) return 1;            /* allow ±inf if produced */
        return v >= -FLT_MAX && v <= FLT_MAX;
    case WGSL_TYPE_F16:
        /* f16 max ≈ 65504 */
        return !isfinite(v) || (v >= -65504.0 && v <= 65504.0);
    case WGSL_TYPE_ABSTRACT_FLOAT:
        return 1;
    default:
        return 0;
    }
}

int wgsl_consteval_materialize(
    WGSLConstEvaluator *cev, WGSLValue *v,
    WGSLTypeInfo *target, const WGSLNode *at_node)
{
    if (!v || v->kind == WGSL_VAL_INVALID) return 0;
    if (!v->type) return 0;

    /* Default concretisation when no target requested. */
    if (!target) {
        target = wgsl_type_concretize(cev->types, v->type);
    }
    if (v->type == target) return 1;

    /* Conversion-rank gate (§6.1.2). */
    int rank = wgsl_type_conversion_rank(v->type, target);
    if (rank < 0) {
        cev_error(cev, at_node,
            "cannot convert value of type %s to %s",
            wgsl_type_kind_name((WGSLTypeKind)v->type->kind),
            wgsl_type_kind_name((WGSLTypeKind)target->kind));
        return 0;
    }

    /* Apply the value-side conversion. */
    WGSLTypeKind tk = (WGSLTypeKind)target->kind;
    if (val_is_int(v)) {
        if (tk == WGSL_TYPE_F32 || tk == WGSL_TYPE_F16 ||
            tk == WGSL_TYPE_ABSTRACT_FLOAT)
        {
            double d = (double)v->u.i;
            if (!float_fits_target(d, tk)) {
                cev_error(cev, at_node,
                    "value %lld out of range for %s",
                    (long long)v->u.i, wgsl_type_kind_name(tk));
                return 0;
            }
            v->kind = WGSL_VAL_FLOAT;
            v->u.f  = d;
        } else if (!int_fits_target(v->u.i, tk)) {
            cev_error(cev, at_node,
                "value %lld out of range for %s",
                (long long)v->u.i, wgsl_type_kind_name(tk));
            return 0;
        }
        v->type = target;
        return 1;
    }
    if (val_is_float(v)) {
        if (!float_fits_target(v->u.f, tk)) {
            cev_error(cev, at_node,
                "value out of range for %s", wgsl_type_kind_name(tk));
            return 0;
        }
        v->type = target;
        return 1;
    }
    if ((v->kind == WGSL_VAL_VEC && target->kind == WGSL_TYPE_VEC) ||
        (v->kind == WGSL_VAL_MAT && target->kind == WGSL_TYPE_MAT) ||
        (v->kind == WGSL_VAL_ARRAY && target->kind == WGSL_TYPE_ARRAY))
    {
        uint32_t expected = 0;
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        if (target->kind == WGSL_TYPE_VEC) {
            expected = target->width;
        } else if (target->kind == WGSL_TYPE_MAT) {
            expected = (uint32_t)target->width * (uint32_t)target->rows;
        } else {
            expected = target->array_len;
        }
        if (expected == 0 || v->u.agg.count != expected) {
            cev_error(cev, at_node, "aggregate constructor shape mismatch");
            return 0;
        }
        for (uint32_t i = 0; i < v->u.agg.count; i++) {
            if (!wgsl_consteval_materialize(
                    cev, &v->u.agg.elems[i], elem, at_node))
                return 0;
        }
        v->type = target;
        return 1;
    }

    if (v->kind == WGSL_VAL_STRUCT && target->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)target->ref;
        if (!sd || v->type->ref != target->ref) {
            cev_error(cev, at_node, "struct constructor type mismatch");
            return 0;
        }
        uint32_t idx = 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            if (idx >= v->u.agg.count) {
                cev_error(cev, at_node, "struct constructor shape mismatch");
                return 0;
            }
            uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            if (attrs >= m->child_count) return 0;
            WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
            if (mt && !wgsl_consteval_materialize(
                    cev, &v->u.agg.elems[idx], mt, at_node))
                return 0;
            idx++;
        }
        v->type = target;
        return idx == v->u.agg.count;
    }

    /* bool or non-convertible aggregate. */
    return target == v->type;
}

/* ── Pair promotion (§6.1.2) ───────────────────────────────────────── */

static int promote_pair(
    WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b,
    const WGSLNode *at_node)
{
    if (a->type == b->type) return 1;
    int ab = wgsl_type_conversion_rank(a->type, b->type);
    int ba = wgsl_type_conversion_rank(b->type, a->type);

    /* Prefer concretising abstract operand when one side is concrete. */
    int a_abs = wgsl_type_is_abstract(a->type);
    int b_abs = wgsl_type_is_abstract(b->type);
    if (a_abs && !b_abs && ab >= 0) {
        return wgsl_consteval_materialize(cev, a, b->type, at_node);
    }
    if (b_abs && !a_abs && ba >= 0) {
        return wgsl_consteval_materialize(cev, b, a->type, at_node);
    }
    /* Both abstract — promote to AbstractFloat if either side is so. */
    if (a_abs && b_abs) {
        if (a->type == cev->types->t_abstract_float ||
            b->type == cev->types->t_abstract_float)
        {
            return wgsl_consteval_materialize(
                       cev, a, cev->types->t_abstract_float, at_node) &&
                   wgsl_consteval_materialize(
                       cev, b, cev->types->t_abstract_float, at_node);
        }
        /* both AbstractInt — already equal, handled by first check */
    }
    /* Both concrete and different — error. */
    if (ab >= 0) return wgsl_consteval_materialize(cev, a, b->type, at_node);
    if (ba >= 0) return wgsl_consteval_materialize(cev, b, a->type, at_node);

    cev_error(cev, at_node,
        "no implicit conversion between %s and %s",
        wgsl_type_kind_name((WGSLTypeKind)a->type->kind),
        wgsl_type_kind_name((WGSLTypeKind)b->type->kind));
    return 0;
}

/* ── Operator folding ──────────────────────────────────────────────── */

/* Apply `% / /` for ints, taking signedness into account. */
static int int_div_mod(
    WGSLConstEvaluator *cev, WGSLNode *at,
    int64_t a, int64_t b, int is_unsigned, int is_mod, int64_t *out)
{
    if (b == 0) {
        cev_error(cev, at, "%s by zero", is_mod ? "modulo" : "division");
        return 0;
    }
    if (is_unsigned) {
        uint64_t ua = (uint32_t)a;       /* low 32 bits */
        uint64_t ub = (uint32_t)b;
        *out = (int64_t)(is_mod ? (ua % ub) : (ua / ub));
        return 1;
    }
    /* Signed: spec mirrors C99: division truncates toward zero,
     * modulo carries the sign of the dividend. */
    if (a == INT64_MIN && b == -1) {
        cev_error(cev, at, "signed integer overflow in division");
        return 0;
    }
    *out = is_mod ? (a % b) : (a / b);
    return 1;
}

static int checked_abstract_int_result(
    WGSLConstEvaluator *cev, WGSLNode *at, __int128 value,
    const char *op_name, int64_t *out)
{
    if (value < (__int128)INT64_MIN || value > (__int128)INT64_MAX) {
        cev_error(cev, at,
            "signed integer overflow in %s for AbstractInt", op_name);
        return 0;
    }
    *out = (int64_t)value;
    return 1;
}

static int eval_binary_int(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    int64_t ai = a->u.i, bi = b->u.i;
    int is_u = (a->type == cev->types->t_u32);
    int64_t r = 0;

    switch (op) {
    case WGSL_TOK_PLUS:   r = ai + bi; break;
    case WGSL_TOK_MINUS:  r = ai - bi; break;
    case WGSL_TOK_STAR:   r = ai * bi; break;
    case WGSL_TOK_SLASH:
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            cev_error(cev, at, "signed integer overflow in division");
            return 0;
        }
        if (!int_div_mod(cev, at, ai, bi, is_u, 0, &r)) return 0;
        break;
    case WGSL_TOK_PERCENT:
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            cev_error(cev, at, "signed integer overflow in division");
            return 0;
        }
        if (!int_div_mod(cev, at, ai, bi, is_u, 1, &r)) return 0;
        break;
    case WGSL_TOK_AMP:    r = ai & bi; break;
    case WGSL_TOK_PIPE:   r = ai | bi; break;
    case WGSL_TOK_CARET:  r = ai ^ bi; break;
    case WGSL_TOK_LESS_LESS: {
        /* §8.9 — shift amount must be < bit-width of the target type.
         * For concrete i32 / u32 the width is 32; for AbstractInt the
         * spec allows the full 64-bit AbstractInt value but the result
         * must still fit when materialized.  We enforce the obvious
         * runtime case here (`1u << 33u` etc.). */
        uint8_t bit_width =
            (a->type == cev->types->t_i32 || a->type == cev->types->t_u32) ? 32 : 64;
        if (bi < 0 || (uint64_t)bi >= bit_width) {
            cev_error(cev, at,
                "shift amount %lld is out of range for %s",
                (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
            return 0;
        }
        uint64_t s = (uint64_t)bi;
        if (a->type == cev->types->t_u32) {
            uint64_t wr = (uint64_t)(uint32_t)ai << s;
            if (wr > UINT32_MAX) {
                cev_error(cev, at, "left shift result out of range for u32");
                return 0;
            }
            r = (int64_t)wr;
        } else {
            __int128 wr = (__int128)ai << s;
            __int128 lo = a->type == cev->types->t_i32
                ? (__int128)INT32_MIN : (__int128)INT64_MIN;
            __int128 hi = a->type == cev->types->t_i32
                ? (__int128)INT32_MAX : (__int128)INT64_MAX;
            if (wr < lo || wr > hi) {
                cev_error(cev, at, "left shift result out of range for %s",
                    wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
                return 0;
            }
            r = (int64_t)wr;
        }
        break;
    }
    case WGSL_TOK_GREATER_GREATER: {
        if (a->type == cev->types->t_abstract_int) {
            if (bi < 0) {
                cev_error(cev, at,
                    "shift amount %lld is out of range for AbstractInt",
                    (long long)bi);
                return 0;
            }
            if (bi >= 63) {
                r = ai < 0 ? -1 : 0;
            } else {
                r = ai >> (uint64_t)bi;
            }
            break;
        }
        uint8_t bit_width =
            (a->type == cev->types->t_i32 || a->type == cev->types->t_u32) ? 32 : 64;
        if (bi < 0 || (uint64_t)bi >= bit_width) {
            cev_error(cev, at,
                "shift amount %lld is out of range for %s",
                (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
            return 0;
        }
        uint64_t s = (uint64_t)bi;
        if (is_u) r = (int64_t)((uint32_t)ai >> s);
        else      r = ai >> s;
        break;
    }
    case WGSL_TOK_EQUAL_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (ai == bi); return 1;
    case WGSL_TOK_BANG_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (ai != bi); return 1;
    case WGSL_TOK_LESS:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai <  (uint32_t)bi) : (ai <  bi); return 1;
    case WGSL_TOK_LESS_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai <= (uint32_t)bi) : (ai <= bi); return 1;
    case WGSL_TOK_GREATER:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai >  (uint32_t)bi) : (ai >  bi); return 1;
    case WGSL_TOK_GREATER_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai >= (uint32_t)bi) : (ai >= bi); return 1;
    default:
        cev_error(cev, at, "unsupported integer operator");
        return 0;
    }
    /* Truncate to target width for concrete ints to mirror runtime. */
    if (a->type == cev->types->t_i32) r = (int32_t)r;
    else if (is_u)                    r = (int64_t)(uint32_t)r;

    out->kind = WGSL_VAL_INT;
    out->type = a->type;        /* a and b were promoted to the same type */
    out->u.i  = r;
    return 1;
}

static int round_float_value_to_type(
    WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out);

static int eval_binary_float(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    double af = a->u.f, bf = b->u.f, r = 0.0;
    switch (op) {
    case WGSL_TOK_PLUS:    r = af + bf; break;
    case WGSL_TOK_MINUS:   r = af - bf; break;
    case WGSL_TOK_STAR:    r = af * bf; break;
    case WGSL_TOK_SLASH:
        if (bf == 0.0) {
            cev_error(cev, at, "division by zero");
            return 0;
        }
        r = af / bf; break;
    case WGSL_TOK_PERCENT:
        if (bf == 0.0) {
            cev_error(cev, at, "modulo by zero");
            return 0;
        }
        /* WGSL §8.7.4: float % uses x - y * trunc(x / y). */
        r = af - bf * trunc(af / bf); break;
    case WGSL_TOK_EQUAL_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af == bf); return 1;
    case WGSL_TOK_BANG_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af != bf); return 1;
    case WGSL_TOK_LESS:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af <  bf); return 1;
    case WGSL_TOK_LESS_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af <= bf); return 1;
    case WGSL_TOK_GREATER:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af >  bf); return 1;
    case WGSL_TOK_GREATER_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af >= bf); return 1;
    default:
        cev_error(cev, at, "unsupported float operator");
        return 0;
    }
    /* §8.7 / §15.7.3 — finite-math assumption.  Inputs are finite at
     * this point (the consteval doesn't import non-finite literals);
     * any non-finite result therefore comes from a domain violation
     * (e.g., `inf - inf` derived through chained ops, or AbstractFloat
     * overflow at ~1e308 from `1e200 * 1e200`). */
    double rounded = 0.0;
    if (!round_float_value_to_type(cev, a->type, r, &rounded)) {
        if (isnan(r)) {
            cev_error(cev, at,
                "floating-point operation produced NaN "
                "(spec §15.7.3 finite-math assumption)");
        } else {
            cev_error(cev, at,
                "floating-point operation produced infinity "
                "(spec §15.7.3 finite-math assumption)");
        }
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = a->type;
    out->u.f  = rounded;
    return 1;
}

static int eval_binary_bool(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    bool ab = a->u.b, bb = b->u.b;
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    switch (op) {
    case WGSL_TOK_AMP_AMP:   out->u.b = ab && bb; return 1;
    case WGSL_TOK_PIPE_PIPE: out->u.b = ab || bb; return 1;
    case WGSL_TOK_AMP:       out->u.b = ab && bb; return 1;
    case WGSL_TOK_PIPE:      out->u.b = ab || bb; return 1;
    case WGSL_TOK_EQUAL_EQUAL: out->u.b = (ab == bb); return 1;
    case WGSL_TOK_BANG_EQUAL:  out->u.b = (ab != bb); return 1;
    default:
        cev_error(cev, at, "operator does not apply to bool operands");
        return 0;
    }
}

static int eval_binary_componentwise(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
        cev_error(cev, n, "logical operator requires bool scalar operands");
        return 0;
    }

    uint32_t aw = value_width(a);
    uint32_t bw = value_width(b);
    uint32_t width = aw > bw ? aw : bw;
    if (aw != 1 && bw != 1 && aw != bw) {
        cev_error(cev, n, "vector operand width mismatch");
        return 0;
    }

    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue av = value_component(a, aw == 1 ? 0 : i);
        WGSLValue bv = value_component(b, bw == 1 ? 0 : i);
        if (val_is_bool(&av) || val_is_bool(&bv)) {
            if (!val_is_bool(&av) || !val_is_bool(&bv)) {
                cev_error(cev, n,
                    "operator does not apply to mixed bool and numeric operands");
                return 0;
            }
            if (!eval_binary_bool(cev, n, op, &av, &bv, &tmp[i])) return 0;
            continue;
        }
        if (!val_is_numeric(&av) || !val_is_numeric(&bv)) {
            cev_error(cev, n, "operator requires numeric or bool operands");
            return 0;
        }
        if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
            if (!val_is_int(&av) || !val_is_int(&bv)) {
                cev_error(cev, n, "shift requires integer operands");
                return 0;
            }
            if (bv.type != cev->types->t_u32 &&
                bv.type != cev->types->t_abstract_int)
            {
                cev_error(cev, n, "shift amount must be u32");
                return 0;
            }
            if (!eval_binary_int(cev, n, op, &av, &bv, &tmp[i])) return 0;
            continue;
        }
        if (!promote_pair(cev, &av, &bv, n)) return 0;
        if (val_is_int(&av)) {
            if (!eval_binary_int(cev, n, op, &av, &bv, &tmp[i])) return 0;
        } else if (val_is_float(&av)) {
            if (!eval_binary_float(cev, n, op, &av, &bv, &tmp[i])) return 0;
        } else {
            cev_error(cev, n, "unhandled operand kind in vector binary op");
            return 0;
        }
    }

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector binary op");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = tmp[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, tmp[0].type);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_numeric_scalar_binary(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, n)) return 0;
    if (val_is_int(a)) return eval_binary_int(cev, n, op, a, b, out);
    if (val_is_float(a)) return eval_binary_float(cev, n, op, a, b, out);
    cev_error(cev, n, "operator requires numeric operands");
    return 0;
}

static int make_matrix_binary_result(
    WGSLConstEvaluator *cev, WGSLNode *n,
    uint32_t cols, uint32_t rows, WGSLValue *elems, WGSLValue *out)
{
    if (cols < 2 || cols > 4 || rows < 2 || rows > 4) return 0;
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * cols * rows);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding matrix binary");
        return 0;
    }
    for (uint32_t i = 0; i < cols * rows; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(cev->types, (uint8_t)cols, (uint8_t)rows, elem);
    out->u.agg.count = cols * rows;
    out->u.agg.elems = stored;
    return 1;
}

static int make_vec_binary_result(
    WGSLConstEvaluator *cev, WGSLNode *n,
    uint32_t width, WGSLValue *elems, WGSLValue *out)
{
    if (width < 2 || width > 4) return 0;
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector binary");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_matrix_binary(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (op != WGSL_TOK_PLUS && op != WGSL_TOK_MINUS && op != WGSL_TOK_STAR)
        return 0;

    if (a->kind == WGSL_VAL_MAT && b->kind == WGSL_VAL_MAT) {
        if (!a->type || !b->type ||
            a->type->kind != WGSL_TYPE_MAT || b->type->kind != WGSL_TYPE_MAT)
            return 0;

        if (op == WGSL_TOK_PLUS || op == WGSL_TOK_MINUS) {
            if (a->type->width != b->type->width ||
                a->type->rows != b->type->rows)
                return 0;
            WGSLValue tmp[16] = {{0}};
            uint32_t count = a->type->width * a->type->rows;
            for (uint32_t i = 0; i < count; i++) {
                WGSLValue av = a->u.agg.elems[i];
                WGSLValue bv = b->u.agg.elems[i];
                if (!eval_numeric_scalar_binary(cev, n, op, &av, &bv, &tmp[i]))
                    return 0;
            }
            return make_matrix_binary_result(
                cev, n, a->type->width, a->type->rows, tmp, out);
        }

        /* mat<K,R> * mat<C,K> -> mat<C,R>. */
        if (a->type->width != b->type->rows) return 0;
        uint32_t K = a->type->width;
        uint32_t R = a->type->rows;
        uint32_t C = b->type->width;
        WGSLValue tmp[16] = {{0}};
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t r = 0; r < R; r++) {
                WGSLValue sum = {0};
                for (uint32_t k = 0; k < K; k++) {
                    WGSLValue av = a->u.agg.elems[k * R + r];
                    WGSLValue bv = b->u.agg.elems[c * K + k];
                    WGSLValue prod = {0};
                    if (!eval_numeric_scalar_binary(
                            cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                        return 0;
                    if (k == 0) {
                        sum = prod;
                    } else if (!eval_numeric_scalar_binary(
                                   cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                    {
                        return 0;
                    }
                }
                tmp[c * R + r] = sum;
            }
        }
        return make_matrix_binary_result(cev, n, C, R, tmp, out);
    }

    if ((a->kind == WGSL_VAL_MAT && val_is_numeric(b)) ||
        (b->kind == WGSL_VAL_MAT && val_is_numeric(a)))
    {
        WGSLValue *mat = a->kind == WGSL_VAL_MAT ? a : b;
        WGSLValue *scalar = a->kind == WGSL_VAL_MAT ? b : a;
        if (!mat->type || mat->type->kind != WGSL_TYPE_MAT) return 0;
        WGSLValue tmp[16] = {{0}};
        uint32_t count = mat->type->width * mat->type->rows;
        for (uint32_t i = 0; i < count; i++) {
            WGSLValue av = a->kind == WGSL_VAL_MAT
                ? mat->u.agg.elems[i] : *scalar;
            WGSLValue bv = a->kind == WGSL_VAL_MAT
                ? *scalar : mat->u.agg.elems[i];
            if (!eval_numeric_scalar_binary(cev, n, op, &av, &bv, &tmp[i]))
                return 0;
        }
        return make_matrix_binary_result(
            cev, n, mat->type->width, mat->type->rows, tmp, out);
    }

    if (op == WGSL_TOK_STAR &&
        a->kind == WGSL_VAL_MAT && b->kind == WGSL_VAL_VEC &&
        a->type && b->type &&
        a->type->kind == WGSL_TYPE_MAT && b->type->kind == WGSL_TYPE_VEC &&
        a->type->width == b->type->width)
    {
        uint32_t C = a->type->width;
        uint32_t R = a->type->rows;
        WGSLValue tmp[4] = {{0}};
        for (uint32_t r = 0; r < R; r++) {
            WGSLValue sum = {0};
            for (uint32_t c = 0; c < C; c++) {
                WGSLValue av = a->u.agg.elems[c * R + r];
                WGSLValue bv = b->u.agg.elems[c];
                WGSLValue prod = {0};
                if (!eval_numeric_scalar_binary(
                        cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                    return 0;
                if (c == 0) {
                    sum = prod;
                } else if (!eval_numeric_scalar_binary(
                               cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                {
                    return 0;
                }
            }
            tmp[r] = sum;
        }
        return make_vec_binary_result(cev, n, R, tmp, out);
    }

    if (op == WGSL_TOK_STAR &&
        a->kind == WGSL_VAL_VEC && b->kind == WGSL_VAL_MAT &&
        a->type && b->type &&
        a->type->kind == WGSL_TYPE_VEC && b->type->kind == WGSL_TYPE_MAT &&
        a->type->width == b->type->rows)
    {
        uint32_t C = b->type->width;
        uint32_t R = b->type->rows;
        WGSLValue tmp[4] = {{0}};
        for (uint32_t c = 0; c < C; c++) {
            WGSLValue sum = {0};
            for (uint32_t r = 0; r < R; r++) {
                WGSLValue av = a->u.agg.elems[r];
                WGSLValue bv = b->u.agg.elems[c * R + r];
                WGSLValue prod = {0};
                if (!eval_numeric_scalar_binary(
                        cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                    return 0;
                if (r == 0) {
                    sum = prod;
                } else if (!eval_numeric_scalar_binary(
                               cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                {
                    return 0;
                }
            }
            tmp[c] = sum;
        }
        return make_vec_binary_result(cev, n, C, tmp, out);
    }

    return 0;
}

static int eval_unary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *n,
    WGSLTokenKind op, WGSLValue inner, WGSLValue *out)
{
    switch (op) {
    case WGSL_TOK_MINUS:
        if (val_is_int(&inner)) {
            if (inner.type == cev->types->t_u32) {
                cev_error(cev, n, "unary minus is not defined for u32");
                return 0;
            }
            /* §6.2.3: negating the smallest representable value of a
             * signed type is a compile-time error.  AbstractInt is
             * 64-bit; concrete `i32` is 32-bit. */
            if (inner.type == cev->types->t_i32 &&
                inner.u.i == INT32_MIN)
            {
                cev_error(cev, n,
                    "signed integer overflow in negation: -(-2147483648) is not representable in i32");
                return 0;
            }
            if (inner.u.i == INT64_MIN) {
                cev_error(cev, n, "signed integer overflow in negation");
                return 0;
            }
            *out = inner; out->u.i = -inner.u.i; return 1;
        }
        if (val_is_float(&inner)) {
            *out = inner; out->u.f = -inner.u.f; return 1;
        }
        cev_error(cev, n, "unary minus requires numeric operand"); return 0;
    case WGSL_TOK_BANG:
        if (val_is_bool(&inner)) {
            out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
            out->u.b = !inner.u.b; return 1;
        }
        cev_error(cev, n, "logical not requires bool operand"); return 0;
    case WGSL_TOK_TILDE:
        if (val_is_int(&inner)) {
            int64_t r = ~inner.u.i;
            if (inner.type == cev->types->t_i32) r = (int32_t)r;
            else if (inner.type == cev->types->t_u32) r = (int64_t)(uint32_t)r;
            *out = inner; out->u.i = r; return 1;
        }
        cev_error(cev, n, "bitwise not requires integer operand"); return 0;
    default:
        cev_error(cev, n, "unsupported unary operator"); return 0;
    }
}

static int eval_unary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 1 || !n->children[0]) {
        cev_error(cev, n, "malformed unary expression");
        return 0;
    }
    WGSLValue inner = {0};
    if (!eval_expr(cev, n->children[0], &inner)) return 0;

    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
    if (inner.kind == WGSL_VAL_VEC) {
        WGSLValue tmp[4] = {{0}};
        uint32_t width = inner.u.agg.count;
        for (uint32_t i = 0; i < width; i++) {
            if (!eval_unary_scalar(
                    cev, n, op, inner.u.agg.elems[i], &tmp[i]))
            {
                return 0;
            }
        }
        return make_componentwise_result(cev, n, width, inner.type, tmp, out);
    }
    return eval_unary_scalar(cev, n, op, inner, out);
}

static int eval_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];

    /* §8.1 — `&&` and `||` short-circuit at const-eval: when the LHS
     * already determines the result, the RHS must not be evaluated.
     * This matters when the RHS would otherwise trigger a const-eval
     * diagnostic (`false && (5/0 == 0)` must not flag div-by-zero). */
    WGSLValue a = {0}, b = {0};
    if (!eval_expr(cev, n->children[0], &a)) return 0;
    if (val_is_bool(&a)) {
        if (op == WGSL_TOK_AMP_AMP && a.u.b == false) {
            out->kind = WGSL_VAL_BOOL;
            out->type = cev->types->t_bool;
            out->u.b  = false;
            return 1;
        }
        if (op == WGSL_TOK_PIPE_PIPE && a.u.b == true) {
            out->kind = WGSL_VAL_BOOL;
            out->type = cev->types->t_bool;
            out->u.b  = true;
            return 1;
        }
    }
    if (!eval_expr(cev, n->children[1], &b)) return 0;

    if (a.kind == WGSL_VAL_MAT || b.kind == WGSL_VAL_MAT) {
        return eval_matrix_binary(cev, n, op, &a, &b, out);
    }

    if (a.kind == WGSL_VAL_VEC || b.kind == WGSL_VAL_VEC) {
        return eval_binary_componentwise(cev, n, op, &a, &b, out);
    }

    /* Bool operands → handle separately (no numeric promotion). */
    if (val_is_bool(&a) && val_is_bool(&b)) {
        return eval_binary_bool(cev, n, op, &a, &b, out);
    }

    /* Mixed types → numeric promotion. */
    if (!val_is_numeric(&a) || !val_is_numeric(&b)) {
        cev_error(cev, n, "operator requires numeric or bool operands");
        return 0;
    }
    if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
        if (!val_is_int(&a) || !val_is_int(&b)) {
            cev_error(cev, n, "shift requires integer operands");
            return 0;
        }
        if (b.type != cev->types->t_u32 &&
            b.type != cev->types->t_abstract_int)
        {
            cev_error(cev, n, "shift amount must be u32");
            return 0;
        }
        return eval_binary_int(cev, n, op, &a, &b, out);
    }
    if (!promote_pair(cev, &a, &b, n)) return 0;

    /* Dispatch on the (now-shared) operand kind. */
    if (val_is_int(&a))   return eval_binary_int  (cev, n, op, &a, &b, out);
    if (val_is_float(&a)) return eval_binary_float(cev, n, op, &a, &b, out);

    cev_error(cev, n, "unhandled operand kind in binary op");
    return 0;
}

