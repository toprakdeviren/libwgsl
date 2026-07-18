/**
 * @file binary_ops.c — binary int/float/bool/matrix evaluation
 */
#include "internal/consteval_priv.h"


int eval_binary_int(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    /* One integer ALU op (arithmetic, bitwise, shift or compare); vectors reach
     * here per-component so they count as `width`. */
    cev_bill_op(cev, op, 0);
    int64_t ai = a->u.i, bi = b->u.i;
    int is_u = (a->type == cev->types->t_u32);
    int64_t r = 0;

    switch (op) {
    case WGSL_TOK_PLUS:   r = ai + bi; break;
    case WGSL_TOK_MINUS:  r = ai - bi; break;
    case WGSL_TOK_STAR:   r = ai * bi; break;
    case WGSL_TOK_SLASH:
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            /* §8.9: overflow is a shader-creation error for a const, but at
             * runtime the GPU wraps (INT32_MIN / -1 → INT32_MIN). */
            if (cev->simt.runtime) { r = INT32_MIN; break; }
            cev_error(cev, at, "signed integer overflow in division");
            return 0;
        }
        if (!int_div_mod(cev, at, ai, bi, is_u, 0, &r)) return 0;
        break;
    case WGSL_TOK_PERCENT:
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            if (cev->simt.runtime) { r = 0; break; }   /* wrap: remainder is 0 */
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
            /* Runtime: GPUs mask the shift amount to the bit width rather than
             * trapping; const context keeps the shader-creation error. */
            if (!cev->simt.runtime) {
                cev_error(cev, at,
                    "shift amount %lld is out of range for %s",
                    (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
                return 0;
            }
            bi = (int64_t)((uint64_t)bi & (uint64_t)(bit_width - 1));
        }
        uint64_t s = (uint64_t)bi;
        if (a->type == cev->types->t_u32) {
            uint64_t wr = (uint64_t)(uint32_t)ai << s;
            if (wr > UINT32_MAX) {
                if (!cev->simt.runtime) {
                    cev_error(cev, at, "left shift result out of range for u32");
                    return 0;
                }
                wr &= 0xFFFFFFFFu;   /* runtime wrap */
            }
            r = (int64_t)wr;
        } else {
            __int128 wr = (__int128)ai << s;
            __int128 lo = a->type == cev->types->t_i32
                ? (__int128)INT32_MIN : (__int128)INT64_MIN;
            __int128 hi = a->type == cev->types->t_i32
                ? (__int128)INT32_MAX : (__int128)INT64_MAX;
            if (wr < lo || wr > hi) {
                if (!cev->simt.runtime) {
                    cev_error(cev, at, "left shift result out of range for %s",
                        wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
                    return 0;
                }
                /* runtime wrap to the target width */
                r = a->type == cev->types->t_i32
                    ? (int64_t)(int32_t)((uint32_t)ai << (s & 31))
                    : (int64_t)((uint64_t)ai << (s & 63));
                break;
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
            /* Runtime masks the amount to the bit width (like <<); const errors. */
            if (!cev->simt.runtime) {
                cev_error(cev, at,
                    "shift amount %lld is out of range for %s",
                    (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
                return 0;
            }
            bi = (int64_t)((uint64_t)bi & (uint64_t)(bit_width - 1));
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

int round_float_value_to_type(
    WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out);

int eval_binary_float(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    /* Bill up front so float comparisons count as a flop too (was asymmetric:
     * int comparisons counted, float comparisons didn't). */
    cev_bill_op(cev, op, 1);
    if (cev->simt.runtime && a->type && a->type->kind == WGSL_TYPE_F16) cev->cost.flop_f16++;  /* 2× throughput */
    double af = a->u.f, bf = b->u.f, r = 0.0;
    switch (op) {
    case WGSL_TOK_PLUS:    r = af + bf; break;
    case WGSL_TOK_MINUS:   r = af - bf; break;
    case WGSL_TOK_STAR:    r = af * bf; break;
    case WGSL_TOK_SLASH:
        if (bf == 0.0 && !cev->simt.runtime) {
            cev_error(cev, at, "division by zero");
            return 0;
        }
        r = af / bf; break;          /* runtime: ±Inf (or NaN for 0/0) per IEEE-754 */
    case WGSL_TOK_PERCENT:
        if (bf == 0.0 && !cev->simt.runtime) {
            cev_error(cev, at, "modulo by zero");
            return 0;
        }
        /* WGSL §8.7.4: float % uses x - y * trunc(x / y). */
        r = af - bf * trunc(af / bf); break;   /* runtime: NaN when bf == 0 */
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
    /* Arithmetic ops (+ - * / %) reach here; comparisons returned above.  The
     * flop was billed at the top by cev_bill_op. */

    /* §8.7 / §15.7.3 — finite-math assumption.  Inputs are finite at
     * this point (the consteval doesn't import non-finite literals);
     * any non-finite result therefore comes from a domain violation
     * (e.g., `inf - inf` derived through chained ops, or AbstractFloat
     * overflow at ~1e308 from `1e200 * 1e200`). */
    double rounded = 0.0;
    if (!round_float_value_to_type(cev, a->type, r, &rounded)) {
        if (cev->simt.runtime) {
            /* runtime IEEE-754: let ±Inf / NaN propagate at the target
             * width so the interpreter can trace and flag them. */
            out->kind = WGSL_VAL_FLOAT;
            out->type = a->type;
            out->u.f  = (a->type == cev->types->t_f32) ? (double)(float)r : r;
            return 1;
        }
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

int eval_binary_bool(
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

int eval_binary_componentwise(
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

int eval_numeric_scalar_binary(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, n)) return 0;
    if (val_is_int(a)) return eval_binary_int(cev, n, op, a, b, out);
    if (val_is_float(a)) return eval_binary_float(cev, n, op, a, b, out);
    cev_error(cev, n, "operator requires numeric operands");
    return 0;
}

int make_matrix_binary_result(
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

int make_vec_binary_result(
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

int eval_matrix_binary(
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

