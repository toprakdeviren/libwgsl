/**
 * @file numeric_geom.c — length, distance, dot, cross, normalize
 */
#include "internal/consteval_priv.h"


int eval_call_length_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "length")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'length' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    WGSLValue first = value_component(&arg, 0);
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    uint32_t width = value_width(&arg);
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = value_component(&arg, i);
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        sum += c.u.f * c.u.f;
    }
    double r = width == 1 ? fabs(first.u.f) : sqrt(sum);
    /* width>1: (width mul + width-1 add) for |v|² + a sqrt (SFU ≈ 4). */
    if (cev->simt.runtime) cev->cost.flop_count += width == 1 ? 1 : (2 * width - 1 + 4);
    return finish_float_builtin_result(cev, call, nm, len, first.type, r, out) ? 1 : -1;
}

int eval_call_distance_dot_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_distance = name_eq(nm, len, "distance");
    int is_dot = name_eq(nm, len, "dot");
    if (!is_distance && !is_dot) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin '%.*s' expects 2 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (is_distance && vals[0].kind != WGSL_VAL_VEC &&
        vals[1].kind != WGSL_VAL_VEC)
    {
        WGSLValue a = vals[0];
        WGSLValue b = vals[1];
        if (!promote_pair(cev, &a, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        if (cev->simt.runtime) cev->cost.flop_count += 1;
        return finish_float_builtin_result(
            cev, call, nm, len, a.type, fabs(a.u.f - b.u.f), out) ? 1 : -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != vals[1].u.agg.count)
    {
        cev_error(cev, call,
            "builtin '%.*s' requires same-width vector arguments",
            (int)len, nm);
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call)) return -1;
    if (is_distance) {
        if (!require_float_scalar(cev, call, nm, len, &a0) ||
            !require_float_scalar(cev, call, nm, len, &b0))
        {
            return -1;
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call) ||
                !promote_pair(cev, &a0, &b, call) ||
                !require_float_scalar(cev, call, nm, len, &a) ||
                !require_float_scalar(cev, call, nm, len, &b))
            {
                return -1;
            }
            double d = a.u.f - b.u.f;
            sum += d * d;
        }
        /* per comp: sub + mul (=2) · (count-1) adds · sqrt (SFU ≈ 4). */
        if (cev->simt.runtime) cev->cost.flop_count += 3 * vals[0].u.agg.count + 3;
        return finish_float_builtin_result(
            cev, call, nm, len, a0.type, sqrt(sum), out) ? 1 : -1;
    }

    if (a0.kind == WGSL_VAL_INT && b0.kind == WGSL_VAL_INT) {
        int64_t sum = 0;
        int is_u = a0.type == cev->types->t_u32;
        int is_abstract = a0.type == cev->types->t_abstract_int;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call))
            {
                return -1;
            }
            if (is_u) {
                uint64_t prod = (uint64_t)(uint32_t)a.u.i * (uint64_t)(uint32_t)b.u.i;
                sum = (int64_t)(uint32_t)((uint32_t)sum + (uint32_t)prod);
            } else if (is_abstract) {
                int64_t prod = 0;
                if (!checked_abstract_int_result(
                        cev, call, (__int128)a.u.i * (__int128)b.u.i,
                        "dot", &prod) ||
                    !checked_abstract_int_result(
                        cev, call, (__int128)sum + (__int128)prod,
                        "dot", &sum))
                {
                    return -1;
                }
            } else {
                sum += a.u.i * b.u.i;
                if (a0.type == cev->types->t_i32) sum = (int32_t)sum;
            }
        }
        out->kind = WGSL_VAL_INT;
        out->type = a0.type;
        out->u.i = sum;
        if (cev->simt.runtime) cev->cost.flop_count += 2 * vals[0].u.agg.count - 1;
        return 1;
    }

    if (!require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        sum += a.u.f * b.u.f;
    }
    if (cev->simt.runtime) cev->cost.flop_count += 2 * vals[0].u.agg.count - 1;
    return finish_float_builtin_result(cev, call, nm, len, a0.type, sum, out) ? 1 : -1;
}

int eval_call_cross_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "cross")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'cross' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != 3 || vals[1].u.agg.count != 3)
    {
        cev_error(cev, call, "builtin 'cross' requires vec3 arguments");
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call) ||
        !require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double av[3], bv[3];
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        av[i] = a.u.f;
        bv[i] = b.u.f;
    }
    WGSLValue tmp[3] = {{0}};
    static const uint8_t idx[3][4] = {
        { 1, 2, 2, 1 },
        { 2, 0, 0, 2 },
        { 0, 1, 1, 0 },
    };
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue lhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][0]],
        };
        WGSLValue lhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][1]],
        };
        WGSLValue rhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][2]],
        };
        WGSLValue rhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][3]],
        };
        WGSLValue lhs = {0}, rhs = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &lhs_a, &lhs_b, &lhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_STAR, &rhs_a, &rhs_b, &rhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_MINUS, &lhs, &rhs, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, 3, a0.type, tmp, out) ? 1 : -1;
}

int eval_call_normalize_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "normalize")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'normalize' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_VEC || arg.u.agg.count == 0) {
        cev_error(cev, call, "builtin 'normalize' requires a vector argument");
        return -1;
    }
    WGSLValue first = arg.u.agg.elems[0];
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    WGSLValue sumv = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = 0.0,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        WGSLValue prod = {0};
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &cv, &cv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sumv, &prod, &next))
        {
            return -1;
        }
        sumv = next;
    }
    double lenv = 0.0;
    if (!round_float_value_to_type(cev, first.type, sqrt(sumv.u.f), &lenv)) {
        cev_error(cev, call,
            "builtin 'normalize' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)");
        return -1;
    }
    if (lenv == 0.0) {
        cev_error(cev, call,
            "builtin 'normalize' domain error: argument must not be the zero vector");
        return -1;
    }
    /* The |v|² muls/adds and the per-component divisions self-count through
     * eval_binary_float; bill the length's sqrt here (SFU ≈ 4). */
    if (cev->simt.runtime) cev->cost.flop_count += 4;
    WGSLValue tmp[4] = {{0}};
    WGSLValue denom = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = lenv,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        if (!eval_binary_float(cev, call, WGSL_TOK_SLASH, &cv, &denom, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(
        cev, call, arg.u.agg.count, arg.type, tmp, out) ? 1 : -1;
}

int fold_float_dot(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    const WGSLValue *a, const WGSLValue *b,
    uint32_t width, WGSLTypeInfo **out_type, double *out_dot)
{
    WGSLValue at = a->u.agg.elems[0];
    WGSLValue bt = b->u.agg.elems[0];
    if (!promote_pair(cev, &at, &bt, call) ||
        !require_float_scalar(cev, call, nm, len, &at) ||
        !require_float_scalar(cev, call, nm, len, &bt))
    {
        return 0;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue ai = a->u.agg.elems[i];
        WGSLValue bi = b->u.agg.elems[i];
        if (!promote_pair(cev, &ai, &bi, call) ||
            !promote_pair(cev, &at, &ai, call) ||
            !promote_pair(cev, &at, &bi, call) ||
            !require_float_scalar(cev, call, nm, len, &ai) ||
            !require_float_scalar(cev, call, nm, len, &bi))
        {
            return 0;
        }
        WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = ai.u.f };
        WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = bi.u.f };
        WGSLValue prod = {0};
        WGSLValue sv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = sum };
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &av, &bv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sv, &prod, &next)) {
            return 0;
        }
        sum = next.u.f;
    }
    if (!isfinite(sum)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    *out_type = at.type;
    *out_dot = sum;
    return 1;
}

int fold_float_inherited_binary(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *type,
    WGSLTokenKind op, double a, double b, double *out)
{
    WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = a };
    WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = b };
    WGSLValue rv = {0};
    if (!eval_binary_float(cev, call, op, &av, &bv, &rv)) return 0;
    *out = rv.u.f;
    return 1;
}

