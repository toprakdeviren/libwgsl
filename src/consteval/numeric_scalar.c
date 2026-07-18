/**
 * @file numeric_scalar.c — scalar min/max/clamp/ldexp and binary/ternary numeric builtins
 */
#include "internal/consteval_priv.h"


int fold_minmax_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call)) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        *out = *a;
        if (a->type == cev->types->t_u32) {
            uint32_t au = (uint32_t)a->u.i, bu = (uint32_t)b->u.i;
            uint32_t r = name_eq(nm, len, "min")
                ? (au < bu ? au : bu)
                : (au > bu ? au : bu);
            out->u.i = (int64_t)r;
        } else {
            out->u.i = name_eq(nm, len, "min")
                ? (a->u.i < b->u.i ? a->u.i : b->u.i)
                : (a->u.i > b->u.i ? a->u.i : b->u.i);
        }
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double r = name_eq(nm, len, "min")
        ? (a->u.f < b->u.f ? a->u.f : b->u.f)
        : (a->u.f > b->u.f ? a->u.f : b->u.f);
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

int fold_float_binary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double x = a->u.f, y = b->u.f, r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "pow")) {
        if (x < 0.0) domain = "base must be >= 0";
        else if (x == 0.0 && y <= 0.0) domain = "zero base requires a positive exponent";
        r = pow(x, y);
    } else if (name_eq(nm, len, "atan2")) {
        r = atan2(x, y);
    } else if (name_eq(nm, len, "step")) {
        r = y < x ? 0.0 : 1.0;
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

int eval_call_binary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "min") || name_eq(nm, len, "max") ||
        name_eq(nm, len, "pow") || name_eq(nm, len, "atan2") ||
        name_eq(nm, len, "step");
    if (!supported) return 0;
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
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        int ok = (name_eq(nm, len, "min") || name_eq(nm, len, "max"))
            ? fold_minmax_scalar(cev, call, nm, len, &a, &b, &tmp[i])
            : fold_float_binary_scalar(cev, call, nm, len, &a, &b, &tmp[i]);
        if (!ok) return -1;
    }
    /* min/max/step → 1 · pow/atan2 → SFU-weighted (see builtin_flop_weight). */
    if (cev->simt.runtime) {
        uint32_t w = builtin_flop_weight(nm, len);
        cev->cost.flop_count += width * w;
        if (w > 1) cev->cost.op_bucket[WGSL_OPB_FTRANS] += width;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

int fold_clamp_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *x, WGSLValue *lo, WGSLValue *hi, WGSLValue *out)
{
    if (!promote_three(cev, x, lo, hi, call)) return 0;
    if (x->kind == WGSL_VAL_INT && lo->kind == WGSL_VAL_INT &&
        hi->kind == WGSL_VAL_INT)
    {
        *out = *x;
        if (x->type == cev->types->t_u32) {
            uint32_t xv = (uint32_t)x->u.i;
            uint32_t lv = (uint32_t)lo->u.i;
            uint32_t hv = (uint32_t)hi->u.i;
            if (lv > hv) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            if (xv < lv) xv = lv;
            if (xv > hv) xv = hv;
            out->u.i = (int64_t)xv;
        } else {
            if (lo->u.i > hi->u.i) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            int64_t r = x->u.i < lo->u.i ? lo->u.i : x->u.i;
            if (r > hi->u.i) r = hi->u.i;
            out->u.i = r;
        }
        (void)nm; (void)len;
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, x) ||
        !require_float_scalar(cev, call, nm, len, lo) ||
        !require_float_scalar(cev, call, nm, len, hi))
    {
        return 0;
    }
    if (lo->u.f > hi->u.f) {
        cev_error(cev, call,
            "builtin 'clamp' domain error: low must be <= high");
        return 0;
    }
    double r = x->u.f < lo->u.f ? lo->u.f : x->u.f;
    if (r > hi->u.f) r = hi->u.f;
    return finish_float_builtin_result(cev, call, nm, len, x->type, r, out);
}

int fold_float_ternary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *a, WGSLValue *b, WGSLValue *c, WGSLValue *out)
{
    if (!promote_three(cev, a, b, c, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b) ||
        !require_float_scalar(cev, call, nm, len, c))
    {
        return 0;
    }
    double r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "mix")) {
        double one_minus_c = 0.0;
        double lhs = 0.0;
        double rhs = 0.0;
        if (!round_float_value_to_type(cev, a->type, 1.0 - c->u.f,
                                       &one_minus_c) ||
            !round_float_value_to_type(cev, a->type, a->u.f * one_minus_c,
                                       &lhs) ||
            !round_float_value_to_type(cev, a->type, b->u.f * c->u.f,
                                       &rhs) ||
            !round_float_value_to_type(cev, a->type, lhs + rhs, &r))
        {
            cev_error(cev, call,
                "builtin '%.*s' produced a non-finite result "
                "(spec §15.7.3 finite-math assumption)",
                (int)len, nm);
            return 0;
        }
    } else if (name_eq(nm, len, "fma")) {
        r = fma(a->u.f, b->u.f, c->u.f);
    } else if (name_eq(nm, len, "smoothstep")) {
        if (a->u.f == b->u.f) {
            domain = "edge0 and edge1 must differ";
        } else {
            double t = (c->u.f - a->u.f) / (b->u.f - a->u.f);
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            r = t * t * (3.0 - 2.0 * t);
        }
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

int eval_call_ternary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "clamp") || name_eq(nm, len, "mix") ||
        name_eq(nm, len, "fma") || name_eq(nm, len, "smoothstep");
    if (!supported) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin '%.*s' expects 3 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    int broadcast = name_eq(nm, len, "mix") ? 2 : -1;
    if (!componentwise_shape(cev, call, nm, len, vals, 3, broadcast,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        WGSLValue c = value_component(&vals[2], i);
        int ok = name_eq(nm, len, "clamp")
            ? fold_clamp_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i])
            : fold_float_ternary_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i]);
        if (!ok) return -1;
    }
    if (cev->simt.runtime) {
        if (name_eq(nm, len, "fma") || name_eq(nm, len, "clamp")) cev->cost.flop_count += 2 * width;
        else if (name_eq(nm, len, "mix") || name_eq(nm, len, "smoothstep")) cev->cost.flop_count += 3 * width;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

int eval_call_ldexp_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "ldexp")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'ldexp' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue x = value_component(&vals[0], i);
        WGSLValue e = value_component(&vals[1], i);
        if (!require_float_scalar(cev, call, nm, len, &x)) return -1;
        if (e.kind != WGSL_VAL_INT) {
            cev_error(cev, call, "'ldexp' requires an integer exponent");
            return -1;
        }
        if (e.type == cev->types->t_u32) {
            cev_error(cev, call, "'ldexp' requires a signed integer exponent");
            return -1;
        }
        if (x.type != cev->types->t_abstract_float &&
            e.type == cev->types->t_abstract_int &&
            !wgsl_consteval_materialize(cev, &e, cev->types->t_i32, call))
        {
            return -1;
        }
        int bias = x.type == cev->types->t_f16 ? 15 :
                   x.type == cev->types->t_f32 ? 127 : 1023;
        if (e.u.i > bias + 1) {
            cev_error(cev, call,
                "builtin 'ldexp' domain error: exponent too large");
            return -1;
        }
        double r = e.u.i < -1075 ? 0.0 : ldexp(x.u.f, (int)e.u.i);
        if (!finish_float_builtin_result(cev, call, nm, len, x.type, r, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

