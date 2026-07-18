/**
 * @file domain_checks.c — ldexp/mix/clamp/smoothstep checks
 */
#include "internal/exprs_priv.h"

void check_ldexp_call(
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

void check_mix_call(
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

uint32_t const_value_width_tc(const WGSLValue *v) {
    return v && v->kind == WGSL_VAL_VEC ? v->u.agg.count : 1;
}

const WGSLValue *const_value_component_tc(const WGSLValue *v, uint32_t i) {
    if (!v) return NULL;
    if (v->kind == WGSL_VAL_VEC) {
        return i < v->u.agg.count ? &v->u.agg.elems[i] : NULL;
    }
    return i == 0 ? v : NULL;
}

int const_value_gt_tc(
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

int const_value_eq_tc(const WGSLValue *a, const WGSLValue *b, int *out)
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

void check_clamp_low_high_call(
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

void check_smoothstep_edges_call(
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

