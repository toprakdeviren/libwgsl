/**
 * @file builtin_helpers.c — const-expression evaluator module.
 */
#include "internal/consteval_priv.h"

/* Builtin call folding (§15.7.7 domain checks). */

int node_ident_text(
    const WGSLConstEvaluator *cev, const WGSLNode *n,
    const char **out, uint32_t *len)
{
    if (!n || n->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = wgsl_node_name_span(n).offset;
    uint32_t l   = wgsl_node_name_span(n).length;
    if (!cev->src || off + l > cev->src->length) return 0;
    *out = cev->src->bytes + off;
    *len = l;
    return 1;
}

int name_eq(const char *nm, uint32_t len, const char *want) {
    size_t wl = strlen(want);
    return len == wl && memcmp(nm, want, wl) == 0;
}

/* Effective-FLOP weight per component for a builtin — used ONLY by the runtime
 * cost model (every caller is gated on cev->simt.runtime).  A transcendental is not
 * one FLOP: on real GPUs it runs on the Special Function Unit at a fraction of
 * FMA throughput, so we bill an approximate FLOP-equivalent — a single native
 * SFU op (rsqrt/exp2/log2/sin class) ≈ 4×, and functions that compose several
 * SFU ops or a polynomial (pow, inverse-trig, hyperbolics) ≈ 8×.  Cheap ALU /
 * shape ops (abs/floor/ceil/min/max/step/…) stay at the default 1.  These are
 * heuristic machine-balance weights, NOT IEEE operation counts; they make the
 * roofline "compute- vs memory-bound" verdict honest for activation-heavy ML
 * kernels (softmax / gelu / attention).  Retune the two lists to taste. */
uint32_t builtin_flop_weight(const char *nm, uint32_t len) {
    if (name_eq(nm, len, "sqrt")   || name_eq(nm, len, "inverseSqrt") ||
        name_eq(nm, len, "exp")    || name_eq(nm, len, "exp2") ||
        name_eq(nm, len, "log")    || name_eq(nm, len, "log2") ||
        name_eq(nm, len, "sin")    || name_eq(nm, len, "cos"))
    {
        return 4;   /* one native SFU instruction */
    }
    if (name_eq(nm, len, "tan")    || name_eq(nm, len, "pow") ||
        name_eq(nm, len, "atan2")  || name_eq(nm, len, "atan") ||
        name_eq(nm, len, "asin")   || name_eq(nm, len, "acos") ||
        name_eq(nm, len, "sinh")   || name_eq(nm, len, "cosh") ||
        name_eq(nm, len, "tanh")   || name_eq(nm, len, "asinh") ||
        name_eq(nm, len, "acosh")  || name_eq(nm, len, "atanh"))
    {
        return 8;   /* composed SFU ops / polynomial approximation */
    }
    return 1;       /* cheap ALU / shape op */
}

double round_even(double x) {
    double lo = floor(x);
    double f = x - lo;
    if (f < 0.5) return lo;
    if (f > 0.5) return lo + 1.0;
    return fmod(lo, 2.0) == 0.0 ? lo : lo + 1.0;
}

int round_float_value_to_type(
    WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out)
{
    (void)cev;
    if (!isfinite(x)) return 0;
    if (type == cev->types->t_f32) {
        float xf = (float)x;
        if (!isfinite((double)xf)) return 0;
        *out = (double)xf;
        return 1;
    }
    if (type == cev->types->t_f16) {
        double ax = fabs(x);
        if (ax == 0.0) {
            *out = signbit(x) ? -0.0 : 0.0;
            return 1;
        }
        /* binary16 overflows past the midpoint between max finite
         * (65504) and +inf; values at/below that round to max finite. */
        if (ax > 65520.0) return 0;
        double rounded = 0.0;
        if (ax < ldexp(1.0, -14)) {
            double step = ldexp(1.0, -24);
            rounded = round_even(ax / step) * step;
        } else {
            int e = (int)floor(log2(ax));
            double step = ldexp(1.0, e - 10);
            rounded = round_even(ax / step) * step;
            if (rounded > 65504.0) rounded = 65504.0;
        }
        *out = signbit(x) ? -rounded : rounded;
        return 1;
    }
    *out = x;
    return 1;
}

int finish_float_builtin_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLTypeInfo *result_type, double r, WGSLValue *out)
{
    if (!isfinite(r)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    if (result_type == cev->types->t_f32 &&
        (r < -FLT_MAX || r > FLT_MAX))
    {
        cev_error(cev, call, "builtin '%.*s' result out of range for f32",
                  (int)len, nm);
        return 0;
    }
    if (result_type == cev->types->t_f16 &&
        (r < -65504.0 || r > 65504.0))
    {
        cev_error(cev, call, "builtin '%.*s' result out of range for f16",
                  (int)len, nm);
        return 0;
    }

    out->kind = WGSL_VAL_FLOAT;
    out->type = result_type;
    out->u.f = r;
    return 1;
}

int finish_float_builtin_result_relaxed_range(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLTypeInfo *result_type, double r, WGSLValue *out)
{
    if (!isfinite(r)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = result_type;
    out->u.f = r;
    return 1;
}

int require_float_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *v)
{
    if (v->kind == WGSL_VAL_INT && v->type == cev->types->t_abstract_int) {
        return wgsl_consteval_materialize(
            cev, v, cev->types->t_abstract_float, call);
    }
    if (v->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, call,
            "'%.*s' requires a floating-point argument", (int)len, nm);
        return 0;
    }
    return 1;
}

int promote_three(
    WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b, WGSLValue *c,
    const WGSLNode *at)
{
    return promote_pair(cev, a, b, at) &&
           promote_pair(cev, a, c, at) &&
           promote_pair(cev, b, c, at);
}

uint32_t value_width(const WGSLValue *v) {
    return v->kind == WGSL_VAL_VEC ? v->u.agg.count : 1;
}

WGSLValue value_component(const WGSLValue *v, uint32_t i) {
    return v->kind == WGSL_VAL_VEC ? v->u.agg.elems[i] : *v;
}

int componentwise_shape(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *vals, int argc, int scalar_broadcast_arg,
    uint32_t *out_width, WGSLTypeInfo **out_vec_type)
{
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    for (int i = 0; i < argc; i++) {
        if (vals[i].kind != WGSL_VAL_VEC) continue;
        if (!vals[i].type || vals[i].type->kind != WGSL_TYPE_VEC) {
            cev_error(cev, call,
                "builtin '%.*s' requires scalar/vector numeric arguments",
                (int)len, nm);
            return 0;
        }
        if (!vec_type) {
            vec_type = vals[i].type;
            width = vals[i].u.agg.count;
        } else if (vals[i].u.agg.count != width) {
            cev_error(cev, call,
                "builtin '%.*s' vector width mismatch in const-eval",
                (int)len, nm);
            return 0;
        }
    }
    if (vec_type) {
        for (int i = 0; i < argc; i++) {
            if (vals[i].kind == WGSL_VAL_VEC) continue;
            if (i == scalar_broadcast_arg) continue;
            cev_error(cev, call,
                "builtin '%.*s' shape mismatch in const-eval",
                (int)len, nm);
            return 0;
        }
    }
    *out_width = width;
    *out_vec_type = vec_type;
    return 1;
}

int make_componentwise_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *vec_type,
    WGSLValue *elems, WGSLValue *out)
{
    if (!vec_type) {
        *out = elems[0];
        return 1;
    }
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector builtin");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

int make_vector_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *elem_type,
    WGSLValue *elems, WGSLValue *out)
{
    if (width < 2 || width > 4 || !elem_type) return 0;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector builtin");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem_type);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

int require_same_width_vectors(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *vals, uint32_t argc, uint32_t *out_width)
{
    if (argc == 0 || vals[0].kind != WGSL_VAL_VEC) {
        cev_error(cev, call,
            "builtin '%.*s' requires vector arguments", (int)len, nm);
        return 0;
    }
    uint32_t width = vals[0].u.agg.count;
    for (uint32_t i = 1; i < argc; i++) {
        if (vals[i].kind != WGSL_VAL_VEC || vals[i].u.agg.count != width) {
            cev_error(cev, call,
                "builtin '%.*s' requires same-width vector arguments",
                (int)len, nm);
            return 0;
        }
    }
    *out_width = width;
    return 1;
}

int vector_width_from_name(const char *nm, uint32_t len) {
    if (len == 4 && memcmp(nm, "vec", 3) == 0 &&
        nm[3] >= '2' && nm[3] <= '4')
    {
        return nm[3] - '0';
    }
    return 0;
}

int matrix_dims_from_name(
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

WGSLSymbol *callee_head_symbol(const WGSLNode *callee) {
    if (!callee) return NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        return wgsl_node_resolved_symbol((WGSLNode *)callee);
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        callee->child_count >= 1 && callee->children[0])
    {
        return wgsl_node_resolved_symbol(callee->children[0]);
    }
    return NULL;
}

uint32_t struct_member_count(const WGSLNode *sd) {
    uint32_t count = 0;
    if (!sd) return 0;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (m && m->kind == WGSL_NODE_DECL_STRUCT_MEMBER) count++;
    }
    return count;
}

int make_zero_value(
    WGSLConstEvaluator *cev, WGSLTypeInfo *target, const WGSLNode *at,
    WGSLValue *out)
{
    if (!target) return 0;
    memset(out, 0, sizeof *out);
    out->type = target;
    switch ((WGSLTypeKind)target->kind) {
    case WGSL_TYPE_BOOL:
        out->kind = WGSL_VAL_BOOL;
        out->u.b = false;
        return 1;
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
        out->kind = WGSL_VAL_INT;
        out->u.i = 0;
        return 1;
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
        out->kind = WGSL_VAL_FLOAT;
        out->u.f = 0.0;
        return 1;
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
    case WGSL_TYPE_ARRAY: {
        uint32_t count = 0;
        if (target->kind == WGSL_TYPE_VEC) {
            count = target->width;
            out->kind = WGSL_VAL_VEC;
        } else if (target->kind == WGSL_TYPE_MAT) {
            count = (uint32_t)target->width * (uint32_t)target->rows;
            out->kind = WGSL_VAL_MAT;
        } else {
            count = target->array_len;
            out->kind = WGSL_VAL_ARRAY;
            if (count == 0) {
                cev_error(cev, at, "runtime-sized array is not constructible");
                return 0;
            }
        }
        WGSLValue *elems = NULL;
        if (count > 0) {
            elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * count);
            if (!elems) {
                cev_error(cev, at, "out of memory while folding constructor");
                return 0;
            }
        }
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        for (uint32_t i = 0; i < count; i++) {
            if (!make_zero_value(cev, elem, at, &elems[i])) return 0;
        }
        out->u.agg.count = count;
        out->u.agg.elems = elems;
        return 1;
    }
    case WGSL_TYPE_STRUCT: {
        WGSLNode *sd = (WGSLNode *)target->ref;
        if (sd && (sd->flags & WGSL_FLAG_CONST_EVALING)) {
            cev_error(cev, at, "cyclic dependency detected in struct constructor");
            return 0;
        }
        if (sd) sd->flags |= WGSL_FLAG_CONST_EVALING;
        uint32_t count = struct_member_count(sd);
        WGSLValue *elems = NULL;
        if (count > 0) {
            elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * count);
            if (!elems) {
                if (sd) sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                cev_error(cev, at, "out of memory while folding struct constructor");
                return 0;
            }
        }
        uint32_t idx = 0;
        for (uint32_t i = 0; sd && i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = wgsl_member_attr_count(m);
            if (attrs >= m->child_count) {
                sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                return 0;
            }
            WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
            if (!mt || !make_zero_value(cev, mt, at, &elems[idx])) {
                sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                return 0;
            }
            idx++;
        }
        if (sd) sd->flags &= ~WGSL_FLAG_CONST_EVALING;
        out->kind = WGSL_VAL_STRUCT;
        out->type = target;
        out->u.agg.count = count;
        out->u.agg.elems = elems;
        return 1;
    }
    default:
        cev_error(cev, at, "type is not constructible");
        return 0;
    }
}

WGSLTypeInfo *constructor_target_type(
    const WGSLConstEvaluator *cev, const WGSLNode *callee)
{
    if (!callee) return NULL;
    WGSLSymbol *sym = callee_head_symbol(callee);
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        if (sym && (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                    sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                    sym->kind == WGSL_SYM_TYPE_ALIAS ||
                    sym->kind == WGSL_SYM_STRUCT))
        {
            return type_from_typespec(cev, callee);
        }
        return NULL;
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        return type_from_typespec(cev, callee);
    }
    return NULL;
}

int convert_scalar_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target,
    const WGSLNode *at);

int convert_constructor_component(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *elem,
    const WGSLNode *at)
{
    if (!v || !elem) return 0;
    if (v->kind == WGSL_VAL_BOOL ||
        v->kind == WGSL_VAL_INT ||
        v->kind == WGSL_VAL_FLOAT)
    {
        return convert_scalar_constructor_value(cev, v, elem, at);
    }
    if (v->kind != WGSL_VAL_INT && v->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, at, "constructor requires scalar components");
        return 0;
    }
    return wgsl_consteval_materialize(cev, v, elem, at);
}

int append_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t width,
    uint32_t *count, WGSLTypeInfo *elem, WGSLValue *src, const WGSLNode *at)
{
    if (src->kind == WGSL_VAL_VEC) {
        for (uint32_t i = 0; i < src->u.agg.count; i++) {
            if (*count >= width) {
                cev_error(cev, at, "too many vector constructor components");
                return 0;
            }
            WGSLValue c = src->u.agg.elems[i];
            if (!convert_constructor_component(cev, &c, elem, at)) return 0;
            dst[(*count)++] = c;
        }
        return 1;
    }
    if (*count >= width) {
        cev_error(cev, at, "too many vector constructor components");
        return 0;
    }
    WGSLValue c = *src;
    if (!convert_constructor_component(cev, &c, elem, at)) return 0;
    dst[(*count)++] = c;
    return 1;
}

int common_constructor_type(
    WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count,
    const WGSLNode *at, WGSLTypeInfo **out)
{
    if (count == 0) {
        *out = cev->types->t_abstract_int;
        return 1;
    }
    WGSLTypeInfo *common = vals[0].type;
    if (!common) return 0;
    for (uint32_t i = 1; i < count; i++) {
        WGSLTypeInfo *t = vals[i].type;
        if (!t) return 0;
        if (common == t) continue;
        int ab = wgsl_type_conversion_rank(common, t);
        int ba = wgsl_type_conversion_rank(t, common);
        if (ab >= 0 && (ba < 0 || ab <= ba)) {
            common = t;
        } else if (ba >= 0) {
            /* Keep the existing common type. */
        } else {
            cev_error(cev, at, "constructor arguments have incompatible types");
            return 0;
        }
    }
    *out = common;
    return 1;
}

int materialize_constructor_values(
    WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count,
    WGSLTypeInfo *target, const WGSLNode *at)
{
    for (uint32_t i = 0; i < count; i++) {
        if (!wgsl_consteval_materialize(cev, &vals[i], target, at))
            return 0;
    }
    return 1;
}

