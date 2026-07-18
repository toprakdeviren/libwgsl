/**
 * @file bitcast_pack.c — const-expression evaluator module.
 */
#include "internal/consteval_priv.h"

double f16_bits_to_double(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp = (uint32_t)((h >> 10) & 0x1fu);
    uint32_t frac = (uint32_t)(h & 0x03ffu);
    double v = 0.0;
    if (exp == 0) {
        v = frac ? ldexp((double)frac, -24) : 0.0;
    } else if (exp == 31) {
        v = frac ? NAN : INFINITY;
    } else {
        v = ldexp(1.0 + (double)frac / 1024.0, (int)exp - 15);
    }
    return sign ? -v : v;
}

uint16_t double_to_f16_bits(double x) {
    uint16_t sign = signbit(x) ? 0x8000u : 0u;
    double ax = fabs(x);
    if (ax == 0.0) return sign;
    if (!isfinite(ax)) return (uint16_t)(sign | 0x7c00u);

    if (ax < ldexp(1.0, -14)) {
        double step = ldexp(1.0, -24);
        uint32_t mant = (uint32_t)round_even(ax / step);
        if (mant > 0x3ffu) mant = 0x400u;
        if (mant == 0x400u) return (uint16_t)(sign | 0x0400u);
        return (uint16_t)(sign | mant);
    }

    int e = (int)floor(log2(ax));
    double base = ldexp(1.0, e);
    uint32_t mant = (uint32_t)round_even((ax / base - 1.0) * 1024.0);
    if (mant == 1024u) {
        mant = 0;
        e += 1;
    }
    int exp = e + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        double step = ldexp(1.0, -24);
        uint32_t sm = (uint32_t)round_even(ax / step);
        if (sm > 0x3ffu) sm = 0x3ffu;
        return (uint16_t)(sign | sm);
    }
    return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)(mant & 0x3ffu));
}

int value_to_u32_bits(
    WGSLConstEvaluator *cev, WGSLNode *at, const WGSLValue *v, uint32_t *bits)
{
    if (!v || !v->type || !bits) return 0;
    if (v->kind == WGSL_VAL_INT) {
        if (v->type != cev->types->t_i32 && v->type != cev->types->t_u32) {
            cev_error(cev, at, "bitcast const-eval requires concrete integer source values");
            return 0;
        }
        *bits = (uint32_t)v->u.i;
        return 1;
    }
    if (v->kind == WGSL_VAL_FLOAT) {
        if (v->type == cev->types->t_f32) {
            float f = (float)v->u.f;
            memcpy(bits, &f, sizeof f);
            return 1;
        }
        cev_error(cev, at, "bitcast const-eval requires f32 float source values");
        return 0;
    }
    cev_error(cev, at, "bitcast const-eval requires numeric source values");
    return 0;
}

int make_bitcast_component(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    uint32_t bits, WGSLValue *out)
{
    if (target == cev->types->t_i32) {
        out->kind = WGSL_VAL_INT;
        out->type = target;
        out->u.i = (int32_t)bits;
        return 1;
    }
    if (target == cev->types->t_u32) {
        out->kind = WGSL_VAL_INT;
        out->type = target;
        out->u.i = (int64_t)(uint32_t)bits;
        return 1;
    }
    if (target == cev->types->t_f32) {
        float f = 0.0f;
        memcpy(&f, &bits, sizeof f);
        if (!isfinite((double)f)) {
            cev_error(cev, call,
                "builtin 'bitcast' produced a non-finite f32 value");
            return 0;
        }
        out->kind = WGSL_VAL_FLOAT;
        out->type = target;
        out->u.f = (double)f;
        return 1;
    }
    if (target == cev->types->t_f16) {
        double f = f16_bits_to_double((uint16_t)bits);
        if (!isfinite(f)) {
            cev_error(cev, call,
                "builtin 'bitcast' produced a non-finite f16 value");
            return 0;
        }
        out->kind = WGSL_VAL_FLOAT;
        out->type = target;
        out->u.f = f;
        return 1;
    }
    cev_error(cev, call, "bitcast<T> target is not a concrete numeric scalar");
    return 0;
}

int eval_call_bitcast_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "bitcast")) return 0;
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "bitcast<T> expects exactly one argument");
        return -1;
    }
    WGSLNode *callee = call->children[0];
    if (!callee || callee->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT ||
        callee->child_count < 2)
    {
        cev_error(cev, call, "bitcast requires an explicit target type");
        return -1;
    }
    WGSLTypeInfo *target = type_from_typespec(cev, callee->children[1]);
    if (!target) {
        cev_error(cev, call, "bitcast target type could not be resolved");
        return -1;
    }

    WGSLValue src = {0};
    if (!eval_expr(cev, call->children[1], &src)) return -1;
    uint32_t src_words[4] = {0};
    uint32_t src_count = 0;
    if (src.kind == WGSL_VAL_VEC) {
        if (src.u.agg.count > 4) {
            cev_error(cev, call, "bitcast source vector is too wide");
            return -1;
        }
        src_count = src.u.agg.count;
        for (uint32_t i = 0; i < src_count; i++) {
            if (!value_to_u32_bits(
                    cev, call->children[1], &src.u.agg.elems[i], &src_words[i]))
            {
                return -1;
            }
        }
    } else {
        src_count = 1;
        if (!value_to_u32_bits(cev, call->children[1], &src, &src_words[0])) {
            return -1;
        }
    }

    if (target->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        uint32_t width = target->width;
        uint32_t needed_words = elem == cev->types->t_f16
            ? (width + 1u) / 2u : width;
        if (elem == cev->types->t_f16 && (width % 2u) != 0u) {
            cev_error(cev, call, "bitcast to vecN<f16> requires an even width");
            return -1;
        }
        if (src_count != needed_words) {
            cev_error(cev, call, "bitcast source and target bit sizes differ");
            return -1;
        }
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * width);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding bitcast");
            return -1;
        }
        for (uint32_t i = 0; i < width; i++) {
            uint32_t bits = elem == cev->types->t_f16
                ? ((src_words[i / 2u] >> ((i % 2u) * 16u)) & 0xffffu)
                : src_words[i];
            if (!make_bitcast_component(cev, call, elem, bits, &elems[i])) {
                return -1;
            }
        }
        out->kind = WGSL_VAL_VEC;
        out->type = target;
        out->u.agg.count = width;
        out->u.agg.elems = elems;
        return 1;
    }

    if (target == cev->types->t_f16) {
        cev_error(cev, call, "bitcast to scalar f16 is not a 32-bit target");
        return -1;
    }
    if (src_count != 1) {
        cev_error(cev, call, "bitcast source and target bit sizes differ");
        return -1;
    }
    return make_bitcast_component(cev, call, target, src_words[0], out) ? 1 : -1;
}

int eval_call_unpack_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_4x8_snorm = name_eq(nm, len, "unpack4x8snorm");
    int is_4x8_unorm = name_eq(nm, len, "unpack4x8unorm");
    int is_4xi8      = name_eq(nm, len, "unpack4xI8");
    int is_4xu8      = name_eq(nm, len, "unpack4xU8");
    int is_2x16_snorm = name_eq(nm, len, "unpack2x16snorm");
    int is_2x16_unorm = name_eq(nm, len, "unpack2x16unorm");
    int is_2x16_float = name_eq(nm, len, "unpack2x16float");
    if (!is_4x8_snorm && !is_4x8_unorm && !is_4xi8 && !is_4xu8 &&
        !is_2x16_snorm && !is_2x16_unorm && !is_2x16_float)
    {
        return 0;
    }
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "builtin '%.*s' expects one argument", (int)len, nm);
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (!wgsl_consteval_materialize(cev, &arg, cev->types->t_u32, call->children[1])) {
        return -1;
    }
    uint32_t x = (uint32_t)arg.u.i;
    uint32_t width = (is_2x16_snorm || is_2x16_unorm || is_2x16_float) ? 2u : 4u;
    WGSLTypeInfo *elem_type =
        is_4xi8 ? cev->types->t_i32 :
        is_4xu8 ? cev->types->t_u32 : cev->types->t_f32;
    WGSLValue elems[4] = {{0}};

    for (uint32_t i = 0; i < width; i++) {
        if (is_4xi8) {
            int8_t v = (int8_t)((x >> (i * 8u)) & 0xffu);
            elems[i].kind = WGSL_VAL_INT;
            elems[i].type = elem_type;
            elems[i].u.i = (int64_t)v;
        } else if (is_4xu8) {
            uint8_t v = (uint8_t)((x >> (i * 8u)) & 0xffu);
            elems[i].kind = WGSL_VAL_INT;
            elems[i].type = elem_type;
            elems[i].u.i = (int64_t)v;
        } else {
            double v = 0.0;
            if (is_4x8_snorm) {
                int8_t s = (int8_t)((x >> (i * 8u)) & 0xffu);
                v = (double)s / 127.0;
                if (v < -1.0) v = -1.0;
            } else if (is_4x8_unorm) {
                uint8_t u = (uint8_t)((x >> (i * 8u)) & 0xffu);
                v = (double)u / 255.0;
            } else if (is_2x16_snorm) {
                int16_t s = (int16_t)((x >> (i * 16u)) & 0xffffu);
                v = (double)s / 32767.0;
                if (v < -1.0) v = -1.0;
            } else if (is_2x16_unorm) {
                uint16_t u = (uint16_t)((x >> (i * 16u)) & 0xffffu);
                v = (double)u / 65535.0;
            } else {
                uint16_t h = (uint16_t)((x >> (i * 16u)) & 0xffffu);
                v = f16_bits_to_double(h);
                if (!isfinite(v)) {
                    cev_error(cev, call,
                        "builtin '%.*s' produced a non-finite result",
                        (int)len, nm);
                    return -1;
                }
            }
            elems[i].kind = WGSL_VAL_FLOAT;
            elems[i].type = elem_type;
            elems[i].u.f = (double)(float)v;
        }
    }

    return make_vector_result(cev, call, width, elem_type, elems, out) ? 1 : -1;
}

int eval_call_pack_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_4x8_snorm = name_eq(nm, len, "pack4x8snorm");
    int is_4x8_unorm = name_eq(nm, len, "pack4x8unorm");
    int is_4xi8      = name_eq(nm, len, "pack4xI8");
    int is_4xu8      = name_eq(nm, len, "pack4xU8");
    int is_4xi8c     = name_eq(nm, len, "pack4xI8Clamp");
    int is_4xu8c     = name_eq(nm, len, "pack4xU8Clamp");
    int is_2x16_snorm = name_eq(nm, len, "pack2x16snorm");
    int is_2x16_unorm = name_eq(nm, len, "pack2x16unorm");
    int is_2x16_float = name_eq(nm, len, "pack2x16float");
    if (!is_4x8_snorm && !is_4x8_unorm && !is_4xi8 && !is_4xu8 &&
        !is_4xi8c && !is_4xu8c &&
        !is_2x16_snorm && !is_2x16_unorm && !is_2x16_float)
    {
        return 0;
    }
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "builtin '%.*s' expects one argument", (int)len, nm);
        return -1;
    }

    uint32_t width = (is_2x16_snorm || is_2x16_unorm || is_2x16_float) ? 2u : 4u;
    WGSLTypeInfo *elem_type =
        (is_4xi8 || is_4xi8c) ? cev->types->t_i32 :
        (is_4xu8 || is_4xu8c) ? cev->types->t_u32 : cev->types->t_f32;
    WGSLTypeInfo *target = wgsl_type_vec(cev->types, (uint8_t)width, elem_type);

    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (!wgsl_consteval_materialize(cev, &arg, target, call->children[1])) {
        return -1;
    }
    if (arg.kind != WGSL_VAL_VEC || arg.u.agg.count != width) {
        cev_error(cev, call, "builtin '%.*s' requires a vec%u argument",
                  (int)len, nm, (unsigned)width);
        return -1;
    }

    uint32_t bits = 0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (is_4xi8 || is_4xi8c) {
            int64_t v = c.u.i;
            if (is_4xi8c) {
                if (v < -128) v = -128;
                if (v > 127) v = 127;
            }
            bits |= ((uint32_t)(uint8_t)(int8_t)v) << (i * 8u);
        } else if (is_4xu8 || is_4xu8c) {
            int64_t v = c.u.i;
            if (is_4xu8c) {
                if (v < 0) v = 0;
                if (v > 255) v = 255;
            }
            bits |= ((uint32_t)(uint8_t)(uint32_t)v) << (i * 8u);
        } else if (is_4x8_snorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 127.0);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            bits |= ((uint32_t)(uint8_t)(int8_t)q) << (i * 8u);
        } else if (is_4x8_unorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 255.0);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            bits |= ((uint32_t)(uint8_t)q) << (i * 8u);
        } else if (is_2x16_snorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 32767.0);
            if (q < -32767) q = -32767;
            if (q > 32767) q = 32767;
            bits |= ((uint32_t)(uint16_t)(int16_t)q) << (i * 16u);
        } else if (is_2x16_unorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 65535.0);
            if (q < 0) q = 0;
            if (q > 65535) q = 65535;
            bits |= ((uint32_t)(uint16_t)q) << (i * 16u);
        } else {
            double x = c.u.f;
            if (!isfinite(x) || x < -65504.0 || x > 65504.0) {
                cev_error(cev, call,
                    "builtin '%.*s' input is outside the finite f16 range",
                    (int)len, nm);
                return -1;
            }
            bits |= ((uint32_t)double_to_f16_bits(x)) << (i * 16u);
        }
    }

    out->kind = WGSL_VAL_INT;
    out->type = cev->types->t_u32;
    out->u.i = (int64_t)bits;
    return 1;
}

int eval_call_dot4_packed_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_i = name_eq(nm, len, "dot4I8Packed");
    int is_u = name_eq(nm, len, "dot4U8Packed");
    if (!is_i && !is_u) return 0;
    if (!call || call->child_count != 3) {
        cev_error(cev, call, "builtin '%.*s' expects two arguments", (int)len, nm);
        return -1;
    }
    WGSLValue a = {0}, b = {0};
    if (!eval_expr(cev, call->children[1], &a) ||
        !eval_expr(cev, call->children[2], &b))
    {
        return -1;
    }
    if (!wgsl_consteval_materialize(cev, &a, cev->types->t_u32, call->children[1]) ||
        !wgsl_consteval_materialize(cev, &b, cev->types->t_u32, call->children[2]))
    {
        return -1;
    }
    uint32_t ua = (uint32_t)a.u.i;
    uint32_t ub = (uint32_t)b.u.i;
    int64_t sum = 0;
    for (uint32_t i = 0; i < 4; i++) {
        uint8_t ba = (uint8_t)((ua >> (i * 8u)) & 0xffu);
        uint8_t bb = (uint8_t)((ub >> (i * 8u)) & 0xffu);
        if (is_i) {
            sum += (int64_t)(int8_t)ba * (int64_t)(int8_t)bb;
        } else {
            sum += (int64_t)ba * (int64_t)bb;
        }
    }
    out->kind = WGSL_VAL_INT;
    out->type = is_i ? cev->types->t_i32 : cev->types->t_u32;
    out->u.i = is_i ? (int64_t)(int32_t)sum : (int64_t)(uint32_t)sum;
    /* One packed instruction = 4 MACs (8 int ops); tensor-path eligible. */
    if (cev->simt.runtime) { cev->cost.iop_count += 8; cev->cost.packed_ops += 1; }
    return 1;
}

int eval_unary_float_builtin_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out)
{
    WGSLTypeInfo *result_type = arg->type;
    if (arg->kind == WGSL_VAL_INT) {
        if (arg->type != cev->types->t_abstract_int) {
            cev_error(cev, call,
                "'%.*s' requires a floating-point argument",
                (int)len, nm);
            return 0;
        }
        if (!wgsl_consteval_materialize(
                cev, arg, cev->types->t_abstract_float, call))
            return 0;
        result_type = cev->types->t_abstract_float;
    }
    if (arg->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, call,
            "'%.*s' requires a floating-point argument", (int)len, nm);
        return 0;
    }

    double x = arg->u.f;
    double r = 0.0;
    const char *domain = NULL;

    if (name_eq(nm, len, "sqrt")) {
        if (x < 0.0) domain = "argument must be >= 0";
        r = sqrt(x);
    } else if (name_eq(nm, len, "inverseSqrt")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = 1.0 / sqrt(x);
    } else if (name_eq(nm, len, "log")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = log(x);
    } else if (name_eq(nm, len, "log2")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = log2(x);
    } else if (name_eq(nm, len, "asin")) {
        if (x < -1.0 || x > 1.0) domain = "argument must be in [-1, 1]";
        r = asin(x);
    } else if (name_eq(nm, len, "acos")) {
        if (x < -1.0 || x > 1.0) domain = "argument must be in [-1, 1]";
        r = acos(x);
    } else if (name_eq(nm, len, "acosh")) {
        if (x < 1.0) domain = "argument must be >= 1";
        r = acosh(x);
    } else if (name_eq(nm, len, "atanh")) {
        if (x <= -1.0 || x >= 1.0) domain = "argument must be in (-1, 1)";
        r = atanh(x);
    } else if (name_eq(nm, len, "ceil")) {
        r = ceil(x);
    } else if (name_eq(nm, len, "floor")) {
        r = floor(x);
    } else if (name_eq(nm, len, "fract")) {
        r = x - floor(x);
    } else if (name_eq(nm, len, "round")) {
        r = round_even(x);
    } else if (name_eq(nm, len, "trunc")) {
        r = trunc(x);
    } else if (name_eq(nm, len, "sin")) {
        r = sin(x);
    } else if (name_eq(nm, len, "cos")) {
        r = cos(x);
    } else if (name_eq(nm, len, "tan")) {
        r = tan(x);
    } else if (name_eq(nm, len, "sinh")) {
        r = sinh(x);
    } else if (name_eq(nm, len, "cosh")) {
        r = cosh(x);
    } else if (name_eq(nm, len, "tanh")) {
        r = tanh(x);
    } else if (name_eq(nm, len, "exp")) {
        r = exp(x);
    } else if (name_eq(nm, len, "exp2")) {
        r = exp2(x);
    } else if (name_eq(nm, len, "atan")) {
        r = atan(x);
    } else if (name_eq(nm, len, "asinh")) {
        r = asinh(x);
    } else if (name_eq(nm, len, "saturate")) {
        r = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
    } else if (name_eq(nm, len, "radians")) {
        r = x * 0.017453292519943295769;
    } else if (name_eq(nm, len, "degrees")) {
        r = x * 57.295779513082320877;
    } else if (name_eq(nm, len, "quantizeToF16")) {
        if (x < -65504.0 || x > 65504.0) {
            domain = "argument must be in the finite f16 range";
        }
        r = x;
    }

    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, result_type, r, out);
}

int eval_call_unary_float_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "sqrt") || name_eq(nm, len, "inverseSqrt") ||
        name_eq(nm, len, "log")  || name_eq(nm, len, "log2") ||
        name_eq(nm, len, "asin") || name_eq(nm, len, "acos") ||
        name_eq(nm, len, "acosh") || name_eq(nm, len, "atanh") ||
        name_eq(nm, len, "ceil") || name_eq(nm, len, "floor") ||
        name_eq(nm, len, "fract") || name_eq(nm, len, "round") ||
        name_eq(nm, len, "trunc") || name_eq(nm, len, "sin") ||
        name_eq(nm, len, "cos") || name_eq(nm, len, "tan") ||
        name_eq(nm, len, "sinh") || name_eq(nm, len, "cosh") ||
        name_eq(nm, len, "tanh") || name_eq(nm, len, "exp") ||
        name_eq(nm, len, "exp2") || name_eq(nm, len, "atan") ||
        name_eq(nm, len, "asinh") || name_eq(nm, len, "saturate") ||
        name_eq(nm, len, "radians") || name_eq(nm, len, "degrees") ||
        name_eq(nm, len, "quantizeToF16");
    if (!supported) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }

    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;

    if (arg.kind == WGSL_VAL_VEC) {
        WGSLTypeInfo *vt = arg.type;
        if (!vt || vt->kind != WGSL_TYPE_VEC) {
            cev_error(cev, call,
                "'%.*s' requires a floating-point argument", (int)len, nm);
            return -1;
        }
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * arg.u.agg.count);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding builtin '%.*s'",
                      (int)len, nm);
            return -1;
        }
        for (uint32_t i = 0; i < arg.u.agg.count; i++) {
            WGSLValue c = arg.u.agg.elems[i];
            if (!eval_unary_float_builtin_scalar(cev, call, nm, len, &c, &elems[i])) {
                return -1;
            }
        }
        out->kind = WGSL_VAL_VEC;
        out->type = vt;
        out->u.agg.count = arg.u.agg.count;
        out->u.agg.elems = elems;
        if (cev->simt.runtime) {
            uint32_t w = builtin_flop_weight(nm, len);
            cev->cost.flop_count += arg.u.agg.count * w;
            if (w > 1) cev->cost.op_bucket[WGSL_OPB_FTRANS] += arg.u.agg.count;  /* SFU/transcendental */
        }
        return 1;
    }

    if (!eval_unary_float_builtin_scalar(cev, call, nm, len, &arg, out)) {
        return -1;
    }
    if (cev->simt.runtime) {
        uint32_t w = builtin_flop_weight(nm, len);
        cev->cost.flop_count += w;
        if (w > 1) cev->cost.op_bucket[WGSL_OPB_FTRANS]++;
    }
    return 1;
}

int fold_abs_sign_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out)
{
    if (name_eq(nm, len, "abs")) {
        if (arg->kind == WGSL_VAL_INT) {
            *out = *arg;
            if (arg->type == cev->types->t_u32 || arg->u.i == INT64_MIN ||
                (arg->type == cev->types->t_i32 && arg->u.i == INT32_MIN))
            {
                return 1;
            }
            if (arg->u.i < 0) out->u.i = -arg->u.i;
            return 1;
        }
        if (!require_float_scalar(cev, call, nm, len, arg)) return 0;
        return finish_float_builtin_result(
            cev, call, nm, len, arg->type, fabs(arg->u.f), out);
    }

    if (name_eq(nm, len, "sign")) {
        if (arg->kind == WGSL_VAL_INT) {
            if (arg->type == cev->types->t_u32) {
                cev_error(cev, call, "'sign' does not accept u32");
                return 0;
            }
            *out = *arg;
            out->u.i = arg->u.i > 0 ? 1 : (arg->u.i < 0 ? -1 : 0);
            return 1;
        }
        if (!require_float_scalar(cev, call, nm, len, arg)) return 0;
        double r = arg->u.f > 0.0 ? 1.0 : (arg->u.f < 0.0 ? -1.0 : 0.0);
        return finish_float_builtin_result(
            cev, call, nm, len, arg->type, r, out);
    }
    return 0;
}

int eval_call_abs_sign_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "abs") && !name_eq(nm, len, "sign")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = value_component(&arg, i);
        if (!fold_abs_sign_scalar(cev, call, nm, len, &c, &tmp[i])) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

