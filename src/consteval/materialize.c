/**
 * @file materialize.c — materialize, pair promotion, op billing helpers
 */
#include "internal/consteval_priv.h"


int int_fits_target(int64_t v, WGSLTypeKind k) {
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

int float_fits_target(double v, WGSLTypeKind k) {
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
            uint32_t attrs = wgsl_member_attr_count(m);
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

/* Pair promotion (§6.1.2). */

int promote_pair(
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

/* Operator folding. */

/* Apply `% / /` for ints, taking signedness into account. */
int int_div_mod(
    WGSLConstEvaluator *cev, WGSLNode *at,
    int64_t a, int64_t b, int is_unsigned, int is_mod, int64_t *out)
{
    if (b == 0) {
        if (cev->simt.runtime) {          /* §8.7 runtime: x / 0 == x, x % 0 == x */
            *out = a; return 1;
        }
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
        if (cev->simt.runtime) { *out = a; return 1; }   /* §8.7 runtime: result is e1 */
        cev_error(cev, at, "signed integer overflow in division");
        return 0;
    }
    *out = is_mod ? (a % b) : (a / b);
    return 1;
}

int checked_abstract_int_result(
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

/* Cost model: bill one scalar op into its instruction-mix bucket and the
 * matching total.  ONE convention (documented in wgsl.h): every executed scalar
 * arithmetic / bitwise / shift / comparison op counts 1 (float → flop_count,
 * integer → iop_count); a float comparison is a flop, an integer comparison is
 * an iop; vectors fold per-component so they self-count as `width`. */
void cev_bill_op(WGSLConstEvaluator *cev, WGSLTokenKind op, int is_float) {
    if (!cev->simt.runtime) return;
    WGSLOpBucket b;
    switch (op) {
    case WGSL_TOK_PLUS: case WGSL_TOK_MINUS:  b = is_float ? WGSL_OPB_FADD : WGSL_OPB_IADD; break;
    case WGSL_TOK_STAR:                        b = is_float ? WGSL_OPB_FMUL : WGSL_OPB_IMUL; break;
    case WGSL_TOK_SLASH: case WGSL_TOK_PERCENT:b = is_float ? WGSL_OPB_FDIV : WGSL_OPB_IDIV; break;
    case WGSL_TOK_AMP: case WGSL_TOK_PIPE: case WGSL_TOK_CARET:      b = WGSL_OPB_IADD;   break;
    case WGSL_TOK_LESS_LESS: case WGSL_TOK_GREATER_GREATER:          b = WGSL_OPB_ISHIFT; break;
    case WGSL_TOK_EQUAL_EQUAL: case WGSL_TOK_BANG_EQUAL:
    case WGSL_TOK_LESS: case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_GREATER: case WGSL_TOK_GREATER_EQUAL:              b = WGSL_OPB_CMP;    break;
    default: return;
    }
    cev->cost.op_bucket[b]++;
    if (b == WGSL_OPB_FADD || b == WGSL_OPB_FMUL || b == WGSL_OPB_FDIV) cev->cost.flop_count++;
    else if (b == WGSL_OPB_CMP) { if (is_float) cev->cost.flop_count++; else cev->cost.iop_count++; }
    else cev->cost.iop_count++;
}

