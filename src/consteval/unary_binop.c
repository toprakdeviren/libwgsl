/**
 * @file unary_binop.c — unary eval, combine_binary, public binop entry
 */
#include "internal/consteval_priv.h"


int eval_unary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *n,
    WGSLTokenKind op, WGSLValue inner, WGSLValue *out)
{
    /* Unary negate / logical-not / bitwise-not are add-class ALU ops; they used
     * to bill nothing.  Float negate → flop, integer/bool → iop. */
    if (cev->simt.runtime && (op == WGSL_TOK_MINUS || op == WGSL_TOK_BANG || op == WGSL_TOK_TILDE)) {
        int isf = val_is_float(&inner);
        cev->cost.op_bucket[isf ? WGSL_OPB_FADD : WGSL_OPB_IADD]++;
        if (isf) cev->cost.flop_count++; else cev->cost.iop_count++;
        if (isf && inner.type && inner.type->kind == WGSL_TYPE_F16) cev->cost.flop_f16++;
    }
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

int eval_unary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 1 || !n->children[0]) {
        cev_error(cev, n, "malformed unary expression");
        return 0;
    }
    WGSLValue inner = {0};
    if (!eval_expr(cev, n->children[0], &inner)) return 0;

    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);
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

/* Combine two evaluated operands under `op` (matrix / vector / bool / numeric
 * with promotion + integer shift rules).  Shared by `eval_binary` and the
 * public `wgsl_consteval_binop`.  Note: mutates `a`/`b` via `promote_pair`. */
int combine_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
                          WGSLValue *a, WGSLValue *b, WGSLValue *out) {
    if (a->kind == WGSL_VAL_MAT || b->kind == WGSL_VAL_MAT) {
        return eval_matrix_binary(cev, n, op, a, b, out);
    }

    if (a->kind == WGSL_VAL_VEC || b->kind == WGSL_VAL_VEC) {
        return eval_binary_componentwise(cev, n, op, a, b, out);
    }

    /* Bool operands → handle separately (no numeric promotion). */
    if (val_is_bool(a) && val_is_bool(b)) {
        return eval_binary_bool(cev, n, op, a, b, out);
    }

    /* Mixed types → numeric promotion. */
    if (!val_is_numeric(a) || !val_is_numeric(b)) {
        cev_error(cev, n, "operator requires numeric or bool operands");
        return 0;
    }
    if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
        if (!val_is_int(a) || !val_is_int(b)) {
            cev_error(cev, n, "shift requires integer operands");
            return 0;
        }
        if (b->type != cev->types->t_u32 &&
            b->type != cev->types->t_abstract_int)
        {
            cev_error(cev, n, "shift amount must be u32");
            return 0;
        }
        return eval_binary_int(cev, n, op, a, b, out);
    }
    if (!promote_pair(cev, a, b, n)) return 0;

    /* Dispatch on the (now-shared) operand kind. */
    if (val_is_int(a))   return eval_binary_int  (cev, n, op, a, b, out);
    if (val_is_float(a)) return eval_binary_float(cev, n, op, a, b, out);

    cev_error(cev, n, "unhandled operand kind in binary op");
    return 0;
}

int eval_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);

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

    return combine_binary(cev, n, op, &a, &b, out);
}

/* Public interpreter: compound-assign / inc / dec need to combine two
 * already-evaluated values under an explicit operator.  Copies the operands so
 * the caller's values survive `promote_pair`'s in-place rewrites. */
int wgsl_consteval_binop(WGSLConstEvaluator *cev, WGSLNode *ctx, WGSLTokenKind op,
                         const WGSLValue *a, const WGSLValue *b, WGSLValue *out) {
    WGSLValue la = *a, lb = *b;
    return combine_binary(cev, ctx, op, &la, &lb, out);
}

