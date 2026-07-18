/**
 * @file core_binary.c — binary operator typing
 */
#include "internal/exprs_priv.h"


int type_binary(
    WGSLTypeChecker *tc, WGSLNode *n, WGSLTypeInfo *expected)
{
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        wgsl_tc_error(tc, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)wgsl_node_op_kind_u32(n);

    /* §4.3 strategy for binary ops:
     *  - Logical: children expect bool.
     *  - Arithmetic / bitwise / shifts: type children FREE (abstract),
     *    then materialize the *result* into `expected`.  Pushing expected
     *    into operands early breaks overflow detection
     *    (`let x: i32 = 1073741824 * 2` must fail as AbstractInt 2^31).
     *  - Comparisons: free children; result is bool. */
    int is_logical = (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE);
    WGSLTypeInfo *lhs_exp = is_logical ? tc->types->t_bool : NULL;
    WGSLTypeInfo *rhs_exp = is_logical ? tc->types->t_bool : NULL;

    if (is_logical) {
        if (!wgsl_tc_type_expr_exp(tc, n->children[0], lhs_exp)) return 0;
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
        int rhs_ok = wgsl_tc_type_expr_exp(tc, n->children[1], rhs_exp);
        if (short_circuit) tc->suppress_consteval_value_errors--;
        if (!rhs_ok) return 0;
    } else {
        if (!wgsl_tc_type_expr_exp(tc, n->children[0], lhs_exp)) return 0;
        if (!wgsl_tc_type_expr_exp(tc, n->children[1], rhs_exp)) return 0;
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
        /* Result type = LHS type; abstract stays abstract for use-site
         * materialization / range checks (§4.3 / §6.2.1). */
        (void)expected;
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
        (void)expected;
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

