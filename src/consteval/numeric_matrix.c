/**
 * @file numeric_matrix.c — reflect, faceForward, refract, transpose
 */
#include "internal/consteval_priv.h"


int eval_call_reflect_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "reflect")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'reflect' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue I = vals[0].u.agg.elems[i];
        WGSLValue N = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &I, &N, call) ||
            !wgsl_consteval_materialize(cev, &I, rt, call) ||
            !wgsl_consteval_materialize(cev, &N, rt, call))
        {
            return -1;
        }
        double r = I.u.f - 2.0 * d * N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

int eval_call_face_forward_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "faceForward")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'faceForward' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 3, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[2], &vals[1],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    int keep = d < 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue N = vals[0].u.agg.elems[i];
        if (!wgsl_consteval_materialize(cev, &N, rt, call)) return -1;
        double r = keep ? N.u.f : -N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

int eval_call_refract_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "refract")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'refract' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    WGSLValue eta = {0};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &eta))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d) ||
        !wgsl_consteval_materialize(cev, &eta, rt, call) ||
        !require_float_scalar(cev, call, nm, len, &eta))
    {
        return -1;
    }

    double eta2 = 0.0;
    double d2 = 0.0;
    double one_minus_d2 = 0.0;
    double term = 0.0;
    double k = 0.0;
    if (!fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta.u.f, eta.u.f, &eta2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, d, d, &d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, d2, &one_minus_d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta2, one_minus_d2, &term) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, term, &k))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        double r = 0.0;
        if (k >= 0.0) {
            WGSLValue I = vals[0].u.agg.elems[i];
            WGSLValue N = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &I, &N, call) ||
                !wgsl_consteval_materialize(cev, &I, rt, call) ||
                !wgsl_consteval_materialize(cev, &N, rt, call))
            {
                return -1;
            }
            double eta_i = 0.0;
            double eta_d = 0.0;
            double sum = 0.0;
            double scaled_n = 0.0;
            WGSLValue sqrt_v = {0};
            if (!fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, I.u.f, &eta_i) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, d, &eta_d) ||
                !finish_float_builtin_result(
                    cev, call, nm, len, rt, sqrt(k), &sqrt_v) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_PLUS, eta_d, sqrt_v.u.f, &sum) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, sum, N.u.f, &scaled_n) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_MINUS, eta_i, scaled_n, &r))
            {
                return -1;
            }
        }
        if (!finish_float_builtin_result_relaxed_range(
                cev, call, nm, len, rt, r, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

int eval_call_transpose_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "transpose")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'transpose' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_MAT || !arg.type ||
        arg.type->kind != WGSL_TYPE_MAT)
    {
        cev_error(cev, call, "builtin 'transpose' requires a matrix argument");
        return -1;
    }
    uint32_t cols = arg.type->width;
    uint32_t rows = arg.type->rows;
    uint32_t count = cols * rows;
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * count);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding transpose");
        return -1;
    }
    for (uint32_t out_col = 0; out_col < rows; out_col++) {
        for (uint32_t out_row = 0; out_row < cols; out_row++) {
            uint32_t src_col = out_row;
            uint32_t src_row = out_col;
            elems[out_col * cols + out_row] =
                arg.u.agg.elems[src_col * rows + src_row];
        }
    }
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(
        cev->types, (uint8_t)rows, (uint8_t)cols,
        (WGSLTypeInfo *)arg.type->ref);
    out->u.agg.count = count;
    out->u.agg.elems = elems;
    return 1;
}

