/**
 * @file generated_overloads.c — expression typing module (was exprs/generated_overloads.inc).
 */
#include "internal/exprs_priv.h"

/* BSHAPE / types: exprs_priv.h */

int resolve_overload_silent(
    WGSLTypeChecker *tc,
    const WGSLOverloadCandidate *cands, int cand_count,
    WGSLTypeInfo *const *arg_types, const int *arg_is_const, int arg_count,
    const char *what, const WGSLNode *at)
{
    size_t saved_count = tc->diag ? tc->diag->count : 0;
    int saved_errors = tc->diag ? tc->diag->error_count : 0;
    int saved_had_error = tc->had_error;
    int picked = resolve_overload(
        tc, cands, cand_count, arg_types, arg_is_const, arg_count, what, at);
    if (picked < 0 && tc->diag) {
        tc->diag->count = saved_count;
        tc->diag->error_count = saved_errors;
        tc->had_error = saved_had_error;
    }
    return picked;
}

int shape_satisfies_req(const WGSLTypeInfo *shape, WGSLBuiltinShapeReq req) {
    if (!shape) return 0;
    switch (req) {
    case BSHAPE_SCALAR_OR_VEC:
        return wgsl_type_is_scalar(shape) || shape->kind == WGSL_TYPE_VEC;
    case BSHAPE_VEC:
        return shape->kind == WGSL_TYPE_VEC;
    case BSHAPE_VEC3:
        return shape->kind == WGSL_TYPE_VEC && shape->width == 3;
    case BSHAPE_MAT:
        return shape->kind == WGSL_TYPE_MAT;
    case BSHAPE_MAT_SQUARE:
        return shape->kind == WGSL_TYPE_MAT && shape->width == shape->rows;
    }
    return 0;
}

WGSLTypeInfo *resolve_t_family_prefix(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name,
    WGSLParamFamily fam, int arity, int total_args, WGSLBuiltinShapeReq req)
{
    if (!builtin_arg_count_is(call, total_args) || arity <= 0 ||
        arity > WGSL_OL_MAX_PARAMS)
    {
        return NULL;
    }

    WGSLTypeInfo *args[WGSL_OL_MAX_PARAMS] = {0};
    int arg_const[WGSL_OL_MAX_PARAMS] = {0};
    for (int i = 0; i < arity; i++) {
        WGSLNode *arg = call->children[1 + i];
        WGSLTypeInfo *t = builtin_arg_type(tc, call, i);
        if (!t) return NULL;
        args[i] = t;
        arg_const[i] = is_const_expr(arg);
    }

    const WGSLTypeInfo *shape = pick_shape_arg(args, arity);
    if (!shape_satisfies_req(shape, req)) return NULL;

    WGSLOverloadCandidate cands[WGSL_OL_MAX_CANDS];
    int nc = build_t_candidates(tc, fam, args, arity, cands);
    if (nc <= 0) return NULL;
    int picked = resolve_overload_silent(
        tc, cands, nc, args, arg_const, arity, name, call);
    return picked >= 0 ? cands[picked].result : NULL;
}

WGSLTypeInfo *resolve_t_family_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name,
    WGSLParamFamily fam, int arity, WGSLBuiltinShapeReq req)
{
    return resolve_t_family_prefix(tc, call, name, fam, arity, arity, req);
}

int match_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    return type_converts_to(builtin_arg_type(tc, call, index), tc->types->t_u32);
}

int match_i32_or_u32_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    WGSLTypeInfo *t = builtin_arg_type(tc, call, index);
    return type_converts_to(t, tc->types->t_i32) ||
           type_converts_to(t, tc->types->t_u32);
}

int subgroup_shuffle_second_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, 1);
    int type_ok = type_converts_to(t, tc->types->t_u32);
    if (!type_ok && strcmp(name, "subgroupShuffle") == 0) {
        type_ok = type_converts_to(t, tc->types->t_i32);
    }
    if (!type_ok) return 0;

    if (call->child_count < 3 || !is_const_expr(call->children[2])) return 1;
    int64_t value = 0;
    if (!eval_const_int_silent(tc, call->children[2], &value)) return 1;
    if (value < 0 || value >= 128) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be in [0, 128)",
            name);
        return 0;
    }
    return 1;
}

int broadcast_second_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!match_i32_or_u32_arg(tc, call, 1)) return 0;

    if (call->child_count < 3 || !is_const_expr(call->children[2])) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be a const-expression",
            name);
        return 0;
    }

    int64_t value = 0;
    if (!eval_const_int_silent(tc, call->children[2], &value)) return 0;
    int64_t hi = strcmp(name, "quadBroadcast") == 0 ? 4 : 128;
    if (value < 0 || value >= hi) {
        wgsl_tc_error(tc, call->children[2],
            "'%s' second argument must be in [0, %lld)",
            name, (long long)hi);
        return 0;
    }
    return 1;
}

int match_bool_arg(WGSLTypeChecker *tc, WGSLNode *call, int index) {
    return builtin_arg_type(tc, call, index) == tc->types->t_bool;
}

int match_vec_arg(WGSLTypeChecker *tc, WGSLNode *call, int index,
                         uint8_t width, WGSLTypeInfo *elem)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, index);
    WGSLTypeInfo *want = wgsl_type_vec(tc->types, width, elem);
    return t && want && wgsl_type_conversion_rank(t, want) >= 0;
}

int match_all_any(WGSLTypeChecker *tc, WGSLNode *call) {
    if (!builtin_arg_count_is(call, 1)) return 0;
    WGSLTypeInfo *t = builtin_arg_type(tc, call, 0);
    if (t == tc->types->t_bool) return 1;
    return t && t->kind == WGSL_TYPE_VEC &&
           (WGSLTypeInfo *)t->ref == tc->types->t_bool;
}

WGSLTypeInfo *resolve_select_result(WGSLTypeChecker *tc, WGSLNode *call) {
    if (!builtin_arg_count_is(call, 3)) return NULL;

    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, "select", PF_NUMERIC_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
    if (!value) {
        WGSLTypeInfo *a = builtin_arg_type(tc, call, 0);
        WGSLTypeInfo *b = builtin_arg_type(tc, call, 1);
        int scalar_bool = a == tc->types->t_bool && b == tc->types->t_bool;
        int vec_bool = a && b && a->kind == WGSL_TYPE_VEC &&
                       b->kind == WGSL_TYPE_VEC &&
                       a->width == b->width &&
                       (WGSLTypeInfo *)a->ref == tc->types->t_bool &&
                       (WGSLTypeInfo *)b->ref == tc->types->t_bool;
        if (!scalar_bool && !vec_bool) return NULL;
        value = a;
    }

    WGSLTypeInfo *cond = builtin_arg_type(tc, call, 2);
    if (cond == tc->types->t_bool) return value;
    if (value && value->kind == WGSL_TYPE_VEC &&
        cond && cond->kind == WGSL_TYPE_VEC &&
        cond->width == value->width &&
        (WGSLTypeInfo *)cond->ref == tc->types->t_bool)
    {
        return value;
    }
    return NULL;
}

int bitfield_const_range_ok(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, int offset_arg, int count_arg)
{
    if (!call || offset_arg < 0 || count_arg < 0) return 0;
    uint32_t off_child = (uint32_t)offset_arg + 1u;
    uint32_t cnt_child = (uint32_t)count_arg + 1u;
    if (off_child >= call->child_count || cnt_child >= call->child_count) {
        return 0;
    }
    WGSLNode *offset_node = call->children[off_child];
    WGSLNode *count_node = call->children[cnt_child];
    if (!is_const_expr(offset_node) || !is_const_expr(count_node)) return 1;

    int64_t offset = 0;
    int64_t count = 0;
    if (!eval_const_int_silent(tc, offset_node, &offset) ||
        !eval_const_int_silent(tc, count_node, &count))
    {
        return 1;
    }
    if (offset < 0 || count < 0 || offset + count > 32) {
        wgsl_tc_error(tc, call,
            "builtin '%s' domain error: offset + count must be <= 32",
            name);
        return 0;
    }
    return 1;
}

WGSLTypeInfo *resolve_extract_bits(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, name, PF_INTEGER_T, 1, 3, BSHAPE_SCALAR_OR_VEC);
    if (!value || !match_u32_arg(tc, call, 1) || !match_u32_arg(tc, call, 2)) {
        return NULL;
    }
    if (!bitfield_const_range_ok(tc, call, name, 1, 2)) return NULL;
    return value;
}

WGSLTypeInfo *resolve_insert_bits(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 4)) return NULL;
    WGSLTypeInfo *value = resolve_t_family_prefix(
        tc, call, name, PF_INTEGER_T, 2, 4, BSHAPE_SCALAR_OR_VEC);
    if (!value || !match_u32_arg(tc, call, 2) || !match_u32_arg(tc, call, 3)) {
        return NULL;
    }
    if (!bitfield_const_range_ok(tc, call, name, 2, 3)) return NULL;
    return value;
}

WGSLTypeInfo *resolve_ldexp_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 2)) return NULL;
    WGSLTypeInfo *x = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 1, 2, BSHAPE_SCALAR_OR_VEC);
    if (!x) return NULL;
    WGSLTypeInfo *e = builtin_arg_type(tc, call, 1);
    WGSLTypeInfo *ee = shape_elem(e);
    int xw = shape_width(x);
    int ew = shape_width(e);
    if (!ee || ew != xw ||
        !(ee == tc->types->t_abstract_int || ee == tc->types->t_i32))
    {
        return NULL;
    }
    return x;
}

WGSLTypeInfo *resolve_mix_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *shape = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
    if (!shape) return NULL;
    WGSLTypeInfo *factor = builtin_arg_type(tc, call, 2);
    WGSLTypeInfo *elem = shape_elem(shape);
    if (type_converts_to(factor, shape)) return shape;
    if (shape->kind == WGSL_TYPE_VEC && type_converts_to(factor, elem)) {
        return shape;
    }
    return NULL;
}

WGSLTypeInfo *resolve_refract_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    if (!builtin_arg_count_is(call, 3)) return NULL;
    WGSLTypeInfo *shape = resolve_t_family_prefix(
        tc, call, name, PF_FLOAT_T, 2, 3, BSHAPE_VEC);
    if (!shape) return NULL;
    WGSLTypeInfo *elem = shape_elem(shape);
    if (!type_converts_to(builtin_arg_type(tc, call, 2), elem)) return NULL;
    return shape;
}

WGSLTypeInfo *resolve_transpose_exact(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name)
{
    WGSLTypeInfo *mat = resolve_t_family_exact(
        tc, call, name, PF_FLOAT_T, 1, BSHAPE_MAT);
    if (!mat || mat->kind != WGSL_TYPE_MAT) return NULL;
    return wgsl_type_mat(
        tc->types, mat->rows, mat->width, (WGSLTypeInfo *)mat->ref);
}

WGSLTypeInfo *resolve_atomic_exact(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type,
    WGSLBuiltinOverloadPattern pattern)
{
    WGSLTypeInfo *T = atomic_pointee_T(arg0_type);
    if (!T) return NULL;
    int argc = builtin_arg_count(call);
    switch (pattern) {
    case BOP_ATOMIC_LOAD:
        return argc == 1 ? T : NULL;
    case BOP_ATOMIC_STORE:
        return argc == 2 && type_converts_to(builtin_arg_type(tc, call, 1), T)
            ? T : NULL;
    case BOP_ATOMIC_RMW:
        return argc == 2 && type_converts_to(builtin_arg_type(tc, call, 1), T)
            ? T : NULL;
    case BOP_ATOMIC_CX:
        return argc == 3 &&
               type_converts_to(builtin_arg_type(tc, call, 1), T) &&
               type_converts_to(builtin_arg_type(tc, call, 2), T)
            ? T : NULL;
    default:
        return NULL;
    }
}

int array_length_arg_ok(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type)
{
    if (!builtin_arg_count_is(call, 1)) return 0;
    if (arg0_type && arg0_type->kind == WGSL_TYPE_PTR) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)arg0_type->ref;
        uint8_t as = (uint8_t)(arg0_type->flags & 0xFFu);
        return elem && elem->kind == WGSL_TYPE_ARRAY &&
               elem->array_len == 0 && as == WGSL_AS_STORAGE;
    }
    if (arg0_type && arg0_type->kind == WGSL_TYPE_ARRAY &&
        call->child_count >= 2 &&
        call->children[1] &&
        call->children[1]->kind == WGSL_NODE_EXPR_ADDR_OF &&
        call->children[1]->child_count >= 1)
    {
        WGSLAddressSpace as = WGSL_AS_NONE;
        WGSLAccessMode am = WGSL_ACCESS_NONE;
        wgsl_tc_get_ref_space_and_mode(tc, call->children[1]->children[0], &as, &am);
        (void)am;
        return as == WGSL_AS_STORAGE && arg0_type->array_len == 0;
    }
    return 0;
}

void check_array_length_call(WGSLTypeChecker *tc, WGSLNode *call,
                                    WGSLTypeInfo *arg0_type)
{
    if (!array_length_arg_ok(tc, call, arg0_type)) {
        wgsl_tc_error(tc, call,
            "arrayLength requires a pointer to a runtime-sized "
            "array in the storage address space");
    }
}

WGSLTypeInfo *resolve_workgroup_uniform_load(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *arg0_type)
{
    if (!builtin_arg_count_is(call, 1) || !arg0_type) return NULL;
    if (arg0_type->kind == WGSL_TYPE_PTR) {
        uint8_t as = (uint8_t)(arg0_type->flags & 0xFFu);
        uint8_t am = (uint8_t)((arg0_type->flags >> 8) & 0xFFu);
        if (as == WGSL_AS_WORKGROUP && am == WGSL_ACCESS_READ_WRITE) {
            WGSLTypeInfo *store = (WGSLTypeInfo *)arg0_type->ref;
            if (store && store->kind == WGSL_TYPE_ATOMIC) {
                return (WGSLTypeInfo *)store->ref;
            }
            if (!wgsl_tc_is_constructible_deep(tc, store)) return NULL;
            return store;
        }
        return NULL;
    }
    if (call->child_count >= 2 &&
        call->children[1] &&
        call->children[1]->kind == WGSL_NODE_EXPR_ADDR_OF &&
        call->children[1]->child_count >= 1)
    {
        WGSLNode *base = call->children[1]->children[0];
        if (base && base->kind == WGSL_NODE_EXPR_IDENT) {
            WGSLSymbol *bs = wgsl_node_resolved_symbol(base);
            if (bs && bs->as == WGSL_AS_WORKGROUP &&
                bs->am == WGSL_ACCESS_READ_WRITE)
            {
                if (!wgsl_tc_is_constructible_deep(tc, arg0_type)) return NULL;
                return arg0_type;
            }
        }
    }
    return NULL;
}

WGSLTypeInfo *result_from_builtin_return(
    WGSLTypeChecker *tc, WGSLBuiltinReturn ret,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo *resolved)
{
    WGSLTypeInfo *src = resolved ? resolved : (arg0_type ? arg0_type : fallback);
    switch (ret) {
    case BR_VOID:
        return tc->types->t_void;
    case BR_BOOL:
        return tc->types->t_bool;
    case BR_I32:
        return tc->types->t_i32;
    case BR_U32:
        return tc->types->t_u32;
    case BR_ARG0:
        return src;
    case BR_ARG0_ELEM: {
        WGSLTypeInfo *elem = element_type_of(src);
        return elem ? elem : src;
    }
    case BR_VEC4F:
        return wgsl_type_vec(tc->types, 4, tc->types->t_f32);
    case BR_VEC4U:
        return wgsl_type_vec(tc->types, 4, tc->types->t_u32);
    case BR_VEC4I:
        return wgsl_type_vec(tc->types, 4, tc->types->t_i32);
    case BR_VEC2F:
        return wgsl_type_vec(tc->types, 2, tc->types->t_f32);
    case BR_ATOMIC_CX_RESULT:
        return wgsl_type_atomic_cx_result(tc->types, src);
    case BR_FREXP_RESULT:
        return wgsl_type_frexp_result(tc->types, src);
    case BR_MODF_RESULT:
        return wgsl_type_modf_result(tc->types, src);
    case BR_TEMPLATE:
        return src;
    }
    return fallback;
}

int builtin_requires_concrete_result(const char *name, uint32_t name_len) {
    return (name_len >= 8 && memcmp(name, "subgroup", 8) == 0) ||
           (name_len >= 4 && memcmp(name, "quad", 4) == 0);
}

int match_builtin_overload_row(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLBuiltinOverload *row,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo **out_resolved)
{
    WGSLTypeInfo *r = NULL;
    switch (row->pattern) {
    case BOP_VOID_0:
        r = builtin_arg_count_is(call, 0) ? tc->types->t_void : NULL;
        break;
    case BOP_ALL_ANY:
        r = match_all_any(tc, call) ? tc->types->t_bool : NULL;
        break;
    case BOP_SELECT:
        r = resolve_select_result(tc, call);
        break;
    case BOP_ARRAY_LENGTH:
        r = array_length_arg_ok(tc, call, arg0_type) ? tc->types->t_u32 : NULL;
        break;
    case BOP_WORKGROUP_UNIFORM_LOAD:
        r = resolve_workgroup_uniform_load(tc, call, arg0_type);
        break;
    case BOP_T_FLOAT_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_FLOAT_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_FLOAT_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 3, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 2, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_NUMERIC_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 3, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_SIGN_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_SIGN_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_T_INTEGER_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_INTEGER_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_EXTRACT_BITS:
        r = resolve_extract_bits(tc, call, row->name);
        break;
    case BOP_INSERT_BITS:
        r = resolve_insert_bits(tc, call, row->name);
        break;
    case BOP_LDEXP:
        r = resolve_ldexp_exact(tc, call, row->name);
        break;
    case BOP_MIX:
        r = resolve_mix_exact(tc, call, row->name);
        break;
    case BOP_DOT:
        r = resolve_t_family_exact(tc, call, row->name, PF_NUMERIC_T, 2, BSHAPE_VEC);
        break;
    case BOP_DOT4_U32:
        r = builtin_arg_count_is(call, 2) &&
            match_u32_arg(tc, call, 0) && match_u32_arg(tc, call, 1)
            ? tc->types->t_u32 : NULL;
        break;
    case BOP_LENGTH:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_DETERMINANT:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_MAT_SQUARE);
        break;
    case BOP_CROSS:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_VEC3);
        break;
    case BOP_TRANSPOSE:
        r = resolve_transpose_exact(tc, call, row->name);
        break;
    case BOP_VEC_FLOAT_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 1, BSHAPE_VEC);
        break;
    case BOP_VEC_FLOAT_2:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 2, BSHAPE_VEC);
        break;
    case BOP_VEC_FLOAT_3:
        r = resolve_t_family_exact(tc, call, row->name, PF_FLOAT_T, 3, BSHAPE_VEC);
        break;
    case BOP_REFRACT:
        r = resolve_refract_exact(tc, call, row->name);
        break;
    case BOP_T_F32_1:
        r = resolve_t_family_exact(tc, call, row->name, PF_F32_T, 1, BSHAPE_SCALAR_OR_VEC);
        break;
    case BOP_ARG_U32_1:
        r = builtin_arg_count_is(call, 1) && match_u32_arg(tc, call, 0)
            ? tc->types->t_u32 : NULL;
        break;
    case BOP_ARG_VEC4F_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_f32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC4I_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_i32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC4U_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 4, tc->types->t_u32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ARG_VEC2F_1:
        r = builtin_arg_count_is(call, 1) &&
            match_vec_arg(tc, call, 0, 2, tc->types->t_f32)
            ? builtin_arg_type(tc, call, 0) : NULL;
        break;
    case BOP_ATOMIC_LOAD:
    case BOP_ATOMIC_STORE:
    case BOP_ATOMIC_RMW:
    case BOP_ATOMIC_CX:
        r = resolve_atomic_exact(tc, call, arg0_type, row->pattern);
        break;
    case BOP_BOOL_1:
        r = builtin_arg_count_is(call, 1) && match_bool_arg(tc, call, 0)
            ? tc->types->t_bool : NULL;
        break;
    case BOP_T_NUMERIC_U32_2:
        if (!builtin_arg_count_is(call, 2)) break;
        if (strcmp(row->name, "subgroupBroadcast") == 0 ||
            strcmp(row->name, "quadBroadcast") == 0)
        {
            if (!broadcast_second_arg_ok(tc, call, row->name)) break;
        } else if (strncmp(row->name, "subgroupShuffle", 15) == 0) {
            if (!subgroup_shuffle_second_arg_ok(tc, call, row->name)) break;
        } else if (!match_i32_or_u32_arg(tc, call, 1)) {
            break;
        }
        {
            r = resolve_t_family_prefix(tc, call, row->name, PF_NUMERIC_T, 1,
                                        2, BSHAPE_SCALAR_OR_VEC);
        }
        break;
    }
    if (!r) return 0;
    *out_resolved = r;
    return 1;
}

int resolve_generated_builtin_overload(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *arg0_type, WGSLTypeInfo *fallback, WGSLTypeInfo **out_type)
{
    /* §4.2 — O(log n) name → contiguous overload slice via sorted index. */
    const WGSLBuiltinOverloadIndex *ix = find_overload_index(name, name_len);
    if (!ix) return 0;
    for (uint16_t i = 0; i < ix->count; i++) {
        const WGSLBuiltinOverload *row = &kBuiltinOverloads[ix->begin + i];
        WGSLTypeInfo *resolved = NULL;
        if (!match_builtin_overload_row(tc, call, row, arg0_type, &resolved)) {
            continue;
        }
        WGSLTypeInfo *result = result_from_builtin_return(
            tc, row->ret, arg0_type, fallback, resolved);
        if (builtin_requires_concrete_result(name, name_len)) {
            result = wgsl_type_concretize(tc->types, result);
        }
        *out_type = result;
        return 1;
    }
    if (builtin_name_eq(name, name_len, "mix")) {
        WGSLTypeInfo *shape = resolve_t_family_prefix(
            tc, call, "mix", PF_FLOAT_T, 2, 3, BSHAPE_SCALAR_OR_VEC);
        check_mix_call(tc, call, name, name_len, shape ? shape : arg0_type);
    } else if (builtin_name_eq(name, name_len, "ldexp")) {
        WGSLTypeInfo *shape = resolve_t_family_prefix(
            tc, call, "ldexp", PF_FLOAT_T, 1, 2, BSHAPE_SCALAR_OR_VEC);
        check_ldexp_call(tc, call, name, name_len, shape ? shape : arg0_type);
    } else if (builtin_name_eq(name, name_len, "arrayLength")) {
        check_array_length_call(tc, call, arg0_type);
    }
    /* §3.4 — overload diagnostic.  Prefer the full §6.1.3 step-by-step
     * explanation when the builtin has a parametric T-family; otherwise
     * fall back to listing the concrete argument types. */
    {
        const WGSLBuiltinEntry *be = find_builtin_entry(name, name_len);
        int explained = 0;
        if (be && be->fam != PF_NONE && be->arity > 0 &&
            call && call->child_count >= 1u + (uint32_t)be->arity)
        {
            int arity = (int)be->arity;
            if (arity > WGSL_OL_MAX_PARAMS) arity = WGSL_OL_MAX_PARAMS;
            WGSLTypeInfo *args[WGSL_OL_MAX_PARAMS] = {0};
            int arg_const[WGSL_OL_MAX_PARAMS] = {0};
            int ok = 1;
            for (int i = 0; i < arity; i++) {
                WGSLNode *arg = call->children[1 + i];
                WGSLTypeInfo *t = builtin_arg_type(tc, call, i);
                if (!t) { ok = 0; break; }
                args[i] = t;
                arg_const[i] = is_const_expr(arg);
            }
            if (ok) {
                WGSLOverloadCandidate cands[WGSL_OL_MAX_CANDS];
                int nc = build_t_candidates(tc, be->fam, args, arity, cands);
                if (nc > 0) {
                    char what[64];
                    size_t wl = name_len < sizeof what - 1 ? name_len : sizeof what - 1;
                    memcpy(what, name, wl); what[wl] = '\0';
                    /* Only use resolve_overload for its *failure* explanation.
                     * A parametric success must NOT accept the call when the
                     * exact §17 row table already rejected it (e.g. sign(u32),
                     * normalize(f32), shuffle mask types). */
                    int picked = resolve_overload(
                        tc, cands, nc, args, arg_const, arity, what, call);
                    if (picked < 0) {
                        explained = 1; /* multi-line diag already emitted */
                    }
                    /* picked >= 0: fall through to the arguments-only message */
                }
            }
        }
        if (!explained) {
            char msg[384];
            size_t u = 0;
            msg[0] = '\0';
            ol_append(msg, sizeof msg, &u, "no matching overload for '");
            if (name && name_len) {
                size_t n = name_len < 64 ? name_len : 64;
                if (u + n + 1 < sizeof msg) {
                    memcpy(msg + u, name, n);
                    u += n;
                    msg[u] = '\0';
                }
            }
            ol_append(msg, sizeof msg, &u, "'\n  arguments: (");
            int nargs = call && call->child_count > 1
                ? (int)call->child_count - 1 : 0;
            for (int i = 0; i < nargs && i < 6; i++) {
                if (i) ol_append(msg, sizeof msg, &u, ", ");
                WGSLTypeInfo *t = builtin_arg_type(tc, call, i);
                ol_append_type(msg, sizeof msg, &u, t);
            }
            if (nargs > 6) ol_append(msg, sizeof msg, &u, ", …");
            ol_append(msg, sizeof msg, &u, ")");
            ol_append(msg, sizeof msg, &u,
                      "\n  no WGSL §17 overload accepts this argument shape");
            wgsl_tc_error(tc, call, "%s", msg);
        }
    }
    *out_type = fallback;
    return 1;
}
