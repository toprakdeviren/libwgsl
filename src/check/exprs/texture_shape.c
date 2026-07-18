/**
 * @file texture_shape.c — texture/sampler shape helpers
 */
#include "internal/exprs_priv.h"

int builtin_name_eq(const char *name, uint32_t len, const char *lit) {
    size_t n = strlen(lit);
    return len == n && memcmp(name, lit, n) == 0;
}

int builtin_arg_count(const WGSLNode *call) {
    return call && call->child_count > 0 ? (int)call->child_count - 1 : 0;
}

int builtin_arg_count_is(WGSLNode *call, int n) {
    return builtin_arg_count(call) == n;
}

WGSLTypeInfo *builtin_arg_type(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index)
{
    if (!call || arg_index < 0) return NULL;
    uint32_t child_index = (uint32_t)arg_index + 1u;
    if (child_index >= call->child_count) return NULL;
    return wgsl_tc_store_type_of(tc, call->children[child_index]);
}

int shape_width(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_VEC) return (int)t->width;
    if (wgsl_type_is_scalar(t)) return 1;
    return 0;
}

WGSLTypeInfo *shape_elem(const WGSLTypeInfo *t) {
    if (!t) return NULL;
    if (t->kind == WGSL_TYPE_VEC) return (WGSLTypeInfo *)t->ref;
    if (wgsl_type_is_scalar(t)) return (WGSLTypeInfo *)(uintptr_t)t;
    return NULL;
}

int type_converts_to(WGSLTypeInfo *from, WGSLTypeInfo *to) {
    return from && to && wgsl_type_conversion_rank(from, to) >= 0;
}

int expect_arg_count(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, int min_args, int max_args)
{
    int got = builtin_arg_count(call);
    if (got >= min_args && got <= max_args) return 1;
    if (min_args == max_args) {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d argument%s; got %d",
            (int)name_len, name, min_args, min_args == 1 ? "" : "s", got);
    } else {
        wgsl_tc_error(tc, call,
            "'%.*s' expects %d-%d arguments; got %d",
            (int)name_len, name, min_args, max_args, got);
    }
    return 0;
}

int expect_sampler_arg(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int comparison)
{
    if (t && t->kind == WGSL_TYPE_SAMPLER && ((int)t->width != 0) == comparison) {
        return 1;
    }
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' requires %s as the sampler argument; got %s",
        (int)name_len, name,
        comparison ? "sampler_comparison" : "sampler",
        type_label(t, got, sizeof got));
    return 0;
}

int expect_float_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *want = width == 1
        ? tc->types->t_f32
        : wgsl_type_vec(tc->types, (uint8_t)width, tc->types->t_f32);
    if (type_converts_to(t, want)) return 1;
    char got[96], exp[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        type_label(want, exp, sizeof exp));
    return 0;
}

int expect_integer_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem && wgsl_type_is_integer(elem) && shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %sinteger %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "an " : "a ",
        width == 1 ? "scalar" : "vector with matching texture dimensionality");
    return 0;
}

int expect_i32_shape(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *t, int width, const char *role)
{
    WGSLTypeInfo *elem = shape_elem(t);
    int ok = elem &&
             (elem == tc->types->t_abstract_int || elem == tc->types->t_i32) &&
             shape_width(t) == width;
    if (ok) return 1;
    char got[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(t, got, sizeof got),
        width == 1 ? "i32 or AbstractInt"
                   : "vecN<i32> or vecN<AbstractInt> with matching offset dimensionality");
    return 0;
}

int expect_value_convertible(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    WGSLTypeInfo *got_t, WGSLTypeInfo *want_t, const char *role)
{
    if (type_converts_to(got_t, want_t)) return 1;
    char got[96], want[96];
    wgsl_tc_error(tc, at,
        "'%.*s' %s has type %s; expected %s",
        (int)name_len, name, role,
        type_label(got_t, got, sizeof got),
        type_label(want_t, want, sizeof want));
    return 0;
}

int eval_const_int_silent(
    WGSLTypeChecker *tc, WGSLNode *n, int64_t *out)
{
    if (!tc || !tc->cev || !tc->diag || !n || !out) return 0;
    size_t saved_count = tc->diag->count;
    int saved_errors = tc->diag->error_count;
    int saved_cev = tc->cev->store.had_error;
    WGSLValue v = {0};
    int ok = wgsl_consteval_expr(tc->cev, n, &v) && v.kind == WGSL_VAL_INT;
    tc->diag->count = saved_count;
    tc->diag->error_count = saved_errors;
    tc->cev->store.had_error = saved_cev;
    if (!ok) return 0;
    *out = v.u.i;
    return 1;
}

int collect_const_int_components(
    WGSLTypeChecker *tc, WGSLNode *n, int width, int64_t *out)
{
    if (!n || !out || width < 1 || width > 4) return 0;
    if (width == 1) return eval_const_int_silent(tc, n, &out[0]);
    if (n->kind != WGSL_NODE_EXPR_CALL || n->child_count < 2) return 0;

    int arg_count = (int)n->child_count - 1;
    if (arg_count == 1) {
        int64_t v = 0;
        if (!eval_const_int_silent(tc, n->children[1], &v)) return 0;
        for (int i = 0; i < width; i++) out[i] = v;
        return 1;
    }
    if (arg_count != width) return 0;
    for (int i = 0; i < width; i++) {
        if (!eval_const_int_silent(tc, n->children[1 + i], &out[i])) {
            return 0;
        }
    }
    return 1;
}

int check_const_int_range(
    WGSLTypeChecker *tc, WGSLNode *at, const char *name, uint32_t name_len,
    int64_t lo, int64_t hi, const char *role)
{
    int64_t v = 0;
    if (!eval_const_int_silent(tc, at, &v)) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s must be a const-expression",
            (int)name_len, name, role);
        return 0;
    }
    if (v < lo || v > hi) {
        wgsl_tc_error(tc, at,
            "'%.*s' %s value %lld is outside [%lld, %lld]",
            (int)name_len, name, role,
            (long long)v, (long long)lo, (long long)hi);
        return 0;
    }
    return 1;
}

int texture_is_depth(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_DEPTH_2D &&
           d <= WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

int texture_is_storage(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_STORAGE_1D && d <= WGSL_TEX_DIM_STORAGE_3D;
}

int texture_is_sampled(WGSLTextureDim d) {
    return d >= WGSL_TEX_DIM_1D && d <= WGSL_TEX_DIM_MULTISAMPLED_2D;
}

int texture_is_multisampled(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_MULTISAMPLED_2D ||
           d == WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D;
}

int texture_is_arrayed(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_STORAGE_2D_ARRAY;
}

int texture_is_gather_dim(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_CUBE ||
           d == WGSL_TEX_DIM_CUBE_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY ||
           d == WGSL_TEX_DIM_DEPTH_CUBE ||
           d == WGSL_TEX_DIM_DEPTH_CUBE_ARRAY;
}

int texture_allows_offset(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_2D ||
           d == WGSL_TEX_DIM_2D_ARRAY ||
           d == WGSL_TEX_DIM_3D ||
           d == WGSL_TEX_DIM_DEPTH_2D ||
           d == WGSL_TEX_DIM_DEPTH_2D_ARRAY;
}

int texture_offset_width(WGSLTextureDim d) {
    return d == WGSL_TEX_DIM_3D ? 3 : 2;
}

int texture_coord_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_CUBE:
    case WGSL_TEX_DIM_CUBE_ARRAY:
    case WGSL_TEX_DIM_DEPTH_CUBE:
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

int texture_dimensions_width(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:
    case WGSL_TEX_DIM_STORAGE_1D:
        return 1;
    case WGSL_TEX_DIM_3D:
    case WGSL_TEX_DIM_STORAGE_3D:
        return 3;
    default:
        return 2;
    }
}

WGSLTypeInfo *storage_texture_elem(
    WGSLTypeChecker *tc, WGSLTexelFormat f)
{
    switch (f) {
    case WGSL_TEX_FORMAT_R8UINT:
    case WGSL_TEX_FORMAT_R16UINT:
    case WGSL_TEX_FORMAT_RG8UINT:
    case WGSL_TEX_FORMAT_R32UINT:
    case WGSL_TEX_FORMAT_RG16UINT:
    case WGSL_TEX_FORMAT_RGBA8UINT:
    case WGSL_TEX_FORMAT_RG32UINT:
    case WGSL_TEX_FORMAT_RGBA16UINT:
    case WGSL_TEX_FORMAT_RGBA32UINT:
    case WGSL_TEX_FORMAT_RGB10A2UINT:
        return tc->types->t_u32;
    case WGSL_TEX_FORMAT_R8SINT:
    case WGSL_TEX_FORMAT_R16SINT:
    case WGSL_TEX_FORMAT_RG8SINT:
    case WGSL_TEX_FORMAT_R32SINT:
    case WGSL_TEX_FORMAT_RG16SINT:
    case WGSL_TEX_FORMAT_RGBA8SINT:
    case WGSL_TEX_FORMAT_RG32SINT:
    case WGSL_TEX_FORMAT_RGBA16SINT:
    case WGSL_TEX_FORMAT_RGBA32SINT:
        return tc->types->t_i32;
    default:
        return tc->types->t_f32;
    }
}

WGSLTypeInfo *texture_value_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureSampleCompare") ||
        builtin_name_eq(name, name_len, "textureSampleCompareLevel"))
    {
        return tc->types->t_f32;
    }
    if (builtin_name_eq(name, name_len, "textureGatherCompare"))
    {
        return wgsl_type_vec(tc->types, 4, tc->types->t_f32);
    }
    if (builtin_name_eq(name, name_len, "textureGather")) {
        WGSLTypeInfo *elem = texture_is_storage(dim)
            ? storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len)
            : (WGSLTypeInfo *)tex->ref;
        if (!elem || texture_is_depth(dim) || dim == WGSL_TEX_DIM_EXTERNAL) {
            elem = tc->types->t_f32;
        }
        return wgsl_type_vec(tc->types, 4, elem);
    }
    if (texture_is_depth(dim)) {
        return tc->types->t_f32;
    }
    WGSLTypeInfo *elem = NULL;
    if (texture_is_storage(dim)) {
        elem = storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len);
    } else if (dim == WGSL_TEX_DIM_EXTERNAL) {
        elem = tc->types->t_f32;
    } else {
        elem = (WGSLTypeInfo *)tex->ref;
        if (!elem) elem = tc->types->t_f32;
    }
    return wgsl_type_vec(tc->types, 4, elem);
}

WGSLTypeInfo *texture_builtin_result(
    WGSLTypeChecker *tc, const char *name, uint32_t name_len, WGSLTypeInfo *tex)
{
    if (builtin_name_eq(name, name_len, "textureStore")) {
        return tc->types->t_void;
    }
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return NULL;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    if (builtin_name_eq(name, name_len, "textureDimensions")) {
        int w = texture_dimensions_width(dim);
        return w == 1 ? tc->types->t_u32
                      : wgsl_type_vec(tc->types, (uint8_t)w, tc->types->t_u32);
    }
    if (builtin_name_eq(name, name_len, "textureNumLayers") ||
        builtin_name_eq(name, name_len, "textureNumLevels") ||
        builtin_name_eq(name, name_len, "textureNumSamples"))
    {
        return tc->types->t_u32;
    }
    return texture_value_result(tc, name, name_len, tex);
}

WGSLTypeInfo *texture_arg_type_for_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *arg0_type)
{
    if (builtin_name_eq(name, name_len, "textureGather") &&
        (!arg0_type || arg0_type->kind != WGSL_TYPE_TEXTURE))
    {
        return builtin_arg_type(tc, call, 1);
    }
    return arg0_type;
}

