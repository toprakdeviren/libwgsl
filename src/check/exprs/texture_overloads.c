/**
 * @file texture_overloads.c — expression typing module (was exprs/texture_overloads.inc).
 */
#include "internal/exprs_priv.h"

void check_texture_offset_arg(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim, int arg_index)
{
    if (arg_index < 0 || (uint32_t)(1 + arg_index) >= call->child_count) return;
    WGSLNode *arg = call->children[1 + arg_index];
    if (!texture_allows_offset(dim)) {
        wgsl_tc_error(tc, arg,
            "'%.*s' does not accept an offset for this texture shape",
            (int)name_len, name);
        return;
    }

    int width = texture_offset_width(dim);
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    expect_i32_shape(tc, arg, name, name_len, t, width, "offset");
    if (!is_const_expr(arg)) {
        wgsl_tc_error(tc, arg,
            "'%.*s' offset must be a const-expression",
            (int)name_len, name);
        return;
    }

    int64_t vals[4] = {0};
    if (collect_const_int_components(tc, arg, width, vals)) {
        for (int i = 0; i < width; i++) {
            if (vals[i] < -8 || vals[i] > 7) {
                wgsl_tc_error(tc, arg,
                    "'%.*s' offset component %d value %lld is outside [-8, 7]",
                    (int)name_len, name, i, (long long)vals[i]);
                return;
            }
        }
    }
}

void check_texture_sample_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim)
{
    int compare =
        builtin_name_eq(name, name_len, "textureSampleCompare") ||
        builtin_name_eq(name, name_len, "textureSampleCompareLevel");
    int base_clamp = builtin_name_eq(name, name_len, "textureSampleBaseClampToEdge");
    int sample_bias = builtin_name_eq(name, name_len, "textureSampleBias");
    int sample_grad = builtin_name_eq(name, name_len, "textureSampleGrad");
    int sample_level = builtin_name_eq(name, name_len, "textureSampleLevel");
    int has_array = texture_is_arrayed(dim);

    if (base_clamp) {
        if (!expect_arg_count(tc, call, name, name_len, 3, 3)) return;
        if (!(dim == WGSL_TEX_DIM_2D || dim == WGSL_TEX_DIM_EXTERNAL)) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires texture_2d<f32> or texture_external; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        if (dim == WGSL_TEX_DIM_2D && tex->ref != tc->types->t_f32) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires texture_2d<f32> or texture_external; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else if (compare) {
        int min_args = has_array ? 5 : 4;
        int max_args = min_args + 1;
        if (!expect_arg_count(tc, call, name, name_len, min_args, max_args)) return;
        if (!texture_is_depth(dim) || texture_is_multisampled(dim)) {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-multisampled depth texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else {
        int min_args = has_array ? 4 : 3;
        int max_args = min_args + 1;
        if (sample_grad) {
            min_args += 2;
            max_args += 2;
        } else if (sample_bias || sample_level)
        {
            min_args += 1;
            max_args += 1;
        }
        if (!expect_arg_count(tc, call, name, name_len, min_args, max_args)) return;
        if (!(texture_is_sampled(dim) || texture_is_depth(dim)) ||
            texture_is_multisampled(dim) ||
            dim == WGSL_TEX_DIM_EXTERNAL)
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a sampled or depth texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        if ((sample_bias || sample_grad) &&
            (!texture_is_sampled(dim) || texture_is_depth(dim) ||
             dim == WGSL_TEX_DIM_1D || texture_is_multisampled(dim)))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-depth sampled 2d/2d-array/3d/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    }

    WGSLTypeInfo *sampler = builtin_arg_type(tc, call, 1);
    expect_sampler_arg(tc, call->children[2], name, name_len, sampler, compare);
    expect_float_shape(tc, call->children[3], name, name_len,
                       builtin_arg_type(tc, call, 2),
                       texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[4], name, name_len,
                             builtin_arg_type(tc, call, 3), 1,
                             "array index");
    }
    int next_arg = has_array ? 4 : 3;
    if (compare) {
        expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg), 1,
                           "depth reference");
        if (builtin_arg_count(call) > next_arg + 1) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 1);
        }
    } else if (sample_bias || sample_level) {
        if (sample_level && texture_is_depth(dim)) {
            expect_integer_shape(tc, call->children[1 + next_arg], name, name_len,
                                 builtin_arg_type(tc, call, next_arg), 1,
                                 "level");
        } else {
            expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                               builtin_arg_type(tc, call, next_arg), 1,
                               sample_bias ? "bias" : "level");
        }
        if (builtin_arg_count(call) > next_arg + 1) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 1);
        }
    } else if (sample_grad) {
        expect_float_shape(tc, call->children[1 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg),
                           texture_coord_width(dim), "ddx");
        expect_float_shape(tc, call->children[2 + next_arg], name, name_len,
                           builtin_arg_type(tc, call, next_arg + 1),
                           texture_coord_width(dim), "ddy");
        if (builtin_arg_count(call) > next_arg + 2) {
            check_texture_offset_arg(tc, call, name, name_len, dim, next_arg + 2);
        }
    } else if (builtin_arg_count(call) > next_arg) {
        check_texture_offset_arg(tc, call, name, name_len, dim, next_arg);
    }
}

void check_texture_load_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim)
{
    int has_array = texture_is_arrayed(dim);
    int base = has_array ? 3 : 2; /* texture + coords [+ array_index] */
    int want = base;
    if (texture_is_multisampled(dim)) want += 1;       /* sample_index */
    else if (dim != WGSL_TEX_DIM_EXTERNAL &&
             !texture_is_storage(dim)) want += 1;      /* mip level */
    if (!expect_arg_count(tc, call, name, name_len, want, want)) return;

    expect_integer_shape(tc, call->children[2], name, name_len,
                         builtin_arg_type(tc, call, 1),
                         texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[3], name, name_len,
                             builtin_arg_type(tc, call, 2), 1,
                             "array index");
    }
}

void check_texture_store_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim)
{
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 4 : 3;
    if (!expect_arg_count(tc, call, name, name_len, want, want)) return;
    if (!texture_is_storage(dim)) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a storage texture; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
        return;
    }
    if ((WGSLAccessMode)tex->rows == WGSL_ACCESS_READ) {
        wgsl_tc_error(tc, call,
            "'%.*s' requires a write or read_write storage texture",
            (int)name_len, name);
    }
    expect_integer_shape(tc, call->children[2], name, name_len,
                         builtin_arg_type(tc, call, 1),
                         texture_coord_width(dim), "coordinates");
    int value_arg = has_array ? 3 : 2;
    if (has_array) {
        expect_integer_shape(tc, call->children[3], name, name_len,
                             builtin_arg_type(tc, call, 2), 1,
                             "array index");
    }
    WGSLTypeInfo *want_value = wgsl_type_vec(
        tc->types, 4, storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len));
    expect_value_convertible(tc, call->children[1 + value_arg], name, name_len,
                             builtin_arg_type(tc, call, value_arg),
                             want_value, "value argument");
}

void check_texture_query_call(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTextureDim dim)
{
    if (builtin_name_eq(name, name_len, "textureNumLayers")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (!texture_is_arrayed(dim)) {
            wgsl_tc_error(tc, call,
                "'%.*s' requires an array texture", (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureNumLevels")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (texture_is_storage(dim) || texture_is_multisampled(dim) ||
            dim == WGSL_TEX_DIM_EXTERNAL)
        {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a sampled or depth texture with mip levels",
                (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureNumSamples")) {
        expect_arg_count(tc, call, name, name_len, 1, 1);
        if (!texture_is_multisampled(dim)) {
            wgsl_tc_error(tc, call,
                "'%.*s' requires a multisampled texture",
                (int)name_len, name);
        }
        return;
    }
    if (builtin_name_eq(name, name_len, "textureDimensions")) {
        int want_min = 1;
        int want_max = (texture_is_storage(dim) ||
                        texture_is_multisampled(dim) ||
                        dim == WGSL_TEX_DIM_EXTERNAL) ? 1 : 2;
        expect_arg_count(tc, call, name, name_len, want_min, want_max);
    }
}

void check_texture_gather_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len,
    WGSLTypeInfo *tex, WGSLTextureDim dim, int component_form)
{
    int compare = builtin_name_eq(name, name_len, "textureGatherCompare");
    int has_array = texture_is_arrayed(dim);
    int tex_arg = component_form ? 1 : 0;
    int sampler_arg = tex_arg + 1;
    int coords_arg = tex_arg + 2;
    int array_arg = coords_arg + 1;

    if (compare) {
        int depth_ref_arg = has_array ? 4 : 3;
        int min_args = depth_ref_arg + 1;
        if (!expect_arg_count(tc, call, name, name_len, min_args, min_args + 1)) {
            return;
        }
        if (!texture_is_depth(dim) || texture_is_multisampled(dim) ||
            !texture_is_gather_dim(dim))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' requires a non-multisampled depth 2d/2d-array/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
        expect_sampler_arg(tc, call->children[1 + sampler_arg], name, name_len,
                           builtin_arg_type(tc, call, sampler_arg), 1);
        expect_float_shape(tc, call->children[1 + coords_arg], name, name_len,
                           builtin_arg_type(tc, call, coords_arg),
                           texture_coord_width(dim), "coordinates");
        if (has_array) {
            expect_integer_shape(tc, call->children[1 + array_arg], name, name_len,
                                 builtin_arg_type(tc, call, array_arg), 1,
                                 "array index");
        }
        expect_float_shape(tc, call->children[1 + depth_ref_arg], name, name_len,
                           builtin_arg_type(tc, call, depth_ref_arg), 1,
                           "depth reference");
        if (builtin_arg_count(call) > min_args) {
            check_texture_offset_arg(tc, call, name, name_len, dim, min_args);
        }
        return;
    }

    int min_args = (component_form ? 4 : 3) + (has_array ? 1 : 0);
    if (!expect_arg_count(tc, call, name, name_len, min_args, min_args + 1)) {
        return;
    }

    if (component_form) {
        WGSLTypeInfo *ct = builtin_arg_type(tc, call, 0);
        expect_integer_shape(tc, call->children[1], name, name_len,
                             ct, 1, "component");
        check_const_int_range(tc, call->children[1], name, name_len,
                              0, 3, "component");
        if (texture_is_depth(dim) || !texture_is_sampled(dim) ||
            texture_is_multisampled(dim) || dim == WGSL_TEX_DIM_EXTERNAL ||
            !texture_is_gather_dim(dim))
        {
            char got[96];
            wgsl_tc_error(tc, call,
                "'%.*s' component form requires a non-depth sampled 2d/2d-array/cube texture; got %s",
                (int)name_len, name, type_label(tex, got, sizeof got));
        }
    } else if (!texture_is_depth(dim) || texture_is_multisampled(dim) ||
               !texture_is_gather_dim(dim))
    {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' without a component argument requires a non-multisampled depth texture; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
    }

    expect_sampler_arg(tc, call->children[1 + sampler_arg], name, name_len,
                       builtin_arg_type(tc, call, sampler_arg), 0);
    expect_float_shape(tc, call->children[1 + coords_arg], name, name_len,
                       builtin_arg_type(tc, call, coords_arg),
                       texture_coord_width(dim), "coordinates");
    if (has_array) {
        expect_integer_shape(tc, call->children[1 + array_arg], name, name_len,
                             builtin_arg_type(tc, call, array_arg), 1,
                             "array index");
    }
    if (builtin_arg_count(call) > min_args) {
        check_texture_offset_arg(tc, call, name, name_len, dim, min_args);
    }
}

void check_texture_call(
    WGSLTypeChecker *tc, WGSLNode *call,
    const char *name, uint32_t name_len, WGSLTypeInfo *tex,
    int gather_component_form)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) {
        char got[96];
        wgsl_tc_error(tc, call,
            "'%.*s' requires a texture argument; got %s",
            (int)name_len, name, type_label(tex, got, sizeof got));
        return;
    }
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;

    if (builtin_name_eq(name, name_len, "textureLoad")) {
        check_texture_load_call(tc, call, name, name_len, dim);
    } else if (builtin_name_eq(name, name_len, "textureStore")) {
        check_texture_store_call(tc, call, name, name_len, tex, dim);
    } else if (builtin_name_eq(name, name_len, "textureDimensions") ||
               builtin_name_eq(name, name_len, "textureNumLayers") ||
               builtin_name_eq(name, name_len, "textureNumLevels") ||
               builtin_name_eq(name, name_len, "textureNumSamples"))
    {
        check_texture_query_call(tc, call, name, name_len, dim);
    } else if (builtin_name_eq(name, name_len, "textureGather") ||
               builtin_name_eq(name, name_len, "textureGatherCompare"))
    {
        check_texture_gather_call(
            tc, call, name, name_len, tex, dim, gather_component_form);
    } else {
        check_texture_sample_call(tc, call, name, name_len, tex, dim);
    }
}

int texture_silent_float_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *want = width == 1
        ? tc->types->t_f32
        : wgsl_type_vec(tc->types, (uint8_t)width, tc->types->t_f32);
    return type_converts_to(t, want);
}

int texture_silent_integer_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *elem = shape_elem(t);
    return elem && wgsl_type_is_integer(elem) && shape_width(t) == width;
}

int texture_silent_i32_shape(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int width)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    WGSLTypeInfo *elem = shape_elem(t);
    return elem &&
           (elem == tc->types->t_abstract_int || elem == tc->types->t_i32) &&
           shape_width(t) == width;
}

int texture_silent_sampler(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, int comparison)
{
    WGSLTypeInfo *t = builtin_arg_type(tc, call, arg_index);
    return t && t->kind == WGSL_TYPE_SAMPLER &&
           (((int)t->width != 0) == comparison);
}

int texture_silent_storage_value(
    WGSLTypeChecker *tc, WGSLNode *call, int arg_index, WGSLTypeInfo *tex)
{
    if (!tex || !texture_is_storage((WGSLTextureDim)tex->width)) return 0;
    WGSLTypeInfo *elem = storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len);
    WGSLTypeInfo *want = wgsl_type_vec(tc->types, 4, elem);
    return type_converts_to(builtin_arg_type(tc, call, arg_index), want);
}

int texture_row_elem_matches(
    WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return 0;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    switch ((WGSLTextureOverloadElem)row->elem) {
    case BTEXEL_ANY:
        return 1;
    case BTEXEL_F32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_f32;
    case BTEXEL_I32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_i32;
    case BTEXEL_U32:
        return !texture_is_storage(dim) && !texture_is_depth(dim) &&
               dim != WGSL_TEX_DIM_EXTERNAL && tex->ref == tc->types->t_u32;
    case BTEXEL_DEPTH:
        return texture_is_depth(dim);
    case BTEXEL_EXTERNAL:
        return dim == WGSL_TEX_DIM_EXTERNAL;
    case BTEXEL_STORAGE_ANY:
        return texture_is_storage(dim);
    case BTEXEL_STORAGE_F32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_f32;
    case BTEXEL_STORAGE_I32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_i32;
    case BTEXEL_STORAGE_U32:
        return texture_is_storage(dim) &&
               storage_texture_elem(tc, (WGSLTexelFormat)tex->array_len) == tc->types->t_u32;
    }
    return 0;
}

int texture_row_texture_matches(
    WGSLTypeChecker *tc, const WGSLTextureOverload *row, WGSLTypeInfo *tex)
{
    if (!tex || tex->kind != WGSL_TYPE_TEXTURE) return 0;
    if (tex->width != row->dim) return 0;
    if (row->access != WGSL_ACCESS_NONE && tex->rows != row->access) return 0;
    return texture_row_elem_matches(tc, row, tex);
}

int texture_match_array_index_if_needed(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index)
{
    return !texture_is_arrayed(dim) ||
           texture_silent_integer_shape(tc, call, arg_index, 1);
}

int texture_match_offset(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTextureDim dim, int arg_index)
{
    return texture_allows_offset(dim) &&
           texture_silent_i32_shape(tc, call, arg_index, texture_offset_width(dim));
}

int texture_match_sample_like(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row,
    WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int base_args = has_array ? 4 : 3;
    int next = has_array ? 4 : 3;

    if (!texture_silent_sampler(tc, call, 1,
            row->kind == BTOV_SAMPLE_COMPARE ||
            row->kind == BTOV_SAMPLE_COMPARE_OFFSET ||
            row->kind == BTOV_SAMPLE_COMPARE_LEVEL ||
            row->kind == BTOV_SAMPLE_COMPARE_LEVEL_OFFSET))
        return 0;
    if (!texture_silent_float_shape(tc, call, 2, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 3)) return 0;

    switch ((WGSLTextureOverloadKind)row->kind) {
    case BTOV_SAMPLE:
        return builtin_arg_count_is(call, base_args);
    case BTOV_SAMPLE_OFFSET:
        return builtin_arg_count_is(call, base_args + 1) &&
               texture_match_offset(tc, call, dim, next);
    case BTOV_SAMPLE_LEVEL:
    case BTOV_SAMPLE_BIAS:
        return builtin_arg_count_is(call, base_args + 1) &&
               (row->kind == BTOV_SAMPLE_LEVEL && texture_is_depth(dim)
                    ? texture_silent_integer_shape(tc, call, next, 1)
                    : texture_silent_float_shape(tc, call, next, 1));
    case BTOV_SAMPLE_LEVEL_OFFSET:
    case BTOV_SAMPLE_BIAS_OFFSET:
        return builtin_arg_count_is(call, base_args + 2) &&
               (row->kind == BTOV_SAMPLE_LEVEL_OFFSET && texture_is_depth(dim)
                    ? texture_silent_integer_shape(tc, call, next, 1)
                    : texture_silent_float_shape(tc, call, next, 1)) &&
               texture_match_offset(tc, call, dim, next + 1);
    case BTOV_SAMPLE_GRAD:
        return builtin_arg_count_is(call, base_args + 2) &&
               texture_silent_float_shape(tc, call, next, texture_coord_width(dim)) &&
               texture_silent_float_shape(tc, call, next + 1, texture_coord_width(dim));
    case BTOV_SAMPLE_GRAD_OFFSET:
        return builtin_arg_count_is(call, base_args + 3) &&
               texture_silent_float_shape(tc, call, next, texture_coord_width(dim)) &&
               texture_silent_float_shape(tc, call, next + 1, texture_coord_width(dim)) &&
               texture_match_offset(tc, call, dim, next + 2);
    case BTOV_SAMPLE_COMPARE:
    case BTOV_SAMPLE_COMPARE_LEVEL:
        return builtin_arg_count_is(call, base_args + 1) &&
               texture_silent_float_shape(tc, call, next, 1);
    case BTOV_SAMPLE_COMPARE_OFFSET:
    case BTOV_SAMPLE_COMPARE_LEVEL_OFFSET:
        return builtin_arg_count_is(call, base_args + 2) &&
               texture_silent_float_shape(tc, call, next, 1) &&
               texture_match_offset(tc, call, dim, next + 1);
    default:
        return 0;
    }
}

int texture_match_load(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 3 : 2;
    int extra = -1;
    if (texture_is_multisampled(dim)) {
        extra = want;
        want += 1;
    } else if (dim != WGSL_TEX_DIM_EXTERNAL && !texture_is_storage(dim)) {
        extra = want;
        want += 1;
    }
    if (!builtin_arg_count_is(call, want)) return 0;
    if (!texture_silent_integer_shape(tc, call, 1, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 2)) return 0;
    return extra < 0 || texture_silent_integer_shape(tc, call, extra, 1);
}

int texture_match_store(
    WGSLTypeChecker *tc, WGSLNode *call, WGSLTypeInfo *tex)
{
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int want = has_array ? 4 : 3;
    int value_arg = has_array ? 3 : 2;
    if (!builtin_arg_count_is(call, want)) return 0;
    if (!texture_silent_integer_shape(tc, call, 1, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, 2)) return 0;
    return texture_silent_storage_value(tc, call, value_arg, tex);
}

int texture_match_gather(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row)
{
    WGSLTextureOverloadKind kind = (WGSLTextureOverloadKind)row->kind;
    int component = kind == BTOV_GATHER_COMPONENT ||
                    kind == BTOV_GATHER_COMPONENT_OFFSET;
    int tex_arg = component ? 1 : 0;
    WGSLTypeInfo *tex = builtin_arg_type(tc, call, tex_arg);
    if (!texture_row_texture_matches(tc, row, tex)) return 0;
    WGSLTextureDim dim = (WGSLTextureDim)tex->width;
    int has_array = texture_is_arrayed(dim);
    int sampler_arg = component ? 2 : 1;
    int coords_arg = component ? 3 : 2;
    int array_arg = coords_arg + 1;
    int depth_ref_arg = has_array ? 4 : 3;
    int base_args =
        kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET
            ? depth_ref_arg + 1
            : (component ? 4 : 3) + (has_array ? 1 : 0);

    if (!builtin_arg_count_is(call,
            base_args + ((kind == BTOV_GATHER_COMPONENT_OFFSET ||
                          kind == BTOV_GATHER_DEPTH_OFFSET ||
                          kind == BTOV_GATHER_COMPARE_OFFSET) ? 1 : 0)))
        return 0;
    if (component && !texture_silent_integer_shape(tc, call, 0, 1)) return 0;
    if (!texture_silent_sampler(
            tc, call, sampler_arg,
            kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET))
        return 0;
    if (!texture_silent_float_shape(tc, call, coords_arg, texture_coord_width(dim))) {
        return 0;
    }
    if (!texture_match_array_index_if_needed(tc, call, dim, array_arg)) return 0;
    if ((kind == BTOV_GATHER_COMPARE || kind == BTOV_GATHER_COMPARE_OFFSET) &&
        !texture_silent_float_shape(tc, call, depth_ref_arg, 1))
    {
        return 0;
    }
    if (kind == BTOV_GATHER_COMPONENT_OFFSET ||
        kind == BTOV_GATHER_DEPTH_OFFSET ||
        kind == BTOV_GATHER_COMPARE_OFFSET)
    {
        return texture_match_offset(tc, call, dim, base_args);
    }
    return 1;
}

int match_texture_overload_row(
    WGSLTypeChecker *tc, WGSLNode *call, const WGSLTextureOverload *row,
    WGSLTypeInfo **out_type)
{
    WGSLTextureOverloadKind kind = (WGSLTextureOverloadKind)row->kind;
    int tex_arg =
        (kind == BTOV_GATHER_COMPONENT || kind == BTOV_GATHER_COMPONENT_OFFSET)
            ? 1 : 0;
    WGSLTypeInfo *tex = builtin_arg_type(tc, call, tex_arg);
    if (kind != BTOV_GATHER_COMPONENT &&
        kind != BTOV_GATHER_COMPONENT_OFFSET &&
        kind != BTOV_GATHER_DEPTH &&
        kind != BTOV_GATHER_DEPTH_OFFSET &&
        kind != BTOV_GATHER_COMPARE &&
        kind != BTOV_GATHER_COMPARE_OFFSET &&
        !texture_row_texture_matches(tc, row, tex))
    {
        return 0;
    }

    WGSLTextureDim dim = tex && tex->kind == WGSL_TYPE_TEXTURE
        ? (WGSLTextureDim)tex->width : WGSL_TEX_DIM_1D;
    int ok = 0;
    switch (kind) {
    case BTOV_SAMPLE:
    case BTOV_SAMPLE_OFFSET:
    case BTOV_SAMPLE_LEVEL:
    case BTOV_SAMPLE_LEVEL_OFFSET:
    case BTOV_SAMPLE_BIAS:
    case BTOV_SAMPLE_BIAS_OFFSET:
    case BTOV_SAMPLE_GRAD:
    case BTOV_SAMPLE_GRAD_OFFSET:
    case BTOV_SAMPLE_COMPARE:
    case BTOV_SAMPLE_COMPARE_OFFSET:
    case BTOV_SAMPLE_COMPARE_LEVEL:
    case BTOV_SAMPLE_COMPARE_LEVEL_OFFSET:
        ok = texture_match_sample_like(tc, call, row, tex);
        break;
    case BTOV_SAMPLE_BASE_CLAMP:
        ok = builtin_arg_count_is(call, 3) &&
             texture_silent_sampler(tc, call, 1, 0) &&
             texture_silent_float_shape(tc, call, 2, texture_coord_width(dim));
        break;
    case BTOV_LOAD:
        ok = texture_match_load(tc, call, tex);
        break;
    case BTOV_STORE:
        ok = texture_match_store(tc, call, tex);
        break;
    case BTOV_GATHER_COMPONENT:
    case BTOV_GATHER_COMPONENT_OFFSET:
    case BTOV_GATHER_DEPTH:
    case BTOV_GATHER_DEPTH_OFFSET:
    case BTOV_GATHER_COMPARE:
    case BTOV_GATHER_COMPARE_OFFSET:
        ok = texture_match_gather(tc, call, row);
        break;
    case BTOV_DIMENSIONS:
        ok = builtin_arg_count_is(call, 1);
        break;
    case BTOV_DIMENSIONS_LEVEL:
        ok = builtin_arg_count_is(call, 2) &&
             texture_silent_integer_shape(tc, call, 1, 1);
        break;
    case BTOV_NUM_LAYERS:
    case BTOV_NUM_LEVELS:
    case BTOV_NUM_SAMPLES:
        ok = builtin_arg_count_is(call, 1);
        break;
    }
    if (!ok) return 0;
    *out_type = texture_builtin_result(tc, row->name, (uint32_t)strlen(row->name), tex);
    return *out_type != NULL;
}

int resolve_generated_texture_overload(
    WGSLTypeChecker *tc, WGSLNode *call, const char *name, uint32_t name_len,
    WGSLTypeInfo **out_type)
{
    int saw_row = 0;
    for (size_t i = 0; i < WGSL_TEXTURE_OVERLOAD_COUNT; i++) {
        const WGSLTextureOverload *row = &kTextureOverloads[i];
        size_t l = strlen(row->name);
        if (l != name_len || memcmp(row->name, name, l) != 0) continue;
        saw_row = 1;
        WGSLTypeInfo *resolved = NULL;
        if (!match_texture_overload_row(tc, call, row, &resolved)) continue;
        *out_type = resolved;
        return 1;
    }
    return saw_row ? -1 : 0;
}

