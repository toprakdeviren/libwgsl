/**
 * @file format.c — type kind names and string formatting.
 */
#include "internal/types.h"
#include "internal/resolver.h"   /* WGSLAddressSpace for ptr/ref format */

#include <stdio.h>
#include <string.h>

/* Stringify. */

const char *wgsl_type_kind_name(WGSLTypeKind k) {
    switch (k) {
    case WGSL_TYPE_INVALID:        return "invalid";
    case WGSL_TYPE_VOID:           return "void";
    case WGSL_TYPE_BOOL:           return "bool";
    case WGSL_TYPE_ABSTRACT_INT:   return "AbstractInt";
    case WGSL_TYPE_ABSTRACT_FLOAT: return "AbstractFloat";
    case WGSL_TYPE_I32:            return "i32";
    case WGSL_TYPE_U32:            return "u32";
    case WGSL_TYPE_F32:            return "f32";
    case WGSL_TYPE_F16:            return "f16";
    case WGSL_TYPE_VEC:            return "vec";
    case WGSL_TYPE_MAT:            return "mat";
    case WGSL_TYPE_ATOMIC:         return "atomic";
    case WGSL_TYPE_ARRAY:          return "array";
    case WGSL_TYPE_STRUCT:         return "struct";
    case WGSL_TYPE_ATOMIC_CX_RESULT: return "__atomic_compare_exchange_result";
    case WGSL_TYPE_FREXP_RESULT:     return "__frexp_result";
    case WGSL_TYPE_MODF_RESULT:      return "__modf_result";
    case WGSL_TYPE_SAMPLER:        return "sampler";
    case WGSL_TYPE_TEXTURE:        return "texture";
    case WGSL_TYPE_PTR:            return "ptr";
    case WGSL_TYPE_REF:            return "ref";
    case WGSL_TYPE_KIND_COUNT:     break;
    }
    return "<unknown>";
}

static const char *texture_dim_name(WGSLTextureDim d) {
    switch (d) {
    case WGSL_TEX_DIM_1D:                   return "texture_1d";
    case WGSL_TEX_DIM_2D:                   return "texture_2d";
    case WGSL_TEX_DIM_2D_ARRAY:             return "texture_2d_array";
    case WGSL_TEX_DIM_3D:                   return "texture_3d";
    case WGSL_TEX_DIM_CUBE:                 return "texture_cube";
    case WGSL_TEX_DIM_CUBE_ARRAY:           return "texture_cube_array";
    case WGSL_TEX_DIM_MULTISAMPLED_2D:      return "texture_multisampled_2d";
    case WGSL_TEX_DIM_EXTERNAL:             return "texture_external";
    case WGSL_TEX_DIM_DEPTH_2D:             return "texture_depth_2d";
    case WGSL_TEX_DIM_DEPTH_2D_ARRAY:       return "texture_depth_2d_array";
    case WGSL_TEX_DIM_DEPTH_CUBE:           return "texture_depth_cube";
    case WGSL_TEX_DIM_DEPTH_CUBE_ARRAY:     return "texture_depth_cube_array";
    case WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D:return "texture_depth_multisampled_2d";
    case WGSL_TEX_DIM_STORAGE_1D:           return "texture_storage_1d";
    case WGSL_TEX_DIM_STORAGE_2D:           return "texture_storage_2d";
    case WGSL_TEX_DIM_STORAGE_2D_ARRAY:     return "texture_storage_2d_array";
    case WGSL_TEX_DIM_STORAGE_3D:           return "texture_storage_3d";
    }
    return "texture";
}

static const char *texel_format_name(WGSLTexelFormat f) {
    switch (f) {
    case WGSL_TEX_FORMAT_NONE:       return "";
    case WGSL_TEX_FORMAT_R8UNORM:    return "r8unorm";
    case WGSL_TEX_FORMAT_R8SNORM:    return "r8snorm";
    case WGSL_TEX_FORMAT_R8UINT:     return "r8uint";
    case WGSL_TEX_FORMAT_R8SINT:     return "r8sint";
    case WGSL_TEX_FORMAT_R16UNORM:   return "r16unorm";
    case WGSL_TEX_FORMAT_R16SNORM:   return "r16snorm";
    case WGSL_TEX_FORMAT_R16UINT:    return "r16uint";
    case WGSL_TEX_FORMAT_R16SINT:    return "r16sint";
    case WGSL_TEX_FORMAT_R16FLOAT:   return "r16float";
    case WGSL_TEX_FORMAT_RG8UNORM:   return "rg8unorm";
    case WGSL_TEX_FORMAT_RG8SNORM:   return "rg8snorm";
    case WGSL_TEX_FORMAT_RG8UINT:    return "rg8uint";
    case WGSL_TEX_FORMAT_RG8SINT:    return "rg8sint";
    case WGSL_TEX_FORMAT_R32UINT:    return "r32uint";
    case WGSL_TEX_FORMAT_R32SINT:    return "r32sint";
    case WGSL_TEX_FORMAT_R32FLOAT:   return "r32float";
    case WGSL_TEX_FORMAT_RG16UNORM:  return "rg16unorm";
    case WGSL_TEX_FORMAT_RG16SNORM:  return "rg16snorm";
    case WGSL_TEX_FORMAT_RG16UINT:   return "rg16uint";
    case WGSL_TEX_FORMAT_RG16SINT:   return "rg16sint";
    case WGSL_TEX_FORMAT_RG16FLOAT:  return "rg16float";
    case WGSL_TEX_FORMAT_RGBA8UNORM: return "rgba8unorm";
    case WGSL_TEX_FORMAT_RGBA8UNORM_SRGB: return "rgba8unorm_srgb";
    case WGSL_TEX_FORMAT_RGBA8SNORM: return "rgba8snorm";
    case WGSL_TEX_FORMAT_RGBA8UINT:  return "rgba8uint";
    case WGSL_TEX_FORMAT_RGBA8SINT:  return "rgba8sint";
    case WGSL_TEX_FORMAT_BGRA8UNORM: return "bgra8unorm";
    case WGSL_TEX_FORMAT_BGRA8UNORM_SRGB: return "bgra8unorm_srgb";
    case WGSL_TEX_FORMAT_RGB10A2UNORM: return "rgb10a2unorm";
    case WGSL_TEX_FORMAT_RGB10A2UINT: return "rgb10a2uint";
    case WGSL_TEX_FORMAT_RG11B10FLOAT: return "rg11b10float";
    case WGSL_TEX_FORMAT_RG11B10UFLOAT: return "rg11b10ufloat";
    case WGSL_TEX_FORMAT_RG32UINT:   return "rg32uint";
    case WGSL_TEX_FORMAT_RG32SINT:   return "rg32sint";
    case WGSL_TEX_FORMAT_RG32FLOAT:  return "rg32float";
    case WGSL_TEX_FORMAT_RGBA16UNORM: return "rgba16unorm";
    case WGSL_TEX_FORMAT_RGBA16SNORM: return "rgba16snorm";
    case WGSL_TEX_FORMAT_RGBA16UINT: return "rgba16uint";
    case WGSL_TEX_FORMAT_RGBA16SINT: return "rgba16sint";
    case WGSL_TEX_FORMAT_RGBA16FLOAT:return "rgba16float";
    case WGSL_TEX_FORMAT_RGBA32UINT: return "rgba32uint";
    case WGSL_TEX_FORMAT_RGBA32SINT: return "rgba32sint";
    case WGSL_TEX_FORMAT_RGBA32FLOAT:return "rgba32float";
    }
    return "";
}

static const char *access_mode_name(WGSLAccessMode a) {
    switch (a) {
    case WGSL_ACCESS_NONE:       return "";
    case WGSL_ACCESS_READ:       return "read";
    case WGSL_ACCESS_WRITE:      return "write";
    case WGSL_ACCESS_READ_WRITE: return "read_write";
    }
    return "";
}

static const char *address_space_name(WGSLAddressSpace a) {
    switch (a) {
    case WGSL_AS_NONE:          return "";
    case WGSL_AS_FUNCTION:      return "function";
    case WGSL_AS_PRIVATE:       return "private";
    case WGSL_AS_WORKGROUP:     return "workgroup";
    case WGSL_AS_UNIFORM:       return "uniform";
    case WGSL_AS_STORAGE:       return "storage";
    case WGSL_AS_HANDLE:        return "handle";
    case WGSL_AS_PUSH_CONSTANT: return "push_constant";
    case WGSL_AS_IMMEDIATE:     return "immediate";
    }
    return "";
}

/* Exported thin wrappers over the private name tables — reused by the
 * reflection JSON emitter (module_json.inc) so the texel-format /
 * access-mode / address-space spellings can't drift out of sync. */
const char *wgsl_texel_format_name(WGSLTexelFormat f)   { return texel_format_name(f); }
const char *wgsl_access_mode_name(WGSLAccessMode a)     { return access_mode_name(a); }
const char *wgsl_address_space_name(WGSLAddressSpace a) { return address_space_name(a); }

/* Append-formatter that respects `cap`.  Returns required length
 * excluding NUL (snprintf semantics); state.written holds bytes
 * actually placed. */
typedef struct { char *buf; size_t cap; size_t total; } Fmt;

static void fmt_str(Fmt *f, const char *s) {
    while (*s) {
        if (f->total + 1 < f->cap) f->buf[f->total] = *s;
        f->total++;
        s++;
    }
}
static void fmt_uint(Fmt *f, unsigned int v) {
    char tmp[16];
    int n = snprintf(tmp, sizeof tmp, "%u", v);
    if (n < 0) return;
    for (int i = 0; i < n; i++) {
        if (f->total + 1 < f->cap) f->buf[f->total] = tmp[i];
        f->total++;
    }
}

static void fmt_type(Fmt *f, const WGSLTypeInfo *t) {
    if (!t) { fmt_str(f, "(null)"); return; }
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_VEC:
        fmt_str(f, "vec");
        fmt_uint(f, t->width);
        fmt_str(f, "<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_MAT:
        fmt_str(f, "mat");
        fmt_uint(f, t->width);
        fmt_str(f, "x");
        fmt_uint(f, t->rows);
        fmt_str(f, "<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_ATOMIC:
        fmt_str(f, "atomic<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_ARRAY:
        fmt_str(f, "array<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        if (t->array_len != 0) {
            fmt_str(f, ", ");
            fmt_uint(f, t->array_len);
        }
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_ATOMIC_CX_RESULT:
        fmt_str(f, "__atomic_compare_exchange_result<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_FREXP_RESULT:
        fmt_str(f, "__frexp_result<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_MODF_RESULT:
        fmt_str(f, "__modf_result<");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ">");
        break;
    case WGSL_TYPE_SAMPLER:
        fmt_str(f, t->width ? "sampler_comparison" : "sampler");
        break;
    case WGSL_TYPE_TEXTURE: {
        WGSLTextureDim dim = (WGSLTextureDim)t->width;
        fmt_str(f, texture_dim_name(dim));
        /* Storage textures: <format, access>.
         * Sampled / multisampled textures: <sampled_type>.
         * Depth + external textures: no template args. */
        if (dim >= WGSL_TEX_DIM_STORAGE_1D && dim <= WGSL_TEX_DIM_STORAGE_3D) {
            fmt_str(f, "<");
            fmt_str(f, texel_format_name((WGSLTexelFormat)t->array_len));
            fmt_str(f, ", ");
            fmt_str(f, access_mode_name((WGSLAccessMode)t->rows));
            fmt_str(f, ">");
        } else if (t->ref) {
            fmt_str(f, "<");
            fmt_type(f, (const WGSLTypeInfo *)t->ref);
            fmt_str(f, ">");
        }
        break;
    }
    case WGSL_TYPE_PTR:
    case WGSL_TYPE_REF: {
        WGSLAddressSpace as = (WGSLAddressSpace)(t->flags & 0xFFu);
        WGSLAccessMode am = (WGSLAccessMode)((t->flags >> 8) & 0xFFu);
        fmt_str(f, t->kind == WGSL_TYPE_PTR ? "ptr<" : "ref<");
        fmt_str(f, address_space_name(as));
        fmt_str(f, ", ");
        fmt_type(f, (const WGSLTypeInfo *)t->ref);
        fmt_str(f, ", ");
        fmt_str(f, access_mode_name(am));
        fmt_str(f, ">");
        break;
    }
    default:
        fmt_str(f, wgsl_type_kind_name((WGSLTypeKind)t->kind));
        break;
    }
}

size_t wgsl_type_format(const WGSLTypeInfo *t, char *buf, size_t cap) {
    Fmt f;
    f.buf = buf;
    f.cap = cap;
    f.total = 0;
    fmt_type(&f, t);
    if (cap > 0) {
        size_t end = f.total < cap ? f.total : cap - 1;
        f.buf[end] = '\0';
    }
    return f.total;
}
