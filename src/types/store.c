/**
 * @file store.c — type interning, constructors, layout, conversion rank.
 */
#include "internal/types.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_TYPES_INITIAL_CAPACITY 64
#define WGSL_TYPES_HTAB_EMPTY       ((size_t)-1)

/* Type construction + interning (open-addressed hash, §4.1). */

static uint64_t type_key_hash(
    WGSLTypeKind kind, uint8_t width, uint8_t rows,
    uint32_t array_len, uint32_t flags, uint32_t aux, void *ref)
{
    uint64_t h = 14695981039346656037ull;
    uint64_t parts[4];
    parts[0] = ((uint64_t)kind << 16) | ((uint64_t)width << 8) | rows;
    parts[1] = array_len;
    parts[2] = ((uint64_t)flags << 32) | aux;
    parts[3] = (uint64_t)(uintptr_t)ref;
    for (int i = 0; i < 4; i++) {
        h ^= parts[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int type_key_eq(const WGSLTypeInfo *t,
    WGSLTypeKind kind, uint8_t width, uint8_t rows,
    uint32_t array_len, uint32_t flags, uint32_t aux, void *ref)
{
    return t->kind == (uint16_t)kind && t->width == width && t->rows == rows
        && t->array_len == array_len && t->flags == flags && t->pad_ == aux
        && t->ref == ref;
}

static int store_htab_rehash(WGSLTypeStore *s, size_t new_cap) {
    if (new_cap < 16) new_cap = 16;
    size_t cap = 16;
    while (cap < new_cap) {
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    size_t *nt = (size_t *)malloc(cap * sizeof *nt);
    if (!nt) return 0;
    for (size_t i = 0; i < cap; i++) nt[i] = WGSL_TYPES_HTAB_EMPTY;
    for (size_t i = 0; i < s->count; i++) {
        WGSLTypeInfo *t = s->interned[i];
        uint64_t h = type_key_hash((WGSLTypeKind)t->kind, t->width, t->rows,
                                   t->array_len, t->flags, t->pad_, t->ref);
        size_t mask = cap - 1;
        size_t slot = (size_t)h & mask;
        while (nt[slot] != WGSL_TYPES_HTAB_EMPTY)
            slot = (slot + 1) & mask;
        nt[slot] = i;
    }
    free(s->htab);
    s->htab = nt;
    s->htab_cap = cap;
    return 1;
}

static int store_htab_insert(WGSLTypeStore *s, size_t idx) {
    if (!s->htab || s->htab_cap == 0) {
        if (!store_htab_rehash(s, 64)) return 0;
    }
    if (s->count * 10 >= s->htab_cap * 7) {
        if (!store_htab_rehash(s, s->htab_cap * 2)) return 0;
    }
    WGSLTypeInfo *t = s->interned[idx];
    uint64_t h = type_key_hash((WGSLTypeKind)t->kind, t->width, t->rows,
                               t->array_len, t->flags, t->pad_, t->ref);
    size_t mask = s->htab_cap - 1;
    size_t slot = (size_t)h & mask;
    while (s->htab[slot] != WGSL_TYPES_HTAB_EMPTY)
        slot = (slot + 1) & mask;
    s->htab[slot] = idx;
    return 1;
}

static int store_grow(WGSLTypeStore *s) {
    if (s->capacity > SIZE_MAX / 2) return 0;
    size_t new_cap = s->capacity ? s->capacity * 2 : WGSL_TYPES_INITIAL_CAPACITY;
    if (new_cap > SIZE_MAX / sizeof *s->interned) return 0;
    WGSLTypeInfo **g = (WGSLTypeInfo **)realloc(
        s->interned, new_cap * sizeof(*g));
    if (!g) return 0;
    s->interned = g;
    s->capacity = new_cap;
    return 1;
}

static WGSLTypeInfo *store_alloc(WGSLTypeStore *s) {
    WGSLTypeInfo *t = (WGSLTypeInfo *)wgsl_arena_calloc(
        s->arena, 1, sizeof(WGSLTypeInfo));
    if (!t) return NULL;
    if (s->count == s->capacity && !store_grow(s)) return NULL;
    s->interned[s->count++] = t;
    return t;
}

/* Call after filling a newly allocated type's key fields. */
static void store_publish(WGSLTypeStore *s, WGSLTypeInfo *t) {
    if (!s || !t || s->count == 0) return;
    size_t idx = s->count - 1;
    if (s->interned[idx] != t) {
        for (size_t i = 0; i < s->count; i++)
            if (s->interned[i] == t) { idx = i; break; }
    }
    (void)store_htab_insert(s, idx);
}

static WGSLTypeInfo *store_find(
    const WGSLTypeStore *s,
    WGSLTypeKind kind, uint8_t width, uint8_t rows,
    uint32_t array_len, uint32_t flags, uint32_t aux, void *ref)
{
    if (s->htab && s->htab_cap) {
        uint64_t h = type_key_hash(kind, width, rows, array_len, flags, aux, ref);
        size_t mask = s->htab_cap - 1;
        size_t slot = (size_t)h & mask;
        for (size_t n = 0; n < s->htab_cap; n++) {
            size_t idx = s->htab[slot];
            if (idx == WGSL_TYPES_HTAB_EMPTY) return NULL;
            if (idx < s->count) {
                WGSLTypeInfo *t = s->interned[idx];
                if (type_key_eq(t, kind, width, rows, array_len, flags, aux, ref))
                    return t;
            }
            slot = (slot + 1) & mask;
        }
        return NULL;
    }
    for (size_t i = 0; i < s->count; i++) {
        WGSLTypeInfo *t = s->interned[i];
        if (type_key_eq(t, kind, width, rows, array_len, flags, aux, ref))
            return t;
    }
    return NULL;
}

static WGSLTypeInfo *make_scalar(WGSLTypeStore *s, WGSLTypeKind k) {
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)k;
    store_publish(s, t);
    return t;
}

int wgsl_types_init(WGSLTypeStore *s, WGSLArena *arena) {
    memset(s, 0, sizeof *s);
    s->arena = arena;

    s->t_void           = make_scalar(s, WGSL_TYPE_VOID);
    s->t_bool           = make_scalar(s, WGSL_TYPE_BOOL);
    s->t_abstract_int   = make_scalar(s, WGSL_TYPE_ABSTRACT_INT);
    s->t_abstract_float = make_scalar(s, WGSL_TYPE_ABSTRACT_FLOAT);
    s->t_i32            = make_scalar(s, WGSL_TYPE_I32);
    s->t_u32            = make_scalar(s, WGSL_TYPE_U32);
    s->t_f32            = make_scalar(s, WGSL_TYPE_F32);
    s->t_f16            = make_scalar(s, WGSL_TYPE_F16);

    if (!s->t_void || !s->t_bool || !s->t_abstract_int ||
        !s->t_abstract_float || !s->t_i32 || !s->t_u32 ||
        !s->t_f32 || !s->t_f16)
    {
        wgsl_types_destroy(s);
        return 0;
    }
    return 1;
}

void wgsl_types_destroy(WGSLTypeStore *s) {
    free(s->interned);
    free(s->htab);
    memset(s, 0, sizeof *s);
}

WGSLTypeInfo *wgsl_type_scalar(const WGSLTypeStore *s, WGSLTypeKind k) {
    switch (k) {
    case WGSL_TYPE_VOID:           return s->t_void;
    case WGSL_TYPE_BOOL:           return s->t_bool;
    case WGSL_TYPE_ABSTRACT_INT:   return s->t_abstract_int;
    case WGSL_TYPE_ABSTRACT_FLOAT: return s->t_abstract_float;
    case WGSL_TYPE_I32:            return s->t_i32;
    case WGSL_TYPE_U32:            return s->t_u32;
    case WGSL_TYPE_F32:            return s->t_f32;
    case WGSL_TYPE_F16:            return s->t_f16;
    default:                       return NULL;
    }
}

WGSLTypeInfo *wgsl_type_vec(WGSLTypeStore *s, uint8_t width, WGSLTypeInfo *elem) {
    if (!elem) return NULL;
    if (width < 2 || width > 4) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_VEC, width, 0, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind  = (uint16_t)WGSL_TYPE_VEC;
    t->width = width;
    t->ref   = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_mat(
    WGSLTypeStore *s, uint8_t cols, uint8_t rows, WGSLTypeInfo *elem)
{
    if (!elem) return NULL;
    if (cols < 2 || cols > 4) return NULL;
    if (rows < 2 || rows > 4) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_MAT, cols, rows, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind  = (uint16_t)WGSL_TYPE_MAT;
    t->width = cols;
    t->rows  = rows;
    t->ref   = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_atomic(WGSLTypeStore *s, WGSLTypeInfo *elem) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ATOMIC, 0, 0, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)WGSL_TYPE_ATOMIC;
    t->ref  = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_atomic_cx_result(WGSLTypeStore *s, WGSLTypeInfo *elem) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ATOMIC_CX_RESULT, 0, 0, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)WGSL_TYPE_ATOMIC_CX_RESULT;
    t->ref  = elem;
    store_publish(s, t);
    return t;
}

/* §17 — `__frexp_result_T` { fract: T, exp: I } where I is i32 / vecN<i32>
 * matching T's shape, and `__modf_result_T` { fract: T, whole: T } where
 * T is the scalar/vec float passed to frexp / modf. */
WGSLTypeInfo *wgsl_type_frexp_result(WGSLTypeStore *s, WGSLTypeInfo *elem) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_FREXP_RESULT, 0, 0, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)WGSL_TYPE_FREXP_RESULT;
    t->ref  = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_modf_result(WGSLTypeStore *s, WGSLTypeInfo *elem) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_MODF_RESULT, 0, 0, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)WGSL_TYPE_MODF_RESULT;
    t->ref  = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_array(WGSLTypeStore *s, WGSLTypeInfo *elem, uint32_t length) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ARRAY, 0, 0, length, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind      = (uint16_t)WGSL_TYPE_ARRAY;
    t->ref       = elem;
    t->array_len = length;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_array_override(
    WGSLTypeStore *s, WGSLTypeInfo *elem, const void *override_sym)
{
    if (!elem) return NULL;
    if (!override_sym) return wgsl_type_array(s, elem, 0);
    uintptr_t key = (uintptr_t)override_sym;
    uint32_t key_lo = (uint32_t)(key & 0xFFFFFFFFu);
    uint32_t key_hi = 0;
#if UINTPTR_MAX > UINT32_MAX
    key_hi = (uint32_t)((key >> 32) & 0xFFFFFFFFu);
#endif
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ARRAY, 0, 0, 0, key_lo, key_hi, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind      = (uint16_t)WGSL_TYPE_ARRAY;
    t->ref       = elem;
    t->array_len = 0;
    t->flags     = key_lo;
    t->pad_      = key_hi;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_sampler(WGSLTypeStore *s, int comparison) {
    uint8_t w = comparison ? 1u : 0u;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_SAMPLER, w, 0, 0, 0, 0, NULL);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind  = (uint16_t)WGSL_TYPE_SAMPLER;
    t->width = w;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_texture(
    WGSLTypeStore *s,
    WGSLTextureDim dim,
    WGSLTypeInfo  *sampled_elem,
    WGSLTexelFormat format,
    WGSLAccessMode  access)
{
    uint8_t  w = (uint8_t)dim;
    uint8_t  r = (uint8_t)access;
    uint32_t a = (uint32_t)format;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_TEXTURE, w, r, a, 0, 0, sampled_elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind      = (uint16_t)WGSL_TYPE_TEXTURE;
    t->width     = w;
    t->rows      = r;
    t->array_len = a;
    t->ref       = sampled_elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_ptr(WGSLTypeStore *s, uint8_t as, WGSLTypeInfo *elem, WGSLAccessMode am) {
    if (!elem) return NULL;
    uint32_t flags = ((uint32_t)as) | (((uint32_t)am) << 8);
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_PTR, 0, 0, 0, flags, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind  = (uint16_t)WGSL_TYPE_PTR;
    t->flags = flags;
    t->ref   = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeInfo *wgsl_type_ref(WGSLTypeStore *s, uint8_t as, WGSLTypeInfo *elem, WGSLAccessMode am) {
    if (!elem) return NULL;
    uint32_t flags = ((uint32_t)as) | (((uint32_t)am) << 8);
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_REF, 0, 0, 0, flags, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind  = (uint16_t)WGSL_TYPE_REF;
    t->flags = flags;
    t->ref   = elem;
    store_publish(s, t);
    return t;
}

WGSLTypeLayout wgsl_type_get_layout(const WGSLTypeInfo *t) {
    WGSLTypeLayout l = { 4, 4 };
    if (!t) return l;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL: l.size = 1; l.align = 1; break;
    case WGSL_TYPE_I32: case WGSL_TYPE_U32: case WGSL_TYPE_F32: l.size = 4; l.align = 4; break;
    case WGSL_TYPE_F16: l.size = 2; l.align = 2; break;
    case WGSL_TYPE_VEC: {
        WGSLTypeLayout el = wgsl_type_get_layout((const WGSLTypeInfo *)t->ref);
        l.align = (t->width == 2) ? 2 * el.align : 4 * el.align;
        l.size = t->width * el.size;
        break;
    }
    case WGSL_TYPE_MAT: {
        WGSLTypeLayout el = wgsl_type_get_layout((const WGSLTypeInfo *)t->ref);
        uint32_t align = (t->rows == 2) ? 2 * el.align : 4 * el.align;
        uint32_t stride = (t->rows * el.size + align - 1) & ~(align - 1);
        l.align = align;
        l.size = (t->width - 1) * stride + t->rows * el.size;
        break;
    }
    case WGSL_TYPE_ARRAY: {
        WGSLTypeLayout el = wgsl_type_get_layout((const WGSLTypeInfo *)t->ref);
        uint32_t stride = (el.size + el.align - 1) & ~(el.align - 1);
        l.align = el.align;
        l.size = (t->array_len > 0) ? (t->array_len - 1) * stride + el.size : 0;
        break;
    }
    default: break;
    }
    return l;
}

/* Conversion rank (WGSL §6.1.2). */

static int rank_scalar(WGSLTypeKind src, WGSLTypeKind dst) {
    if (src == dst) return 0;
    if (src == WGSL_TYPE_ABSTRACT_FLOAT) {
        if (dst == WGSL_TYPE_F32) return 1;
        if (dst == WGSL_TYPE_F16) return 2;
        return -1;
    }
    if (src == WGSL_TYPE_ABSTRACT_INT) {
        if (dst == WGSL_TYPE_I32)            return 3;
        if (dst == WGSL_TYPE_U32)            return 4;
        if (dst == WGSL_TYPE_ABSTRACT_FLOAT) return 5;
        if (dst == WGSL_TYPE_F32)            return 6;
        if (dst == WGSL_TYPE_F16)            return 7;
        return -1;
    }
    return -1;
}

int wgsl_type_conversion_rank(const WGSLTypeInfo *src, const WGSLTypeInfo *dst) {
    if (!src || !dst) return -1;
    if (src == dst)   return 0;     /* interned identity */

    /* Scalars */
    if (wgsl_type_is_scalar(src) && wgsl_type_is_scalar(dst)) {
        return rank_scalar((WGSLTypeKind)src->kind, (WGSLTypeKind)dst->kind);
    }

    /* vecN<S> → vecN<T> with same width */
    if (src->kind == WGSL_TYPE_VEC && dst->kind == WGSL_TYPE_VEC) {
        if (src->width != dst->width) return -1;
        return wgsl_type_conversion_rank(
            (const WGSLTypeInfo *)src->ref, (const WGSLTypeInfo *)dst->ref);
    }

    /* matCxR<S> → matCxR<T> with same dimensions */
    if (src->kind == WGSL_TYPE_MAT && dst->kind == WGSL_TYPE_MAT) {
        if (src->width != dst->width || src->rows != dst->rows) return -1;
        return wgsl_type_conversion_rank(
            (const WGSLTypeInfo *)src->ref, (const WGSLTypeInfo *)dst->ref);
    }

    /* array<S, N> → array<T, N> — fixed-size only, same length */
    if (src->kind == WGSL_TYPE_ARRAY && dst->kind == WGSL_TYPE_ARRAY) {
        if (src->array_len == 0 || dst->array_len == 0) return -1;
        if (src->array_len != dst->array_len)            return -1;
        return wgsl_type_conversion_rank(
            (const WGSLTypeInfo *)src->ref, (const WGSLTypeInfo *)dst->ref);
    }

    return -1;
}

/* Concretization (WGSL §6.1.2 default materialization). */

WGSLTypeInfo *wgsl_type_concretize(WGSLTypeStore *s, WGSLTypeInfo *t) {
    if (!t) return NULL;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_ABSTRACT_INT:   return s->t_i32;
    case WGSL_TYPE_ABSTRACT_FLOAT: return s->t_f32;
    case WGSL_TYPE_VEC: {
        WGSLTypeInfo *e = wgsl_type_concretize(s, (WGSLTypeInfo *)t->ref);
        if (e == t->ref) return t;
        return wgsl_type_vec(s, t->width, e);
    }
    case WGSL_TYPE_MAT: {
        WGSLTypeInfo *e = wgsl_type_concretize(s, (WGSLTypeInfo *)t->ref);
        if (e == t->ref) return t;
        return wgsl_type_mat(s, t->width, t->rows, e);
    }
    case WGSL_TYPE_ARRAY: {
        WGSLTypeInfo *e = wgsl_type_concretize(s, (WGSLTypeInfo *)t->ref);
        if (e == t->ref) return t;
        return wgsl_type_array(s, e, t->array_len);
    }
    default: return t;
    }
}

