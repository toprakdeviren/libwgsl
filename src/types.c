/**
 * @file types.c — WGSL type representation, interning, conversion rank.
 *                 See internal/types.h.
 */
#include "internal/types.h"
#include "internal/resolver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_TYPES_INITIAL_CAPACITY 64

/* ── Type construction + interning ──────────────────────────────── */

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

/* Find an already-interned type matching the given shape.  Linear
 * scan — fine while corpora stay in the low thousands; swap for a
 * hash table when a profile demands it. */
static WGSLTypeInfo *store_find(
    const WGSLTypeStore *s,
    WGSLTypeKind kind, uint8_t width, uint8_t rows,
    uint32_t array_len, uint32_t flags, uint32_t aux, void *ref)
{
    for (size_t i = 0; i < s->count; i++) {
        WGSLTypeInfo *t = s->interned[i];
        if (t->kind != (uint16_t)kind) continue;
        if (t->width != width)         continue;
        if (t->rows  != rows)          continue;
        if (t->array_len != array_len) continue;
        if (t->flags != flags)         continue;
        if (t->pad_ != aux)            continue;
        if (t->ref != ref)             continue;
        return t;
    }
    return NULL;
}

static WGSLTypeInfo *make_scalar(WGSLTypeStore *s, WGSLTypeKind k) {
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)k;
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

/* ── Predicates ─────────────────────────────────────────────────── */

int wgsl_type_is_scalar(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL:
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
        return 1;
    default:
        return 0;
    }
}

int wgsl_type_is_integer(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
        return 1;
    default: return 0;
    }
}

int wgsl_type_is_float(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
        return 1;
    default: return 0;
    }
}

int wgsl_type_is_numeric(const WGSLTypeInfo *t) {
    return wgsl_type_is_integer(t) || wgsl_type_is_float(t);
}

int wgsl_type_is_abstract(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ABSTRACT_INT)   return 1;
    if (t->kind == WGSL_TYPE_ABSTRACT_FLOAT) return 1;
    /* Composites of abstracts are abstract. */
    if (t->kind == WGSL_TYPE_VEC || t->kind == WGSL_TYPE_MAT ||
        t->kind == WGSL_TYPE_ARRAY)
    {
        return wgsl_type_is_abstract((const WGSLTypeInfo *)t->ref);
    }
    return 0;
}

int wgsl_type_is_concrete(const WGSLTypeInfo *t) {
    return t && !wgsl_type_is_abstract(t);
}

int wgsl_type_is_vector(const WGSLTypeInfo *t) { return t && t->kind == WGSL_TYPE_VEC; }
int wgsl_type_is_matrix(const WGSLTypeInfo *t) {
    return t ? t->kind == WGSL_TYPE_MAT : 0;
}

int wgsl_type_is_plain(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch (t->kind) {
    case WGSL_TYPE_BOOL:
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
    case WGSL_TYPE_ATOMIC:
    case WGSL_TYPE_ARRAY:
    case WGSL_TYPE_STRUCT:
    case WGSL_TYPE_ATOMIC_CX_RESULT:
    case WGSL_TYPE_FREXP_RESULT:
    case WGSL_TYPE_MODF_RESULT:
        return 1;
    default:
        return 0;
    }
}

int wgsl_type_contains_atomic(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (t->kind == WGSL_TYPE_ATOMIC) return 1;
    if (t->kind == WGSL_TYPE_ARRAY) {
        return wgsl_type_contains_atomic((const WGSLTypeInfo *)t->ref);
    }
    /* Struct members aren't directly available here. */
    return 0;
}

/* ── §6.2.11 / §6.2.12 / §6.2.13 — composite / constructible / footprint
 *
 * Struct members are NOT walked here (same reason as
 * wgsl_type_contains_atomic — the WGSLTypeInfo for a struct only
 * carries the AST node, and resolved member types live on the
 * type-checker side-table).  The deep walk lives in check.c. */

int wgsl_type_nest_depth(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_VEC:    return 1;
    case WGSL_TYPE_MAT:    return 2;
    case WGSL_TYPE_ARRAY:
        return 1 + wgsl_type_nest_depth((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_STRUCT:
        /* Lower bound — caller can refine via the side-table walk. */
        return 1;
    default:
        return 0;
    }
}

int wgsl_type_is_constructible(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL:
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
        return 1;
    case WGSL_TYPE_ARRAY:
        /* Per §6.2.12: fixed-size with generation-fixed footprint AND
         * constructible element type.  Runtime-sized (array_len == 0)
         * fails on both counts. */
        if (t->array_len == 0) return 0;
        return wgsl_type_is_constructible((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_STRUCT:
        /* Conservative — caller walks members for the precise answer. */
        return 1;
    case WGSL_TYPE_ATOMIC_CX_RESULT:
        /* §17.8 — the implicitly-defined `__atomic_compare_exchange_result<T>`
         * struct that `atomicCompareExchangeWeak` returns is constructible
         * even though it can't be spelled in WGSL source: shaders may
         * bind its value to a `let` or pass it as a function argument. */
        return 1;
    case WGSL_TYPE_FREXP_RESULT:
    case WGSL_TYPE_MODF_RESULT:
        /* §17 — `__frexp_result_T` / `__modf_result_T` are constructible
         * for the same reason as ATOMIC_CX_RESULT: their values may be
         * bound to `let` / passed as arguments / decomposed with `.fract`
         * / `.exp` / `.whole`. */
        return 1;
    default:
        /* atomic / ptr / ref / sampler / texture / void / invalid. */
        return 0;
    }
}

int wgsl_type_has_generation_fixed_footprint(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL:
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
    case WGSL_TYPE_ATOMIC:
        return 1;
    case WGSL_TYPE_ARRAY:
        if (t->array_len == 0) return 0;
        return wgsl_type_has_generation_fixed_footprint(
            (const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_STRUCT:
        return 1;
    default:
        return 0;
    }
}

int wgsl_type_has_fixed_footprint(const WGSLTypeInfo *t) {
    if (!t) return 0;
    if (wgsl_type_has_generation_fixed_footprint(t)) return 1;
    /* Override-sized arrays retain their override identity in flags/pad_,
     * but the concrete length is still not available in this type shape. */
    return 0;
}

/* §6.4.1 — Storable.  Plain types that can occupy a memory location:
 * scalar / vec / mat / atomic / array (any length) / struct.  Excludes
 * opaque kinds (texture, sampler, ptr, ref). */
int wgsl_type_is_storable(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_BOOL:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
    case WGSL_TYPE_ATOMIC:
    case WGSL_TYPE_ARRAY:
    case WGSL_TYPE_STRUCT:
    case WGSL_TYPE_ATOMIC_CX_RESULT:
    case WGSL_TYPE_FREXP_RESULT:
    case WGSL_TYPE_MODF_RESULT:
        return 1;
    default:
        return 0;
    }
}

/* §6.4.2 — Host-shareable.  Concrete-numeric scalar (no `bool`, no
 * abstract), vec / mat over those, atomic, array of host-shareable,
 * struct of host-shareable.  Struct walk is shallow (returns 1 for
 * struct kinds); deep validation lives at the validator's struct-decl
 * site. */
int wgsl_type_is_host_shareable(const WGSLTypeInfo *t) {
    if (!t) return 0;
    switch ((WGSLTypeKind)t->kind) {
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
        return 1;
    case WGSL_TYPE_VEC:
        return wgsl_type_is_host_shareable((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_MAT: {
        /* Matrix element type is f32 or f16. */
        const WGSLTypeInfo *e = (const WGSLTypeInfo *)t->ref;
        return e && (e->kind == WGSL_TYPE_F32 || e->kind == WGSL_TYPE_F16);
    }
    case WGSL_TYPE_ATOMIC:
        /* Atomic stores i32 / u32; both host-shareable. */
        return 1;
    case WGSL_TYPE_ARRAY:
        /* Element host-shareability is what matters; both fixed and
         * runtime-sized arrays of host-shareable elements are OK. */
        return wgsl_type_is_host_shareable((const WGSLTypeInfo *)t->ref);
    case WGSL_TYPE_STRUCT:
        /* Shallow accept; deep walk happens in check.c's struct-aware
         * version when the validator surfaces the rule. */
        return 1;
    default:
        return 0;
    }
}

/* ── Conversion rank (WGSL §6.1.2) ──────────────────────────────── */

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

/* ── Concretization (WGSL §6.1.2 default materialization) ──────── */

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

/* ── Stringify ──────────────────────────────────────────────────── */

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
