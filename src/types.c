/**
 * @file types.c — WGSL type representation, interning, conversion rank.
 *                 See internal/types.h.
 */
#include "internal/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_TYPES_INITIAL_CAPACITY 64

/* ── Type construction + interning ──────────────────────────────── */

static int store_grow(WGSLTypeStore *s) {
    size_t new_cap = s->capacity ? s->capacity * 2 : WGSL_TYPES_INITIAL_CAPACITY;
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
    uint32_t array_len, void *ref)
{
    for (size_t i = 0; i < s->count; i++) {
        WGSLTypeInfo *t = s->interned[i];
        if (t->kind != (uint16_t)kind) continue;
        if (t->width != width)         continue;
        if (t->rows  != rows)          continue;
        if (t->array_len != array_len) continue;
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
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_VEC, width, 0, 0, elem);
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
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_MAT, cols, rows, 0, elem);
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
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ATOMIC, 0, 0, 0, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind = (uint16_t)WGSL_TYPE_ATOMIC;
    t->ref  = elem;
    return t;
}

WGSLTypeInfo *wgsl_type_array(WGSLTypeStore *s, WGSLTypeInfo *elem, uint32_t length) {
    if (!elem) return NULL;
    WGSLTypeInfo *hit = store_find(s, WGSL_TYPE_ARRAY, 0, 0, length, elem);
    if (hit) return hit;
    WGSLTypeInfo *t = store_alloc(s);
    if (!t) return NULL;
    t->kind      = (uint16_t)WGSL_TYPE_ARRAY;
    t->ref       = elem;
    t->array_len = length;
    return t;
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
int wgsl_type_is_matrix(const WGSLTypeInfo *t) { return t && t->kind == WGSL_TYPE_MAT; }

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
    case WGSL_TYPE_KIND_COUNT:     break;
    }
    return "<unknown>";
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
