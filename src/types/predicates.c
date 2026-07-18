/**
 * @file predicates.c — type predicates, constructible / footprint / storable.
 */
#include "internal/types.h"

/* Predicates. */

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

/* Composite, constructible, and footprint predicates.
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
