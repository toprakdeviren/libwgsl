/**
 * @file types.h — WGSL type representation + conversion rank.
 *
 * `WGSLTypeInfo` is the resolver's / type-checker's view of a WGSL
 * type.  All instances are arena-allocated and **interned** by a
 * `WGSLTypeStore`: querying the same shape twice returns the same
 * pointer, so `t == u` is type identity.
 *
 * Pre-interned at session init:
 *   void · bool · AbstractInt · AbstractFloat · i32 · u32 · f32 · f16
 *
 * Built on demand via interning:
 *   vec2/3/4<T> · matCxR<T> for C,R∈{2,3,4} · atomic<T> · array<T,N> ·
 *   array<T> (runtime) · struct (one per AST decl).
 *
 * Conversion rank (`wgsl_type_conversion_rank`) implements the table
 * in WGSL §6.1.2 — see `docs/wgsl/02-tip-sistemi.md`.  Lower is better.
 * Returns -1 for "no automatic conversion".
 *
 * Layout: 24 bytes, 8-byte aligned.  Compile-time gated; never grow
 * without a deliberate bump.
 *
 * Phase 4 covers scalar / vec / mat / atomic / array / struct.
 * Ptr / Ref / Texture / Sampler are stubbed and land in Phase 7
 * (when the type checker actually uses them).
 */
#ifndef WGSL_INTERNAL_TYPES_H
#define WGSL_INTERNAL_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "internal/arena.h"

typedef enum {
    WGSL_TYPE_INVALID = 0,
    WGSL_TYPE_VOID,

    /* Scalars (pre-interned). */
    WGSL_TYPE_BOOL,
    WGSL_TYPE_ABSTRACT_INT,
    WGSL_TYPE_ABSTRACT_FLOAT,
    WGSL_TYPE_I32,
    WGSL_TYPE_U32,
    WGSL_TYPE_F32,
    WGSL_TYPE_F16,

    /* Composites (interned on demand). */
    WGSL_TYPE_VEC,        /* width = 2/3/4, ref = element type            */
    WGSL_TYPE_MAT,        /* width = cols, rows = rows, ref = element     */
    WGSL_TYPE_ATOMIC,     /* ref = element type (i32 or u32)              */
    WGSL_TYPE_ARRAY,      /* ref = element, array_len = N (0 = runtime)   */

    /* Aggregates. */
    WGSL_TYPE_STRUCT,     /* ref = WGSLNode* (the DECL_STRUCT AST node)   */

    /* Internal / Synthetic types. */
    WGSL_TYPE_ATOMIC_CX_RESULT, /* ref = element type                     */

    WGSL_TYPE_KIND_COUNT,
} WGSLTypeKind;

typedef struct WGSLTypeInfo {
    uint16_t kind;        /* WGSLTypeKind                                 */
    uint8_t  width;       /* vec width / mat columns; 0 otherwise         */
    uint8_t  rows;        /* mat rows; 0 otherwise                        */
    uint32_t array_len;   /* array length; 0 = runtime-sized or n/a       */
    void    *ref;         /* element type ptr (composites) / AST (struct) */
    uint32_t flags;       /* reserved for ref/ptr/addr_space (Phase 7)    */
    uint32_t pad_;        /* keep size at 24 bytes                        */
} WGSLTypeInfo;

/* Layout gate: 24 B / 8-aligned on LP64 ABIs (native arm64 / x86_64);
 * 20 B / 4-aligned on ILP32 ABIs (wasm32 / Emscripten Phase 10).  The
 * size shrink is the natural consequence of `void *ref` being 4 bytes
 * instead of 8 — every other field is fixed-width.  Both layouts are
 * intentional and exercised in CI. */
#if defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8
_Static_assert(sizeof(WGSLTypeInfo) == 24,
    "WGSLTypeInfo must be 24 bytes on LP64 ABIs.");
_Static_assert(_Alignof(WGSLTypeInfo) == 8,
    "WGSLTypeInfo must be 8-byte aligned on LP64 ABIs.");
#elif defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 4
_Static_assert(sizeof(WGSLTypeInfo) == 20,
    "WGSLTypeInfo must be 20 bytes on ILP32 ABIs (e.g., wasm32).");
_Static_assert(_Alignof(WGSLTypeInfo) == 4,
    "WGSLTypeInfo must be 4-byte aligned on ILP32 ABIs.");
#endif
_Static_assert(WGSL_TYPE_KIND_COUNT < (1u << 16),
    "WGSLTypeKind must fit in 16 bits.");

typedef struct WGSLTypeStore {
    WGSLArena    *arena;        /* borrowed; not owned                     */
    WGSLTypeInfo **interned;    /* heap-grown ptr list; owns its allocation */
    size_t        count;
    size_t        capacity;

    /* Pre-interned scalars (fast path — no lookup). */
    WGSLTypeInfo *t_void;
    WGSLTypeInfo *t_bool;
    WGSLTypeInfo *t_abstract_int;
    WGSLTypeInfo *t_abstract_float;
    WGSLTypeInfo *t_i32;
    WGSLTypeInfo *t_u32;
    WGSLTypeInfo *t_f32;
    WGSLTypeInfo *t_f16;
} WGSLTypeStore;

/** Initialize using `arena` for type storage.  Pre-interns scalars. */
int wgsl_types_init(WGSLTypeStore *s, WGSLArena *arena);

/** Release the heap intern table.  Type bodies live in the arena and
 *  go away when the arena is destroyed. */
void wgsl_types_destroy(WGSLTypeStore *s);

/** Returns one of the pre-interned scalars.  `k` must be a scalar /
 *  void kind (WGSL_TYPE_VOID..WGSL_TYPE_F16); otherwise NULL. */
WGSLTypeInfo *wgsl_type_scalar(const WGSLTypeStore *s, WGSLTypeKind k);

WGSLTypeInfo *wgsl_type_vec   (WGSLTypeStore *s, uint8_t width, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_mat   (WGSLTypeStore *s, uint8_t cols, uint8_t rows, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_atomic(WGSLTypeStore *s, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_atomic_cx_result(WGSLTypeStore *s, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_array (WGSLTypeStore *s, WGSLTypeInfo *elem, uint32_t length);

/* — Predicates — */

int wgsl_type_is_scalar  (const WGSLTypeInfo *t);
int wgsl_type_is_numeric (const WGSLTypeInfo *t);
int wgsl_type_is_integer (const WGSLTypeInfo *t);
int wgsl_type_is_float   (const WGSLTypeInfo *t);
int wgsl_type_is_abstract(const WGSLTypeInfo *t);
int wgsl_type_is_concrete(const WGSLTypeInfo *t);
int wgsl_type_is_vector  (const WGSLTypeInfo *t);
int wgsl_type_is_matrix  (const WGSLTypeInfo *t);

/* — Conversion rank (WGSL §6.1.2).  Returns -1 if no conversion. — */
int wgsl_type_conversion_rank(const WGSLTypeInfo *src, const WGSLTypeInfo *dst);

/**
 * Return the concretization of `t` per WGSL §6.1.2 — the concrete
 * type that an abstract value materializes to in a context that does
 * not steer it to anything more specific.  Defaults:
 *   AbstractInt   → i32
 *   AbstractFloat → f32
 *   vec/mat/array of abstract → corresponding composite of concrete
 *   already-concrete → returns `t`
 */
WGSLTypeInfo *wgsl_type_concretize(WGSLTypeStore *s, WGSLTypeInfo *t);

/* — Stringify — */

/** Short kind name for diagnostics. */
const char *wgsl_type_kind_name(WGSLTypeKind k);

/**
 * Format `t` into a human-readable WGSL-shaped string ("vec3<f32>",
 * "mat4x4<f32>", "array<u32, 4>", "atomic<i32>", "AbstractFloat", …).
 * Returns the number of bytes that *would* have been written
 * (excluding NUL); `cap` may be 0 to size-probe.
 */
size_t wgsl_type_format(const WGSLTypeInfo *t, char *buf, size_t cap);

#endif /* WGSL_INTERNAL_TYPES_H */
