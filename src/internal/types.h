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
    WGSL_TYPE_FREXP_RESULT,     /* ref = element type (T = f / vecN<f>)   */
    WGSL_TYPE_MODF_RESULT,      /* ref = element type (T = f / vecN<f>)   */

    /* Opaque handle types (§6.5). */
    WGSL_TYPE_SAMPLER,    /* width = 0 normal, 1 comparison               */
    WGSL_TYPE_TEXTURE,    /* width      = WGSLTextureDim                  */
                          /* rows       = WGSLAccessMode (storage only)   */
                          /* array_len  = WGSLTexelFormat (storage only)  */
                          /* ref        = sampled element type            */
                          /*              (NULL for depth/external/storage)*/

    /* Memory views (§6.4) */
    WGSL_TYPE_PTR,        /* ref = element, flags = addr_space | access<<8*/
    WGSL_TYPE_REF,        /* ref = element, flags = addr_space | access<<8*/

    WGSL_TYPE_KIND_COUNT,
} WGSLTypeKind;

/* §6.5 texture flavours.  Encodes dimensionality + kind in one slot so
 * the interner's (kind,width,rows,array_len,ref) key suffices.        */
typedef enum {
    WGSL_TEX_DIM_1D = 0,
    WGSL_TEX_DIM_2D,
    WGSL_TEX_DIM_2D_ARRAY,
    WGSL_TEX_DIM_3D,
    WGSL_TEX_DIM_CUBE,
    WGSL_TEX_DIM_CUBE_ARRAY,
    WGSL_TEX_DIM_MULTISAMPLED_2D,
    WGSL_TEX_DIM_EXTERNAL,
    /* Depth. */
    WGSL_TEX_DIM_DEPTH_2D,
    WGSL_TEX_DIM_DEPTH_2D_ARRAY,
    WGSL_TEX_DIM_DEPTH_CUBE,
    WGSL_TEX_DIM_DEPTH_CUBE_ARRAY,
    WGSL_TEX_DIM_DEPTH_MULTISAMPLED_2D,
    /* Storage. */
    WGSL_TEX_DIM_STORAGE_1D,
    WGSL_TEX_DIM_STORAGE_2D,
    WGSL_TEX_DIM_STORAGE_2D_ARRAY,
    WGSL_TEX_DIM_STORAGE_3D,
} WGSLTextureDim;

/* §6.5.10 — WGSL texel formats for storage textures.  Order matters
 * only for the format-name lookup table in resolver.c. */
typedef enum {
    WGSL_TEX_FORMAT_NONE = 0,
    /* 8-bit components. */
    WGSL_TEX_FORMAT_R8UNORM,
    WGSL_TEX_FORMAT_R8SNORM,
    WGSL_TEX_FORMAT_R8UINT,
    WGSL_TEX_FORMAT_R8SINT,
    /* 16-bit components. */
    WGSL_TEX_FORMAT_R16UINT,
    WGSL_TEX_FORMAT_R16SINT,
    WGSL_TEX_FORMAT_R16FLOAT,
    WGSL_TEX_FORMAT_RG8UNORM,
    WGSL_TEX_FORMAT_RG8SNORM,
    WGSL_TEX_FORMAT_RG8UINT,
    WGSL_TEX_FORMAT_RG8SINT,
    /* 32-bit components (and similar total sizes). */
    WGSL_TEX_FORMAT_R32UINT,
    WGSL_TEX_FORMAT_R32SINT,
    WGSL_TEX_FORMAT_R32FLOAT,
    WGSL_TEX_FORMAT_RG16UINT,
    WGSL_TEX_FORMAT_RG16SINT,
    WGSL_TEX_FORMAT_RG16FLOAT,
    WGSL_TEX_FORMAT_RGBA8UNORM,
    WGSL_TEX_FORMAT_RGBA8UNORM_SRGB,
    WGSL_TEX_FORMAT_RGBA8SNORM,
    WGSL_TEX_FORMAT_RGBA8UINT,
    WGSL_TEX_FORMAT_RGBA8SINT,
    WGSL_TEX_FORMAT_BGRA8UNORM,
    WGSL_TEX_FORMAT_BGRA8UNORM_SRGB,
    WGSL_TEX_FORMAT_RGB10A2UNORM,
    WGSL_TEX_FORMAT_RG11B10FLOAT,
    /* 64-bit components. */
    WGSL_TEX_FORMAT_RG32UINT,
    WGSL_TEX_FORMAT_RG32SINT,
    WGSL_TEX_FORMAT_RG32FLOAT,
    WGSL_TEX_FORMAT_RGBA16UINT,
    WGSL_TEX_FORMAT_RGBA16SINT,
    WGSL_TEX_FORMAT_RGBA16FLOAT,
    /* 128-bit components. */
    WGSL_TEX_FORMAT_RGBA32UINT,
    WGSL_TEX_FORMAT_RGBA32SINT,
    WGSL_TEX_FORMAT_RGBA32FLOAT,
    /* texture_formats_tier1 extras (§6.5.10). */
    WGSL_TEX_FORMAT_R16UNORM,
    WGSL_TEX_FORMAT_R16SNORM,
    WGSL_TEX_FORMAT_RG16UNORM,
    WGSL_TEX_FORMAT_RG16SNORM,
    WGSL_TEX_FORMAT_RGBA16UNORM,
    WGSL_TEX_FORMAT_RGBA16SNORM,
    WGSL_TEX_FORMAT_RGB10A2UINT,
    WGSL_TEX_FORMAT_RG11B10UFLOAT,
} WGSLTexelFormat;

/* §13.3 access modes — also used as the second template arg on
 * texture_storage_*.  The resolver's PREDECLARED_ACCESS_MODE
 * symbols carry these values implicitly via name lookup. */
typedef enum {
    WGSL_ACCESS_NONE       = 0,
    WGSL_ACCESS_READ       = 1,
    WGSL_ACCESS_WRITE      = 2,
    WGSL_ACCESS_READ_WRITE = 3,
} WGSLAccessMode;

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

typedef struct {
    uint32_t size;
    uint32_t align;
} WGSLTypeLayout;

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
WGSLTypeInfo *wgsl_type_frexp_result(WGSLTypeStore *s, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_modf_result (WGSLTypeStore *s, WGSLTypeInfo *elem);
WGSLTypeInfo *wgsl_type_array (WGSLTypeStore *s, WGSLTypeInfo *elem, uint32_t length);

/* §6.2.9 array equality — override-sized arrays.
 *
 * Two override-sized arrays compare equal only when their override
 * symbols match.  The interner keeps `ref` as the element type and
 * stores the override symbol pointer split across `flags` and `pad_`,
 * preserving full pointer identity without growing `WGSLTypeInfo`.
 * Pointer-equality on the resulting `WGSLTypeInfo *` then matches the
 * spec's identifier-equality requirement. */
WGSLTypeInfo *wgsl_type_array_override(
    WGSLTypeStore *s, WGSLTypeInfo *elem, const void *override_sym);

/* Opaque handle types (§6.5).  Interned by shape so identity-equality
 * works.  `sampled_elem` is the texture's sampled type (i32/u32/f32)
 * for sampled / multisampled flavours, and NULL for depth / external /
 * storage textures.  `format` and `access` are NONE for non-storage. */
WGSLTypeInfo *wgsl_type_sampler(WGSLTypeStore *s, int comparison);
WGSLTypeInfo *wgsl_type_texture(
    WGSLTypeStore *s,
    WGSLTextureDim dim,
    WGSLTypeInfo  *sampled_elem,
    WGSLTexelFormat format,
    WGSLAccessMode  access);

/* Memory views (§6.4). `flags` packs AddressSpace in bottom 8 bits, AccessMode in next 8. */
WGSLTypeInfo *wgsl_type_ptr(WGSLTypeStore *s, uint8_t as, WGSLTypeInfo *elem, WGSLAccessMode am);
WGSLTypeInfo *wgsl_type_ref(WGSLTypeStore *s, uint8_t as, WGSLTypeInfo *elem, WGSLAccessMode am);

/* — Predicates — */

int wgsl_type_is_scalar  (const WGSLTypeInfo *t);
int wgsl_type_is_numeric (const WGSLTypeInfo *t);
int wgsl_type_is_integer (const WGSLTypeInfo *t);
int wgsl_type_is_float   (const WGSLTypeInfo *t);
int wgsl_type_is_abstract(const WGSLTypeInfo *t);
int wgsl_type_is_concrete(const WGSLTypeInfo *t);
int wgsl_type_is_vector  (const WGSLTypeInfo *t);
int wgsl_type_is_matrix  (const WGSLTypeInfo *t);
int wgsl_type_is_plain   (const WGSLTypeInfo *t);
int wgsl_type_contains_atomic(const WGSLTypeInfo *t);

/* §6.2.11 — composite NestDepth.
 *
 * vec=1, mat=2, array<E>=1+NestDepth(E), struct=1+max(member NestDepth).
 * Returns 0 for non-composite types.  Struct members are NOT walked
 * here — types.c doesn't have access to the type-checker's per-node
 * type side-table; the deep walk lives in check.c.  As a lower bound
 * for struct, this returns 1 (every WGSL struct has ≥1 member). */
int wgsl_type_nest_depth(const WGSLTypeInfo *t);

/* §6.2.12 — Constructible.
 *
 * scalar | vec | mat | fixed-size array (const-expr length) of
 * constructible | struct of constructible.  Excludes atomic,
 * runtime-sized array, and composites containing them.  Struct
 * members are NOT walked here (same reason as nest_depth); the
 * shallow predicate returns 1 for struct kinds and the deep walk
 * happens in check.c via `wgsl_typecheck_is_constructible`. */
int wgsl_type_is_constructible(const WGSLTypeInfo *t);

/* §6.2.13 — Generation-fixed footprint.
 *
 * scalar | vec | mat | atomic | fixed-size array (const-expr length)
 * Generation-fixed footprint OR fixed-size array (override-sized variant
 * counts here even though it doesn't have GFF).  v1 can't yet
 * distinguish override-sized from runtime-sized arrays in the type
 * info (§6.2.9 identity-tracking gap), so this predicate currently
 * coincides with GFF. */
int wgsl_type_has_generation_fixed_footprint(const WGSLTypeInfo *t);

/* §6.2.13 — Fixed footprint.
 *
 * Generation-fixed footprint OR fixed-size array (override-sized variant
 * counts here even though it doesn't have GFF).  v1 can't yet
 * distinguish override-sized from runtime-sized arrays in the type
 * info (§6.2.9 identity-tracking gap), so this predicate currently
 * coincides with GFF. */
int wgsl_type_has_fixed_footprint(const WGSLTypeInfo *t);

/* §6.4.1 — Storable.
 *
 * Plain types that fit in a memory location: scalar / vec / mat /
 * atomic / array / struct.  Excludes opaque kinds (texture, sampler,
 * ptr, ref). */
int wgsl_type_is_storable(const WGSLTypeInfo *t);

/* §6.4.2 — Host-shareable.
 *
 * Concrete-numeric scalar (i32 / u32 / f32 / f16) | vec of host-
 * shareable scalar | mat (f32/f16 element) | atomic | array of host-
 * shareable | struct of host-shareable.  Excludes `bool` and any type
 * containing it.  The spec deep-walk for structs is shallow here
 * (returns 1 for struct kinds); a struct-aware deep version lives in
 * check.c when the validator needs it. */
int wgsl_type_is_host_shareable(const WGSLTypeInfo *t);

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

/** Compute the size and alignment of `t` per WGSL §14.4. */
WGSLTypeLayout wgsl_type_get_layout(const WGSLTypeInfo *t);

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
