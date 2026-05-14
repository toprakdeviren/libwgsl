/**
 * @file layout.h — Memory layout calculator (WGSL §14.4).
 *
 * Computes alignment, size, and per-member offsets for WGSL types,
 * honouring per-AS rules (uniform AS bumps `RequiredAlignOf` for
 * struct / array types via `roundUp(16, AlignOf)`).
 *
 * Used by the validator (to enforce @align / @size / uniform-AS layout
 * constraints) and by the public `wgsl_module_json` (to expose per-
 * member offsets to consumers).
 *
 * Struct walks need to resolve member type-specs back to TypeInfo and
 * to const-evaluate `@align(N)` / `@size(N)` arguments — both via the
 * type checker's side table and the consteval module.  The helpers
 * therefore take a small `WGSLLayoutCtx` carrying the dependencies
 * explicitly so this module can live outside `validate.c`.
 */
#ifndef WGSL_INTERNAL_LAYOUT_H
#define WGSL_INTERNAL_LAYOUT_H

#include <stdint.h>
#include "internal/types.h"
#include "internal/resolver.h"     /* for WGSLAddressSpace */
#include "internal/check.h"        /* WGSLTypeChecker */
#include "internal/consteval.h"    /* WGSLConstEvaluator */
#include "internal/source.h"       /* WGSLSource */

typedef struct {
    WGSLTypeChecker    *tc;     /* member type lookup via wgsl_typecheck_type_of */
    WGSLConstEvaluator *cev;    /* @align(N) / @size(N) const-eval */
    const WGSLSource   *src;    /* attribute name match needs source bytes */
} WGSLLayoutCtx;

/* AlignOf(T) — basic alignment, ignoring address-space adjustments.
 * For struct/array, this is the underlying max-member / element align;
 * the uniform-AS bump lives in `wgsl_layout_required_align_of`. */
uint32_t wgsl_layout_align_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as);

/* SizeOf(T) — byte size for fixed-footprint types, 0 for runtime-sized
 * arrays.  Mat / array sizes account for stride padding via the
 * recursive `wgsl_layout_array_stride` call. */
uint32_t wgsl_layout_size_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as);

/* RequiredAlignOf(T, AS).  For uniform AS, struct / array types are
 * bumped to roundUp(16, AlignOf(T)) per spec §14.4.5; other AS use
 * AlignOf(T). */
uint32_t wgsl_layout_required_align_of(const WGSLLayoutCtx *cx, const WGSLTypeInfo *t, WGSLAddressSpace as);

/* StrideOf(array<elem, N>, AS) = roundUp(RequiredAlignOf(elem), SizeOf(elem)). */
uint32_t wgsl_layout_array_stride(const WGSLLayoutCtx *cx, const WGSLTypeInfo *elem, WGSLAddressSpace as);

/* Struct member offsets per spec §14.4.2 + §14.4.5.
 *   offsets_out[i] receives the byte offset of i-th DECL_STRUCT_MEMBER
 *   child (in source order), capped by `cap`.
 *   *out_align / *out_size receive the struct's overall align / total
 *   size (total size is rounded up to the struct's align).
 * Returns 1 on success, 0 if `t` isn't a struct, the AST is malformed,
 * or `cap` is too small. */
int wgsl_layout_struct(
    const WGSLLayoutCtx *cx,
    const WGSLTypeInfo *t, WGSLAddressSpace as,
    uint32_t *offsets_out, uint32_t cap,
    uint32_t *out_align, uint32_t *out_size);

#endif
