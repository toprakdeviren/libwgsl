/**
 * @file arena.h — Bump allocator for AST / TypeInfo / scope nodes.
 *
 * One arena = one logical region (typically: one parse session).
 * `wgsl_arena_destroy` releases everything at once.  Allocations are
 * 8-byte aligned by default; use `_aligned` for stronger demands.
 *
 * Counters (`wgsl_arena_bytes`, `wgsl_arena_alloc_count`) are exposed for
 * allocator churn benchmarks and embedders that want coarse telemetry.
 */
#ifndef WGSL_INTERNAL_ARENA_H
#define WGSL_INTERNAL_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct WGSLArenaChunk WGSLArenaChunk;

typedef struct WGSLArena {
    WGSLArenaChunk *head;          /* most-recent chunk; allocations go here */
    size_t          total_bytes;   /* cumulative bytes returned by alloc     */
    size_t          alloc_count;   /* cumulative successful allocations      */
} WGSLArena;

/** Initialize.  No allocation; ready for use. */
void wgsl_arena_init(WGSLArena *a);

/** Release every chunk.  Safe on a zero-init arena.  Sets `*a` to empty. */
void wgsl_arena_destroy(WGSLArena *a);

/** Allocate `size` bytes, 8-byte aligned.  Returns NULL on OOM or `size==0`. */
void *wgsl_arena_alloc(WGSLArena *a, size_t size);

/** Allocate `size` bytes with custom alignment (must be a power of two). */
void *wgsl_arena_alloc_aligned(WGSLArena *a, size_t size, size_t align);

/** Zero-fill convenience: `count * size` bytes, 8-byte aligned, memset to 0. */
void *wgsl_arena_calloc(WGSLArena *a, size_t count, size_t size);

/** Total bytes returned to callers (does not include chunk headers/padding). */
size_t wgsl_arena_bytes(const WGSLArena *a);

/** Number of successful allocations. */
size_t wgsl_arena_alloc_count(const WGSLArena *a);

#endif /* WGSL_INTERNAL_ARENA_H */
