/**
 * @file arena.c — Linked-list bump allocator.  See `internal/arena.h`.
 */
#include "internal/arena.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_ARENA_DEFAULT_CHUNK_BYTES (64u * 1024u)  /* 64 KiB */

struct WGSLArenaChunk {
    WGSLArenaChunk *next;
    size_t          capacity;   /* usable bytes after the header */
    size_t          used;       /* bytes consumed from `data`    */
    /* `data` follows; access via (char *)(c + 1).  Aligned to */
    /* `max_align_t` since chunk is `malloc`-allocated.        */
};

void wgsl_arena_init(WGSLArena *a) {
    a->head        = NULL;
    a->total_bytes = 0;
    a->alloc_count = 0;
}

void wgsl_arena_destroy(WGSLArena *a) {
    WGSLArenaChunk *c = a->head;
    while (c) {
        WGSLArenaChunk *next = c->next;
        free(c);
        c = next;
    }
    a->head        = NULL;
    a->total_bytes = 0;
    a->alloc_count = 0;
}

static WGSLArenaChunk *wgsl_arena_new_chunk(size_t cap) {
    /* Defense-in-depth: callers above this layer compute `cap` from
     * sizeof or counted-element multiplications already overflow-checked
     * in `wgsl_arena_calloc`, but a future direct `wgsl_arena_alloc`
     * caller could pass a `cap` near `SIZE_MAX`.  Reject before the
     * `sizeof *c + cap` addition wraps and `malloc` succeeds with a
     * tiny allocation. */
    if (cap > SIZE_MAX - sizeof(WGSLArenaChunk)) return NULL;
    WGSLArenaChunk *c = (WGSLArenaChunk *)malloc(sizeof *c + cap);
    if (!c) return NULL;
    c->next     = NULL;
    c->capacity = cap;
    c->used     = 0;
    return c;
}

static int is_pow2(size_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

void *wgsl_arena_alloc_aligned(WGSLArena *a, size_t size, size_t align) {
    if (size == 0 || !is_pow2(align)) return NULL;

    WGSLArenaChunk *c = a->head;
    if (c) {
        char *base = (char *)(c + 1);
        size_t off = c->used;
        size_t pad = ((align - ((uintptr_t)(base + off) & (align - 1))) & (align - 1));
        if (off <= c->capacity &&
            pad <= c->capacity - off &&
            size <= c->capacity - off - pad)
        {
            void *p = base + off + pad;
            c->used = off + pad + size;
            a->total_bytes = (size > SIZE_MAX - a->total_bytes)
                ? SIZE_MAX : a->total_bytes + size;
            a->alloc_count += 1;
            return p;
        }
    }

    size_t cap = WGSL_ARENA_DEFAULT_CHUNK_BYTES;
    if (align - 1 > SIZE_MAX - size) return NULL;
    size_t need = size + (align - 1);
    if (need > cap) cap = need;

    WGSLArenaChunk *nc = wgsl_arena_new_chunk(cap);
    if (!nc) return NULL;
    nc->next = a->head;
    a->head  = nc;

    char *base = (char *)(nc + 1);
    size_t pad = ((align - ((uintptr_t)base & (align - 1))) & (align - 1));
    if (pad > nc->capacity || size > nc->capacity - pad) {
        a->head = nc->next;
        free(nc);
        return NULL;
    }
    void *p = base + pad;
    nc->used = pad + size;
    a->total_bytes = (size > SIZE_MAX - a->total_bytes)
        ? SIZE_MAX : a->total_bytes + size;
    a->alloc_count += 1;
    return p;
}

void *wgsl_arena_alloc(WGSLArena *a, size_t size) {
    return wgsl_arena_alloc_aligned(a, size, 8);
}

void *wgsl_arena_calloc(WGSLArena *a, size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    /* Overflow check: count * size */
    if (count > (size_t)-1 / size) return NULL;
    size_t total = count * size;
    void *p = wgsl_arena_alloc(a, total);
    if (p) memset(p, 0, total);
    return p;
}

size_t wgsl_arena_bytes(const WGSLArena *a) {
    return a->total_bytes;
}

size_t wgsl_arena_alloc_count(const WGSLArena *a) {
    return a->alloc_count;
}
