/**
 * `sizeof(WGSLNode)` layout gate.
 *
 * The compile-time guarantee is the `_Static_assert` in
 * `src/internal/ast.h`.  This binary is the runtime confirmation —
 * the value held by the actually-running, optimizer-shaped image
 * matches what the header asserts.
 *
 * It also smoke-tests the arena: alignment, multi-chunk crossing,
 * counter increments, and clean teardown.
 */
#include "internal/arena.h"
#include "internal/ast.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int fail = 0;

#define CHECK(cond, msg) do {                                      \
    if (!(cond)) {                                                 \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg); \
        fail += 1;                                                 \
    }                                                              \
} while (0)

int main(void) {
    /* — (a.1) Layout gate — */
    CHECK(sizeof(WGSLNode) == 48,    "sizeof(WGSLNode) must be 48");
    CHECK(_Alignof(WGSLNode) == 8,   "alignof(WGSLNode) must be 8");

    /* Field offsets are part of the contract too — guard them. */
    CHECK(offsetof(WGSLNode, kind)        == 0,  "kind @ 0");
    CHECK(offsetof(WGSLNode, flags)       == 2,  "flags @ 2");
    CHECK(offsetof(WGSLNode, child_count) == 4,  "child_count @ 4");
    CHECK(offsetof(WGSLNode, span_offset) == 8,  "span_offset @ 8");
    CHECK(offsetof(WGSLNode, span_length) == 12, "span_length @ 12");
    CHECK(offsetof(WGSLNode, children)    == 16, "children @ 16");
    CHECK(offsetof(WGSLNode, payload)     == 24, "payload @ 24");

    /* — (a.2) Kind enum sanity — */
    CHECK(WGSL_NODE_INVALID == 0,            "WGSL_NODE_INVALID stays at 0");
    CHECK(WGSL_NODE_KIND_COUNT > 30,          "kind enum holds the v1 kinds");
    CHECK(WGSL_NODE_KIND_COUNT < (1 << 16),   "kind fits in uint16_t");

    /* — (a.3) Arena smoke: align + multi-chunk + counters + teardown — */
    WGSLArena a;
    wgsl_arena_init(&a);
    CHECK(wgsl_arena_alloc_count(&a) == 0, "fresh arena: 0 allocs");
    CHECK(wgsl_arena_bytes(&a)       == 0, "fresh arena: 0 bytes");

    /* Place the first node, verify alignment and counters. */
    WGSLNode *n = (WGSLNode *)wgsl_arena_calloc(&a, 1, sizeof *n);
    CHECK(n != NULL,                       "first alloc returns non-NULL");
    CHECK(((uintptr_t)n & 7) == 0,         "node is 8-byte aligned");
    CHECK(n->kind == 0 && n->flags == 0,   "calloc zero-fills");
    CHECK(wgsl_arena_alloc_count(&a) == 1, "alloc count = 1");
    CHECK(wgsl_arena_bytes(&a) == sizeof *n, "byte total = sizeof(node)");

    n->kind        = (uint16_t)WGSL_NODE_TRANSLATION_UNIT;
    n->span_offset = 0;
    n->span_length = 1234;

    /* Pour 2000 nodes through to cross at least one 64 KiB chunk
     * boundary (2000 * 48 = 96 000 bytes > 65 536). */
    enum { N = 2000 };
    for (int i = 0; i < N; i++) {
        WGSLNode *m = (WGSLNode *)wgsl_arena_calloc(&a, 1, sizeof *m);
        CHECK(m != NULL,                  "bulk alloc returns non-NULL");
        CHECK(((uintptr_t)m & 7) == 0,    "every node is 8-byte aligned");
        m->kind = (uint16_t)((i % (WGSL_NODE_KIND_COUNT - 1)) + 1);
    }
    CHECK(wgsl_arena_alloc_count(&a) == (size_t)(N + 1), "alloc count tracked");
    CHECK(wgsl_arena_bytes(&a)       == (size_t)(N + 1) * sizeof(WGSLNode),
          "byte total tracked");

    /* — (a.4) Custom alignment honored — */
    void *p64 = wgsl_arena_alloc_aligned(&a, 8, 64);
    CHECK(p64 != NULL,                    "64-byte aligned alloc returns non-NULL");
    CHECK(((uintptr_t)p64 & 63) == 0,     "64-byte alignment honored");

    /* — (a.5) Edge cases — */
    CHECK(wgsl_arena_alloc(&a, 0) == NULL,           "zero size returns NULL");
    CHECK(wgsl_arena_alloc_aligned(&a, 8, 0) == NULL, "zero align rejected");
    CHECK(wgsl_arena_alloc_aligned(&a, 8, 7) == NULL, "non-pow2 align rejected");
    CHECK(wgsl_arena_alloc_aligned(&a, SIZE_MAX, 8) == NULL,
          "oversized allocation rejected");

    /* — (a.6) Teardown leaves arena in a reusable empty state — */
    wgsl_arena_destroy(&a);
    CHECK(wgsl_arena_alloc_count(&a) == 0, "post-destroy: 0 allocs");
    CHECK(wgsl_arena_bytes(&a)       == 0, "post-destroy: 0 bytes");
    /* Re-using a destroyed arena must work without re-init. */
    void *q = wgsl_arena_alloc(&a, 16);
    CHECK(q != NULL,                       "destroyed arena is reusable");
    wgsl_arena_destroy(&a);

    if (fail == 0) {
        printf("PASS  test_sizeof  "
               "sizeof(WGSLNode)=%zu align=%zu kinds=%d\n",
               sizeof(WGSLNode), _Alignof(WGSLNode),
               (int)WGSL_NODE_KIND_COUNT);
        return 0;
    } else {
        fprintf(stderr, "FAIL  test_sizeof  %d check(s) failed\n", fail);
        return 1;
    }
}
