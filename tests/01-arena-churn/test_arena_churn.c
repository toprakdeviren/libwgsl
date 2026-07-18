/**
 * Arena-churn baseline.
 *
 * Drives a synthetic workload shaped like a realistic WGSL parse —
 * nested expression trees + function bodies — through the arena, and
 * reports ns-per-alloc, total bytes, and chunks consumed.  The
 * baseline informs whether per-kind pooling is worth adding.
 *
 * PASS criterion (smoke):  every alloc returns non-NULL and the
 * counter math adds up.  Timing is printed for the human reader and
 * persisted under `docs/libwgsl.md` after the first run.
 */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "internal/arena.h"
#include "internal/ast.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Workload sized so total node count lands in the 500 K – 1 M range,
 * comparable to the AST one would build for a few thousand lines of
 * WGSL.  Tune up if the run time falls under ~50 ms (timer noise). */
#define ITER_COUNT      1000
#define EXPR_TREE_DEPTH 30      /* one chained binary expression */
#define FN_BODY_STMTS   50      /* per fn body                   */
#define STMT_EXPR_DEPTH 5

/* Build a left-recursive binary expression tree:
 *   ((((LIT op LIT) op LIT) op LIT) ... )
 * Each node owns a 2-element children array. */
static WGSLNode *build_binop_chain(WGSLArena *a, int depth) {
    WGSLNode *cur = (WGSLNode *)wgsl_arena_calloc(a, 1, sizeof *cur);
    if (!cur) return NULL;
    cur->kind = WGSL_NODE_EXPR_LITERAL_INT;
    cur->payload[0] = 0;

    for (int i = 0; i < depth; i++) {
        WGSLNode *rhs = (WGSLNode *)wgsl_arena_calloc(a, 1, sizeof *rhs);
        if (!rhs) return NULL;
        rhs->kind = WGSL_NODE_EXPR_LITERAL_INT;
        rhs->payload[0] = (uint64_t)(i + 1);

        WGSLNode *bin = (WGSLNode *)wgsl_arena_calloc(a, 1, sizeof *bin);
        if (!bin) return NULL;
        bin->kind = WGSL_NODE_EXPR_BINARY;
        bin->child_count = 2;
        bin->children = (WGSLNode **)wgsl_arena_calloc(a, 2, sizeof(WGSLNode *));
        if (!bin->children) return NULL;
        bin->children[0] = cur;
        bin->children[1] = rhs;

        cur = bin;
    }
    return cur;
}

/* Build a synthetic function body: compound statement with N
 * expression-statement children, each containing a small expr tree. */
static WGSLNode *build_fn_body(WGSLArena *a) {
    WGSLNode *block = (WGSLNode *)wgsl_arena_calloc(a, 1, sizeof *block);
    if (!block) return NULL;
    block->kind = WGSL_NODE_STMT_COMPOUND;
    block->child_count = FN_BODY_STMTS;
    block->children = (WGSLNode **)wgsl_arena_calloc(a, FN_BODY_STMTS, sizeof(WGSLNode *));
    if (!block->children) return NULL;

    for (int i = 0; i < FN_BODY_STMTS; i++) {
        WGSLNode *expr = build_binop_chain(a, STMT_EXPR_DEPTH);
        if (!expr) return NULL;
        WGSLNode *stmt = (WGSLNode *)wgsl_arena_calloc(a, 1, sizeof *stmt);
        if (!stmt) return NULL;
        stmt->kind = WGSL_NODE_STMT_FN_CALL;
        stmt->child_count = 1;
        stmt->children = (WGSLNode **)wgsl_arena_calloc(a, 1, sizeof(WGSLNode *));
        if (!stmt->children) return NULL;
        stmt->children[0] = expr;
        block->children[i] = stmt;
    }
    return block;
}

static double seconds_since(struct timespec t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0.tv_sec)
         + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

int main(void) {
    int fail = 0;

    /* Warm-up: one full build to prime the page cache + branch predictor.
     * Avoids the first iteration carrying outsized cost. */
    {
        WGSLArena warm; wgsl_arena_init(&warm);
        for (int i = 0; i < 4; i++) {
            (void)build_fn_body(&warm);
            (void)build_binop_chain(&warm, EXPR_TREE_DEPTH);
        }
        wgsl_arena_destroy(&warm);
    }

    WGSLArena a; wgsl_arena_init(&a);

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < ITER_COUNT; i++) {
        WGSLNode *body = build_fn_body(&a);
        WGSLNode *deep = build_binop_chain(&a, EXPR_TREE_DEPTH);
        if (!body || !deep) { fail = 1; break; }
    }

    double elapsed = seconds_since(t0);

    size_t allocs = wgsl_arena_alloc_count(&a);
    size_t bytes  = wgsl_arena_bytes(&a);

    /* Sanity: every iteration must produce both a body and a deep chain.
     * Per iteration = (1 fn body) + (1 deep expression).
     * fn body  = 1 block + 1 children[] + FN_BODY_STMTS × (1 stmt + 1 children[1] + nested expr)
     * Per nested expr (depth d) = (d+1) literals + d binops + d children[2]
     *                           = (1 + d) + d + d = 1 + 3d allocs
     */
    size_t alloc_per_nested = (size_t)(1 + 3 * STMT_EXPR_DEPTH);
    size_t alloc_per_body   = 1 /*block*/ + 1 /*block-children[]*/
                             + (size_t)FN_BODY_STMTS *
                                 (1 /*stmt*/ + 1 /*stmt-children[1]*/ + alloc_per_nested);
    size_t alloc_per_deep   = (size_t)(1 + 3 * EXPR_TREE_DEPTH);
    size_t alloc_per_iter   = alloc_per_body + alloc_per_deep;
    size_t expected_allocs  = (size_t)ITER_COUNT * alloc_per_iter;

    if (allocs != expected_allocs) {
        fprintf(stderr,
                "FAIL  alloc count: got %zu, expected %zu\n",
                allocs, expected_allocs);
        fail = 1;
    }

    if (fail == 0) {
        double ns_per_alloc = (elapsed * 1e9) / (double)allocs;
        double mb_per_sec   = ((double)bytes / 1.0e6) / elapsed;
        printf("PASS  test_arena_churn  "
               "%zu allocs in %.3f ms  "
               "(%.1f ns/alloc, %.1f MB/s, %.2f MB total)\n",
               allocs,
               elapsed * 1.0e3,
               ns_per_alloc,
               mb_per_sec,
               (double)bytes / 1.0e6);
    } else {
        fprintf(stderr, "FAIL  test_arena_churn\n");
    }

    wgsl_arena_destroy(&a);
    return fail ? 1 : 0;
}
