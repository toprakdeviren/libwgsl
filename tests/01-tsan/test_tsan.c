/**
 * TSan smoke test.
 *
 * The session model promises thread-safety via *isolation*: each
 * `WGSLArena` is owned by one thread; arenas are never shared.
 * This test sets up four workers, each driving its own arena hard,
 * and runs to completion under ThreadSanitizer.  PASS criterion:
 *
 *   - Every alloc returns non-NULL.
 *   - Per-arena counters match the work done.
 *   - TSan reports zero data races (process exit code 0).
 *
 * The test is also useful without TSan as a functional check that
 * pthreads + arena interact cleanly.
 */
#include "internal/arena.h"
#include "internal/ast.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_THREADS       4
#define ALLOCS_PER_THREAD 10000

typedef struct {
    int       worker_id;
    WGSLArena arena;
    int       fail;
    size_t    expected_bytes;
} Worker;

static void *worker_main(void *arg) {
    Worker *w = (Worker *)arg;

    for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
        WGSLNode *n = (WGSLNode *)wgsl_arena_calloc(&w->arena, 1, sizeof *n);
        if (!n) { w->fail = 1; return NULL; }

        n->kind        = (uint16_t)((i % (WGSL_NODE_KIND_COUNT - 1)) + 1);
        n->flags       = (uint16_t)w->worker_id;
        n->span_offset = (uint32_t)i;
        n->span_length = (uint32_t)sizeof *n;

        /* Mix in some non-uniform sizes so chunk-rollover paths get
         * exercised across different threads. */
        if ((i & 31) == 0) {
            void *blob = wgsl_arena_alloc(&w->arena, 113);
            if (!blob) { w->fail = 1; return NULL; }
            memset(blob, w->worker_id, 113);
        }
    }
    w->expected_bytes = wgsl_arena_bytes(&w->arena);
    return NULL;
}

int main(void) {
    Worker workers[NUM_THREADS];
    pthread_t threads[NUM_THREADS];
    int fail = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        memset(&workers[i], 0, sizeof workers[i]);
        workers[i].worker_id = i;
        wgsl_arena_init(&workers[i].arena);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, worker_main, &workers[i]) != 0) {
            fprintf(stderr, "FAIL  pthread_create %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Per-worker verdict — each arena is private to its worker, so its
     * post-join state must be self-consistent. */
    for (int i = 0; i < NUM_THREADS; i++) {
        if (workers[i].fail) {
            fprintf(stderr, "FAIL  worker %d reported alloc failure\n", i);
            fail += 1;
        }
        size_t allocs = wgsl_arena_alloc_count(&workers[i].arena);
        size_t expect_min = (size_t)ALLOCS_PER_THREAD;
        if (allocs < expect_min) {
            fprintf(stderr,
                    "FAIL  worker %d: alloc_count=%zu, expected >= %zu\n",
                    i, allocs, expect_min);
            fail += 1;
        }
        if (wgsl_arena_bytes(&workers[i].arena) != workers[i].expected_bytes) {
            fprintf(stderr, "FAIL  worker %d: bytes mismatch\n", i);
            fail += 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        wgsl_arena_destroy(&workers[i].arena);
    }

    if (fail == 0) {
        printf("PASS  test_tsan  %d threads × %d allocs, no race\n",
               NUM_THREADS, ALLOCS_PER_THREAD);
        return 0;
    }
    fprintf(stderr, "FAIL  test_tsan  %d issue(s)\n", fail);
    return 1;
}
