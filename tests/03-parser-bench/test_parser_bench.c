/**
 * Parser benchmark gate.
 *
 * Uses the real ML/transformer corpus under examples/shaders/ rather
 * than a synthetic stress sample.  For each shader:
 *   1. slurp once (file I/O excluded from timing)
 *   2. run wgsl_tokenize + wgsl_parse + free, ITER times, fresh arena
 *   3. record the *minimum* wall time (best estimate of the
 *      JIT/branch-predictor-warm path)
 *
 * Gate: aggregate ms/Kloc must stay under 4.0.
 * Goal:  ≤ 2 ms / Kloc on M-arm64-1 (Naga's order of magnitude).
 *
 * The first iteration is also reported separately as a "cold-ish" pass
 * — the file bytes are already in page cache from the slurp, but the
 * arena is freshly malloc'd, so this approximates first-parse cost.
 */
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/utf8.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SHADERS_DIR "examples/shaders"
#define ITERATIONS  30

static int fail = 0;

static char *slurp(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    *out_len = got;
    return buf;
}

static int line_count(const char *bytes, size_t len) {
    int n = 1;
    for (size_t i = 0; i < len; i++) if (bytes[i] == '\n') n += 1;
    return n;
}

static int has_suffix(const char *name, const char *suf) {
    size_t nl = strlen(name), sl = strlen(suf);
    return nl >= sl && memcmp(name + nl - sl, suf, sl) == 0;
}

static double seconds_since(struct timespec t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0.tv_sec)
         + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* One round-trip through lex + parse + arena destroy.  Returns success
 * (1) or failure (0); reports tokens + decls via out params if non-NULL. */
static int parse_round(const WGSLSource *src, size_t *out_tokens, size_t *out_decls) {
    WGSLArena arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;    wgsl_diag_init(&diag);
    WGSLLexResult lex = {0};
    WGSLAst ast       = {0};

    int ok = wgsl_tokenize(src, &arena, &diag, &lex)
          && wgsl_parse(&lex, src, &arena, &diag, &ast)
          && !wgsl_diag_has_error(&diag);

    if (out_tokens) *out_tokens = lex.count;
    if (out_decls)  *out_decls  = ast.root ? ast.root->child_count : 0;

    wgsl_diag_destroy(&diag);
    wgsl_arena_destroy(&arena);
    return ok;
}

typedef struct {
    const char *name;
    int    loc;
    size_t tokens;
    size_t decls;
    double min_ms;       /* min wall over ITER measured runs */
    double cold_ms;      /* iteration 1 (fresh arena, file already in page cache) */
} Row;

static int compare_rows(const void *a, const void *b) {
    return strcmp(((const Row *)a)->name, ((const Row *)b)->name);
}

int main(void) {
    {
        DIR *_d = opendir(SHADERS_DIR);
        if (!_d) {
            fprintf(stderr, "SKIP  %s: %s not present (corpus optional)\n",
                    __FILE__, SHADERS_DIR);
            return 0;
        }
        closedir(_d);
    }

    wgsl_utf8_init();

    DIR *d = opendir(SHADERS_DIR);
    if (!d) {
        fprintf(stderr, "FAIL  cannot opendir %s (run from wgsl/ root?)\n", SHADERS_DIR);
        return 1;
    }

    Row rows[64];
    int n_rows = 0;
    char path_buf[1024];

    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!has_suffix(e->d_name, ".wgsl")) continue;
        if (e->d_name[0] == '.') continue;
        if (n_rows >= 64) break;
        snprintf(path_buf, sizeof path_buf, "%s/%s", SHADERS_DIR, e->d_name);

        size_t len = 0;
        char *bytes = slurp(path_buf, &len);
        if (!bytes) {
            fprintf(stderr, "FAIL  slurp %s\n", path_buf);
            fail += 1;
            continue;
        }

        WGSLSource src;
        wgsl_source_init(&src, bytes, len);

        size_t tokens = 0, decls = 0;

        /* Cold-ish iteration */
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (!parse_round(&src, &tokens, &decls)) {
            fprintf(stderr, "FAIL  parse %s\n", e->d_name);
            fail += 1;
            wgsl_source_destroy(&src);
            free(bytes);
            continue;
        }
        double cold_ms = seconds_since(t0) * 1.0e3;

        /* Warm: take the min of ITER runs. */
        double min_ms = 1.0e9;
        for (int i = 0; i < ITERATIONS; i++) {
            struct timespec ti;
            clock_gettime(CLOCK_MONOTONIC, &ti);
            (void)parse_round(&src, NULL, NULL);
            double dt = seconds_since(ti) * 1.0e3;
            if (dt < min_ms) min_ms = dt;
        }

        Row *row    = &rows[n_rows++];
        row->name   = strdup(e->d_name);
        row->loc    = line_count(bytes, len);
        row->tokens = tokens;
        row->decls  = decls;
        row->min_ms = min_ms;
        row->cold_ms = cold_ms;

        wgsl_source_destroy(&src);
        free(bytes);
    }
    closedir(d);

    qsort(rows, (size_t)n_rows, sizeof(Row), compare_rows);

    int    total_loc    = 0;
    size_t total_tokens = 0;
    double total_min_ms = 0.0;
    double total_cold_ms = 0.0;

    fprintf(stderr,
            "  %-28s %6s %8s %12s %12s %10s\n",
            "shader", "LOC", "tokens", "warm ms", "cold ms", "ms/Kloc");
    for (int i = 0; i < n_rows; i++) {
        const Row *r = &rows[i];
        fprintf(stderr,
                "  %-28s %6d %8zu %12.4f %12.4f %10.3f\n",
                r->name, r->loc, r->tokens, r->min_ms, r->cold_ms,
                r->loc > 0 ? r->min_ms * 1000.0 / (double)r->loc : 0.0);
        total_loc    += r->loc;
        total_tokens += r->tokens;
        total_min_ms += r->min_ms;
        total_cold_ms += r->cold_ms;
    }
    double agg_ms_per_kloc = total_loc > 0
        ? total_min_ms * 1000.0 / (double)total_loc : 0.0;
    double agg_cold_ms_per_kloc = total_loc > 0
        ? total_cold_ms * 1000.0 / (double)total_loc : 0.0;

    fprintf(stderr,
            "  %-28s %6d %8zu %12.4f %12.4f %10.3f\n",
            "[ total ]", total_loc, total_tokens,
            total_min_ms, total_cold_ms, agg_ms_per_kloc);

    /* Parser throughput gate. */
    const double GATE_MS_PER_KLOC = 4.0;
    if (agg_ms_per_kloc > GATE_MS_PER_KLOC) {
        fprintf(stderr,
                "FAIL  parser throughput gate: %.3f ms/Kloc > %.1f (warm-min aggregate)\n",
                agg_ms_per_kloc, GATE_MS_PER_KLOC);
        fail += 1;
    }

    /* Free names */
    for (int i = 0; i < n_rows; i++) free((void *)rows[i].name);

    if (fail == 0) {
        printf("PASS  test_parser_bench  "
               "%d shaders / %d LOC, warm %.3f ms/Kloc, cold %.3f ms/Kloc "
               "(gate ≤ %.1f)\n",
               n_rows, total_loc, agg_ms_per_kloc, agg_cold_ms_per_kloc,
               GATE_MS_PER_KLOC);
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_bench  %d issue(s)\n", fail);
    return 1;
}
