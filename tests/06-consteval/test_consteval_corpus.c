/**
 * Const-evaluator coverage across the real shader corpus.
 *
 * Mirrors the resolver's corpus harness (`test_resolver_corpus.c`):
 *   1. `_shared.wgsl` is prepended to every kernel compilation.
 *   2. Each `.wgsl` is split at `// --- KERNEL: <name> ---` markers.
 *   3. For each kernel section: lex → parse → resolve → const-eval.
 *      Gate on every section completing with zero ERROR diagnostics.
 *
 * What this proves:
 *   - Every module-scope `const NAME = …;` in the corpus folds
 *     successfully (scalar arith on AbstractInt/AbstractFloat / typed
 *     i32 / u32 / f32 / f16).
 *   - Every `const_assert` (none currently in the corpus) — and any
 *     fn-scope const decl — is reachable and evaluates without error.
 *
 * The corpus does not currently exercise const vec/mat declarations.
 */
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
#include "internal/types.h"
#include "internal/utf8.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHADERS_DIR "examples/shaders"

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

static int has_suffix(const char *name, const char *suf) {
    size_t nl = strlen(name), sl = strlen(suf);
    return nl >= sl && memcmp(name + nl - sl, suf, sl) == 0;
}

static int find_kernel_markers(
    const char *src, size_t len, size_t *out, int max)
{
    static const char tag[] = "// --- KERNEL:";
    const size_t tl = sizeof tag - 1;
    int count = 0;
    for (size_t i = 0; i + tl <= len && count < max; i++) {
        int line_start = (i == 0) || src[i - 1] == '\n';
        if (line_start && memcmp(src + i, tag, tl) == 0) {
            out[count++] = i;
        }
    }
    return count;
}

static char *concat3(
    const char *a, size_t al,
    const char *b, size_t bl,
    const char *c, size_t cl,
    size_t *out_len)
{
    size_t total = al + bl + cl;
    char *r = (char *)malloc(total + 1);
    if (!r) return NULL;
    memcpy(r,             a, al);
    memcpy(r + al,        b, bl);
    memcpy(r + al + bl,   c, cl);
    r[total] = '\0';
    *out_len = total;
    return r;
}

typedef struct {
    int sections;
    int sections_ok;
    int total_consts;
    int total_const_asserts;
} FileStats;

static int count_kind(const WGSLNode *root, WGSLNodeKind k) {
    if (!root) return 0;
    int n = 0;
    for (uint32_t i = 0; i < root->child_count; i++) {
        if (root->children[i] && root->children[i]->kind == (uint16_t)k) n += 1;
    }
    return n;
}

static int eval_module(
    const char *bytes, size_t len, const char *display, FileStats *st)
{
    WGSLArena arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;    wgsl_diag_init(&diag);
    WGSLSource src;      wgsl_source_init(&src, bytes, len);
    WGSLLexResult lex = {0};
    WGSLAst ast       = {0};
    WGSLTypeStore ts  = {0};
    WGSLResolver res  = {0};
    WGSLConstEvaluator cev = {0};

    int lex_ok    = wgsl_tokenize(&src, &arena, &diag, &lex);
    int parse_ok  = wgsl_parse(&lex, &src, &arena, &diag, &ast);
    wgsl_types_init(&ts, &arena);
    int res_ok    = wgsl_resolve(&ast, &src, &arena, &diag, &ts, &res);
    int ce_ok     = wgsl_consteval(&ast, &src, &arena, &diag, &ts, &res, &cev);

    int ok = lex_ok && parse_ok && res_ok && ce_ok && !wgsl_diag_has_error(&diag);
    if (ok) {
        st->sections_ok += 1;
        st->total_consts += count_kind(ast.root, WGSL_NODE_DECL_CONST);
        st->total_const_asserts +=
            count_kind(ast.root, WGSL_NODE_DECL_CONST_ASSERT);
    } else {
        fprintf(stderr,
            "FAIL  %s — lex=%d parse=%d res=%d ce=%d  %zu diag(s):\n",
            display, lex_ok, parse_ok, res_ok, ce_ok,
            wgsl_diag_count(&diag));
        size_t printed = 0;
        for (size_t i = 0; i < wgsl_diag_count(&diag) && printed < 5; i++) {
            const WGSLDiagnostic *d = wgsl_diag_at(&diag, i);
            if (d->severity != WGSL_DIAG_ERROR) continue;
            fprintf(stderr, "        %s:%u:%u  %s\n",
                    display, d->line, d->column, d->message);
            printed += 1;
        }
        fail += 1;
    }

    wgsl_consteval_destroy(&cev);
    wgsl_resolver_destroy(&res);
    wgsl_types_destroy(&ts);
    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&src);
    wgsl_arena_destroy(&arena);
    return ok;
}

static void run_file(
    const char *display,
    const char *shared, size_t shared_len,
    const char *file,   size_t file_len,
    FileStats *st)
{
    size_t markers[64];
    int nm = find_kernel_markers(file, file_len, markers, 64);

    if (nm == 0) {
        st->sections += 1;
        if (shared_len == 0) {
            eval_module(file, file_len, display, st);
        } else {
            size_t mlen = 0;
            char *m = concat3(shared, shared_len, "", 0, file, file_len, &mlen);
            if (m) { eval_module(m, mlen, display, st); free(m); }
        }
        return;
    }

    const char *pre  = file;
    size_t pre_len   = markers[0];

    for (int i = 0; i < nm; i++) {
        size_t s_off = markers[i];
        size_t s_end = (i + 1 < nm) ? markers[i + 1] : file_len;
        const char *body = file + s_off;
        size_t body_len  = s_end - s_off;

        char section_name[64] = "section";
        const char *line_end = memchr(body, '\n', body_len);
        size_t header_len = line_end ? (size_t)(line_end - body) : body_len;
        if (header_len < sizeof section_name) {
            memcpy(section_name, body, header_len);
            section_name[header_len] = '\0';
        }
        char display_buf[256];
        snprintf(display_buf, sizeof display_buf,
                 "%s :: %s", display, section_name);

        size_t mlen = 0;
        char *m = concat3(shared, shared_len, pre, pre_len, body, body_len, &mlen);
        if (m) {
            st->sections += 1;
            eval_module(m, mlen, display_buf, st);
            free(m);
        }
    }
}

int main(void) {
    {
        DIR *_d = opendir(SHADERS_DIR);
        if (!_d) {
            fprintf(stderr, "SKIP  %s: %s not present — VACUOUS (no corpus coverage; not a real pass)\n",
                    __FILE__, SHADERS_DIR);
            printf("PASS  corpus  VACUOUS (no %s)\n", SHADERS_DIR);
            return 0;
        }
        closedir(_d);
    }

    wgsl_utf8_init();

    size_t shared_len = 0;
    char *shared = slurp(SHADERS_DIR "/_shared.wgsl", &shared_len);
    if (!shared) {
        fprintf(stderr, "FAIL  cannot read %s/_shared.wgsl\n", SHADERS_DIR);
        return 1;
    }

    FileStats shared_stats = {0};
    run_file("_shared.wgsl", "", 0, shared, shared_len, &shared_stats);

    DIR *d = opendir(SHADERS_DIR);
    if (!d) { free(shared); return 1; }

    typedef struct { char *name; FileStats st; } Row;
    Row rows[64];
    int nrows = 0;

    char *names[64];
    int n_names = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!has_suffix(e->d_name, ".wgsl")) continue;
        if (e->d_name[0] == '.') continue;
        if (strcmp(e->d_name, "_shared.wgsl") == 0) continue;
        if (n_names >= 64) break;
        names[n_names++] = strdup(e->d_name);
    }
    closedir(d);

    for (int i = 1; i < n_names; i++) {
        char *k = names[i];
        int j = i - 1;
        while (j >= 0 && strcmp(names[j], k) > 0) {
            names[j + 1] = names[j]; j -= 1;
        }
        names[j + 1] = k;
    }

    for (int i = 0; i < n_names; i++) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", SHADERS_DIR, names[i]);
        size_t flen = 0;
        char *fbuf = slurp(path, &flen);
        if (!fbuf) {
            fprintf(stderr, "FAIL  cannot open %s\n", path);
            fail += 1;
            free(names[i]);
            continue;
        }
        rows[nrows].name = names[i];
        rows[nrows].st = (FileStats){0};
        run_file(names[i], shared, shared_len, fbuf, flen, &rows[nrows].st);
        nrows += 1;
        free(fbuf);
    }

    int total_sections = shared_stats.sections;
    int total_ok       = shared_stats.sections_ok;
    int total_consts   = shared_stats.total_consts;
    int total_asserts  = shared_stats.total_const_asserts;
    fprintf(stderr,
        "  %-32s %9s %5s %7s %9s\n",
        "shader", "sections", "ok", "consts", "asserts");
    fprintf(stderr,
        "  %-32s %9d %5d %7d %9d\n",
        "_shared.wgsl",
        shared_stats.sections, shared_stats.sections_ok,
        shared_stats.total_consts, shared_stats.total_const_asserts);
    for (int i = 0; i < nrows; i++) {
        fprintf(stderr,
            "  %-32s %9d %5d %7d %9d\n",
            rows[i].name,
            rows[i].st.sections, rows[i].st.sections_ok,
            rows[i].st.total_consts, rows[i].st.total_const_asserts);
        total_sections += rows[i].st.sections;
        total_ok       += rows[i].st.sections_ok;
        total_consts   += rows[i].st.total_consts;
        total_asserts  += rows[i].st.total_const_asserts;
        free(rows[i].name);
    }
    fprintf(stderr,
        "  %-32s %9d %5d %7d %9d\n",
        "[ total ]",
        total_sections, total_ok, total_consts, total_asserts);

    free(shared);

    if (fail == 0) {
        printf(
            "PASS  test_consteval_corpus  %d sections (%d const decls, %d const_asserts)\n",
            total_sections, total_consts, total_asserts);
        return 0;
    }
    fprintf(stderr,
        "FAIL  test_consteval_corpus  %d section(s) failed\n", fail);
    return 1;
}
