/**
 * Phase 3 Round B Iter 3 — real-world WGSL corpus parse smoke.
 *
 * Iterates every `.wgsl` under `examples/shaders/` and runs the pipeline
 * on each.  Pass criterion: every file produces an AST with zero
 * error-severity diagnostics.  Per-file stats (LOC, tokens, top-level
 * decls, fns, structs) are printed for the human reader.
 *
 * The shaders carry intentional cross-kernel binding overlap (multiple
 * `@binding(0)` per file because the engine splits them on the
 * `// --- KERNEL: <name> ---` markers before compile).  That conflict
 * is a *semantic* check the validator handles later — the parser is
 * supposed to accept the surface form unconditionally.
 */
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/utf8.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHADERS_DIR "examples/shaders"

static int fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

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

static int kind_count(const WGSLNode *root, WGSLNodeKind k) {
    if (!root) return 0;
    int n = 0;
    for (uint32_t i = 0; i < root->child_count; i++)
        if (root->children[i]->kind == (uint16_t)k) n += 1;
    return n;
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

typedef struct { const char *name; int loc, tokens, decls, fns, structs; } Row;

static int compare_rows(const void *a, const void *b) {
    return strcmp(((const Row *)a)->name, ((const Row *)b)->name);
}

static int parse_one(const char *path, const char *display, Row *row) {
    size_t len = 0;
    char *src = slurp(path, &len);
    if (!src) {
        fprintf(stderr, "FAIL  cannot open %s\n", path);
        return 0;
    }

    WGSLArena arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;    wgsl_diag_init(&diag);
    WGSLSource source;   wgsl_source_init(&source, src, len);
    WGSLLexResult lex = {0};
    WGSLAst ast       = {0};

    int lex_ok   = wgsl_tokenize(&source, &arena, &diag, &lex);
    int parse_ok = wgsl_parse(&lex, &source, &arena, &diag, &ast);
    int ok = lex_ok && parse_ok && !wgsl_diag_has_error(&diag);

    if (!ok) {
        fprintf(stderr, "FAIL  %s — lex_ok=%d parse_ok=%d, %zu diag(s):\n",
                display, lex_ok, parse_ok, wgsl_diag_count(&diag));
        for (size_t i = 0; i < wgsl_diag_count(&diag); i++) {
            const WGSLDiagnostic *d = wgsl_diag_at(&diag, i);
            if (d->severity == WGSL_DIAG_ERROR) {
                fprintf(stderr,
                        "        %s:%u:%u-%u:%u  %s\n",
                        display, d->line, d->column, d->end_line, d->end_column,
                        d->message);
            }
        }
        fail += 1;
    }

    row->name    = display;
    row->loc     = line_count(src, len);
    row->tokens  = (int)lex.count;
    row->decls   = ast.root ? (int)ast.root->child_count : 0;
    row->fns     = kind_count(ast.root, WGSL_NODE_DECL_FUNCTION);
    row->structs = kind_count(ast.root, WGSL_NODE_DECL_STRUCT);

    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&source);
    wgsl_arena_destroy(&arena);
    free(src);
    return ok;
}

int main(void) {
    wgsl_utf8_init();

    DIR *d = opendir(SHADERS_DIR);
    if (!d) {
        fprintf(stderr, "FAIL  cannot opendir %s (run from wgsl/ root?)\n", SHADERS_DIR);
        return 1;
    }

    /* Two passes: first count, then parse — keeps output stable-sorted. */
    Row rows[64];
    int n_rows = 0;
    char path_buf[1024];

    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!has_suffix(e->d_name, ".wgsl")) continue;
        if (e->d_name[0] == '.') continue;
        if (n_rows >= 64) break;
        snprintf(path_buf, sizeof path_buf, "%s/%s", SHADERS_DIR, e->d_name);

        Row *row = &rows[n_rows];
        char *name_copy = strdup(e->d_name);
        if (!parse_one(path_buf, name_copy, row)) {
            /* failure already reported; continue so we surface every bad file */
        }
        n_rows += 1;
    }
    closedir(d);

    qsort(rows, (size_t)n_rows, sizeof(Row), compare_rows);

    int total_loc = 0, total_tokens = 0, total_decls = 0, total_fns = 0, total_structs = 0;
    fprintf(stderr,
            "  %-28s %6s %8s %6s %4s %7s\n",
            "shader", "LOC", "tokens", "decls", "fns", "structs");
    for (int i = 0; i < n_rows; i++) {
        const Row *r = &rows[i];
        fprintf(stderr,
                "  %-28s %6d %8d %6d %4d %7d\n",
                r->name, r->loc, r->tokens, r->decls, r->fns, r->structs);
        total_loc += r->loc;
        total_tokens += r->tokens;
        total_decls += r->decls;
        total_fns += r->fns;
        total_structs += r->structs;
    }
    fprintf(stderr,
            "  %-28s %6d %8d %6d %4d %7d\n",
            "[ total ]",
            total_loc, total_tokens, total_decls, total_fns, total_structs);

    /* Free the strdup'd names. */
    for (int i = 0; i < n_rows; i++) free((void *)rows[i].name);

    CHECK(n_rows >= 1, "found at least one shader");

    if (fail == 0) {
        printf("PASS  test_parser_corpus  %d shaders, %d total LOC, %d tokens, %d decls\n",
               n_rows, total_loc, total_tokens, total_decls);
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_corpus  %d issue(s)\n", fail);
    return 1;
}
