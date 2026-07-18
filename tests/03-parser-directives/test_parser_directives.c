/**
 * Directive parser tests.
 *
 * Covers:
 *   - enable f16;            (single)
 *   - enable f16, subgroups; (multi)
 *   - requires …;
 *   - diagnostic(off, derivative_uniformity);
 *   - diagnostic(warning, foo.bar);   (dotted rule)
 */
#include "internal/parser.h"
#include "internal/lexer.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

typedef struct {
    WGSLArena      arena;
    WGSLSource     src;
    WGSLDiagBag    diag;
    WGSLLexResult  lex;
    WGSLAst        ast;
    int            ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    int parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    p->ok = lex_ok && parse_ok && !wgsl_diag_has_error(&p->diag);
}
static void done(P *p) {
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int span_eq(const WGSLNode *n, const WGSLSource *s, const char *want) {
    size_t wl = strlen(want);
    if (n->span_length != wl) return 0;
    return memcmp(s->bytes + n->span_offset, want, wl) == 0;
}

int main(void) {
    wgsl_utf8_init();

    /* — enable single — */
    {
        P p; run(&p, "enable f16;");
        CHECK(p.ok, "enable single: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(d->kind == WGSL_NODE_DIR_ENABLE, "enable: kind");
        CHECK(wgsl_dir_name_count(d) == 1,    "enable: 1 ext");
        CHECK(d->child_count == 1,             "enable: 1 child");
        CHECK(span_eq(d->children[0], &p.src, "f16"), "enable: f16");
        done(&p);
    }
    /* — enable multi — */
    {
        P p; run(&p, "enable f16, subgroups, clip_distances;");
        CHECK(p.ok, "enable multi: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(wgsl_dir_name_count(d) == 3, "enable multi: 3 exts");
        CHECK(d->child_count == 3,           "enable multi: 3 children");
        CHECK(span_eq(d->children[1], &p.src, "subgroups"),     "enable: subgroups");
        CHECK(span_eq(d->children[2], &p.src, "clip_distances"), "enable: clip_distances");
        done(&p);
    }
    /* — enable trailing comma — */
    {
        P p; run(&p, "enable f16, subgroups,;");
        CHECK(p.ok, "enable trailing-comma: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(wgsl_dir_name_count(d) == 2, "enable trailing-comma: 2 exts");
        done(&p);
    }

    /* — requires — */
    {
        P p; run(&p, "requires readonly_and_readwrite_storage_textures;");
        CHECK(p.ok, "requires: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(d->kind == WGSL_NODE_DIR_REQUIRES, "requires: kind");
        CHECK(d->child_count == 1,                "requires: 1 child");
        CHECK(span_eq(d->children[0], &p.src,
                      "readonly_and_readwrite_storage_textures"), "requires: name");
        done(&p);
    }

    /* — diagnostic single-part rule — */
    {
        P p; run(&p, "diagnostic(off, derivative_uniformity);");
        CHECK(p.ok, "diagnostic 1-part: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(d->kind == WGSL_NODE_DIR_DIAGNOSTIC, "diag: kind");
        CHECK(wgsl_diag_rule_parts(d) == 1,         "diag: 1 rule part");
        CHECK(d->child_count == 2,                  "diag: 2 children (sev + rule)");
        CHECK(span_eq(d->children[0], &p.src, "off"),                   "diag: severity");
        CHECK(span_eq(d->children[1], &p.src, "derivative_uniformity"), "diag: rule");
        done(&p);
    }
    /* — diagnostic dotted rule — */
    {
        P p; run(&p, "diagnostic(warning, ext.subgroup_uniformity);");
        CHECK(p.ok, "diagnostic dotted: ok");
        const WGSLNode *d = p.ast.root->children[0];
        CHECK(wgsl_diag_rule_parts(d) == 2, "diag dotted: 2 rule parts");
        CHECK(d->child_count == 3,           "diag dotted: 3 children");
        CHECK(span_eq(d->children[1], &p.src, "ext"),                    "diag dotted: r1");
        CHECK(span_eq(d->children[2], &p.src, "subgroup_uniformity"),    "diag dotted: r2");
        done(&p);
    }

    /* — Mixed module: directives + decls — */
    {
        P p; run(&p,
            "enable f16;"
            "diagnostic(off, derivative_uniformity);"
            "const X: f32 = 1.0;"
            "fn main() { }");
        CHECK(p.ok, "mixed: ok");
        CHECK(p.ast.root->child_count == 4, "mixed: 4 decls");
        CHECK(p.ast.root->children[0]->kind == WGSL_NODE_DIR_ENABLE,     "mixed: 0");
        CHECK(p.ast.root->children[1]->kind == WGSL_NODE_DIR_DIAGNOSTIC, "mixed: 1");
        CHECK(p.ast.root->children[2]->kind == WGSL_NODE_DECL_CONST,     "mixed: 2");
        CHECK(p.ast.root->children[3]->kind == WGSL_NODE_DECL_FUNCTION,  "mixed: 3");
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_directives\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_directives  %d check(s) failed\n", fail);
    return 1;
}
