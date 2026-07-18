/**
 * Parser smoke coverage for the hello-triangle example.
 *
 * Asserts on `examples/hello-triangle.wgsl`:
 *   - lex + parse return ok (no error diagnostics)
 *   - root has exactly 2 function decls
 *   - first fn is `vs_main` with 1 fn-attr, 1 param, 1 ret-attr, ret-type
 *   - second fn is `fs_main` with 1 fn-attr, 0 params, 1 ret-attr, ret-type
 *   - vs_main body has [let, return] (2 stmts)
 *   - vs_main's let initializer is a CALL to a TEMPLATED_IDENT (`array<…>`)
 *   - vs_main's return value is a CALL to `vec4f`
 *   - fs_main body has [return]
 *   - the AST dump succeeds (smoke for the printer)
 */
#include "internal/ast_dump.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HELLO_PATH "examples/hello-triangle.wgsl"

static int fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static char *slurp(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL  cannot open %s (run from wgsl/ root?)\n", path);
        return NULL;
    }
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

static int name_eq(const WGSLNode *n, const WGSLSource *s, const char *want) {
    uint32_t off = wgsl_node_name_span(n).offset;
    uint32_t len = wgsl_node_name_span(n).length;
    size_t wl = strlen(want);
    if (len != wl) return 0;
    return memcmp(s->bytes + off, want, wl) == 0;
}

int main(void) {
    wgsl_utf8_init();

    size_t len = 0;
    char *src = slurp(HELLO_PATH, &len);
    if (!src) return 1;

    WGSLArena arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;    wgsl_diag_init(&diag);
    WGSLSource source;   wgsl_source_init(&source, src, len);
    WGSLLexResult lex = {0};
    WGSLAst ast       = {0};

    int lex_ok = wgsl_tokenize(&source, &arena, &diag, &lex);
    int parse_ok = wgsl_parse(&lex, &source, &arena, &diag, &ast);

    CHECK(lex_ok && parse_ok,             "lex + parse ok");
    CHECK(!wgsl_diag_has_error(&diag),    "no error diagnostic");
    CHECK(ast.root != NULL,               "root non-null");
    CHECK(ast.root->kind == WGSL_NODE_TRANSLATION_UNIT, "root is TU");
    CHECK(ast.root->child_count == 2,     "TU has 2 decls");

    if (ast.root->child_count == 2) {
        const WGSLNode *vs = ast.root->children[0];
        const WGSLNode *fs = ast.root->children[1];

        CHECK(vs->kind == WGSL_NODE_DECL_FUNCTION, "vs is fn");
        CHECK(name_eq(vs, &source, "vs_main"),     "vs name = vs_main");
        uint32_t vs_fa  = wgsl_fn_attr_count(vs);
        uint32_t vs_p   = wgsl_fn_param_count(vs);
        uint32_t vs_ra  = wgsl_fn_ret_attr_count(vs);
        uint32_t vs_hr  = wgsl_fn_has_return_type(vs);
        CHECK(vs_fa == 1 && vs_p == 1 && vs_ra == 1 && vs_hr == 1,
              "vs counts: 1 fn-attr / 1 param / 1 ret-attr / ret-type");

        CHECK(fs->kind == WGSL_NODE_DECL_FUNCTION, "fs is fn");
        CHECK(name_eq(fs, &source, "fs_main"),     "fs name = fs_main");
        uint32_t fs_fa  = wgsl_fn_attr_count(fs);
        uint32_t fs_p   = wgsl_fn_param_count(fs);
        uint32_t fs_ra  = wgsl_fn_ret_attr_count(fs);
        uint32_t fs_hr  = wgsl_fn_has_return_type(fs);
        CHECK(fs_fa == 1 && fs_p == 0 && fs_ra == 1 && fs_hr == 1,
              "fs counts: 1 fn-attr / 0 params / 1 ret-attr / ret-type");

        /* vs_main body: last child of vs */
        const WGSLNode *vs_body = vs->children[vs->child_count - 1];
        CHECK(vs_body->kind == WGSL_NODE_STMT_COMPOUND,    "vs body is compound");
        CHECK(vs_body->child_count == 2,                    "vs body has 2 stmts");

        const WGSLNode *let = vs_body->children[0];
        CHECK(let->kind == WGSL_NODE_DECL_LET,              "vs body[0] = let");
        CHECK(name_eq(let, &source, "pos"),                 "let name = pos");
        const WGSLNode *let_init = let->children[let->child_count - 1];
        CHECK(let_init->kind == WGSL_NODE_EXPR_CALL,        "let init = call");
        const WGSLNode *let_callee = let_init->children[0];
        CHECK(let_callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT,
              "let callee = templated_ident (array<...>)");

        const WGSLNode *ret = vs_body->children[1];
        CHECK(ret->kind == WGSL_NODE_STMT_RETURN,           "vs body[1] = return");
        const WGSLNode *retval = ret->children[0];
        CHECK(retval->kind == WGSL_NODE_EXPR_CALL,          "vs return = call");

        /* fs_main body */
        const WGSLNode *fs_body = fs->children[fs->child_count - 1];
        CHECK(fs_body->kind == WGSL_NODE_STMT_COMPOUND,     "fs body is compound");
        CHECK(fs_body->child_count == 1,                    "fs body has 1 stmt");
        const WGSLNode *fs_ret = fs_body->children[0];
        CHECK(fs_ret->kind == WGSL_NODE_STMT_RETURN,        "fs body[0] = return");
        const WGSLNode *fs_retval = fs_ret->children[0];
        CHECK(fs_retval->kind == WGSL_NODE_EXPR_CALL,       "fs return = call");
        CHECK(fs_retval->child_count == 5,
              "fs return: 1 callee + 4 float args");
    }

    /* Smoke: dump succeeds */
    if (fail == 0) {
        fprintf(stderr, "--- AST dump (hello-triangle) ---\n");
        wgsl_ast_dump(ast.root, &source, stderr);
        fprintf(stderr, "----------------------------------\n");
        printf("PASS  test_parser_hello  (file=%zu B, decls=%u)\n",
               len, ast.root->child_count);
    } else {
        fprintf(stderr, "FAIL  test_parser_hello  %d check(s) failed\n", fail);
    }

    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&source);
    wgsl_arena_destroy(&arena);
    free(src);

    return fail ? 1 : 0;
}
