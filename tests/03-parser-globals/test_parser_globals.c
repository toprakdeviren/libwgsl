/**
 * Phase 3 Round B Iter 2 — module-scope declarations.
 *
 * Covers:
 *   - struct (with members, attributes on members)
 *   - alias
 *   - module-scope const
 *   - override (no init / with init / with attributes)
 *   - module-scope var (no template / 1-arg template / 2-arg template
 *                       / with attributes)
 *   - module-scope const_assert
 *   - error: attributes on a kind that doesn't take them
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

static int name_eq(const WGSLNode *n, const WGSLSource *s, const char *want) {
    uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(n->payload[0] >> 32);
    size_t wl = strlen(want);
    if (len != wl) return 0;
    return memcmp(s->bytes + off, want, wl) == 0;
}

int main(void) {
    wgsl_utf8_init();

    /* — struct — */
    {
        P p; run(&p,
            "struct Vertex {"
            "  @location(0) pos: vec3<f32>,"
            "  @location(1) uv: vec2<f32>,"
            "  color: vec4<f32>,"
            "}");
        CHECK(p.ok, "struct: ok");
        const WGSLNode *s = p.ast.root->children[0];
        CHECK(s->kind == WGSL_NODE_DECL_STRUCT, "struct: kind");
        CHECK(name_eq(s, &p.src, "Vertex"),    "struct: name");
        CHECK(s->child_count == 3,              "struct: 3 members");
        const WGSLNode *m0 = s->children[0];
        CHECK(m0->kind == WGSL_NODE_DECL_STRUCT_MEMBER, "m0: kind");
        CHECK(name_eq(m0, &p.src, "pos"),               "m0: name");
        uint32_t m0_attrs = (uint32_t)m0->payload[1];
        CHECK(m0_attrs == 1,                            "m0: 1 attribute");
        const WGSLNode *m2 = s->children[2];
        CHECK(name_eq(m2, &p.src, "color"),             "m2: name");
        uint32_t m2_attrs = (uint32_t)m2->payload[1];
        CHECK(m2_attrs == 0,                            "m2: 0 attributes");
        done(&p);
    }

    /* — alias — */
    {
        P p; run(&p, "alias UV = vec2<f32>;");
        CHECK(p.ok, "alias: ok");
        const WGSLNode *a = p.ast.root->children[0];
        CHECK(a->kind == WGSL_NODE_DECL_ALIAS, "alias: kind");
        CHECK(name_eq(a, &p.src, "UV"),         "alias: name");
        CHECK(a->child_count == 1,              "alias: 1 child (type)");
        CHECK(a->children[0]->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT,
              "alias: child = templ");
        done(&p);
    }

    /* — module-scope const — */
    {
        P p; run(&p, "const PI: f32 = 3.14159;");
        CHECK(p.ok, "module const: ok");
        const WGSLNode *c = p.ast.root->children[0];
        CHECK(c->kind == WGSL_NODE_DECL_CONST, "module const: kind");
        CHECK(name_eq(c, &p.src, "PI"),         "module const: name");
        done(&p);
    }

    /* — override (no init) — */
    {
        P p; run(&p, "override threshold: f32;");
        CHECK(p.ok, "override no-init: ok");
        const WGSLNode *o = p.ast.root->children[0];
        CHECK(o->kind == WGSL_NODE_DECL_OVERRIDE, "override: kind");
        CHECK(name_eq(o, &p.src, "threshold"),     "override: name");
        uint32_t has_type = (uint32_t)(o->payload[2] & 0xFFFFFFFFu);
        uint32_t has_init = (uint32_t)(o->payload[2] >> 32);
        CHECK(has_type == 1 && has_init == 0, "override: type-only");
        done(&p);
    }
    /* — override with attribute and init — */
    {
        P p; run(&p, "@id(0) override factor: f32 = 2.0;");
        CHECK(p.ok, "override @id: ok");
        const WGSLNode *o = p.ast.root->children[0];
        CHECK(name_eq(o, &p.src, "factor"), "override @id: name");
        uint32_t attr_count = (uint32_t)(o->payload[1] & 0xFFFFFFFFu);
        uint32_t has_type   = (uint32_t)(o->payload[2] & 0xFFFFFFFFu);
        uint32_t has_init   = (uint32_t)(o->payload[2] >> 32);
        CHECK(attr_count == 1 && has_type == 1 && has_init == 1,
              "override @id: counts");
        const WGSLNode *attr = o->children[0];
        CHECK(attr->kind == WGSL_NODE_ATTRIBUTE, "override: child[0] is attr");
        done(&p);
    }

    /* — module-scope var with template + attrs — */
    {
        P p; run(&p, "@group(0) @binding(1) var<storage, read_write> buf: array<u32>;");
        CHECK(p.ok, "var<storage,read_write>: ok");
        const WGSLNode *v = p.ast.root->children[0];
        CHECK(v->kind == WGSL_NODE_DECL_VAR, "var: kind");
        CHECK(name_eq(v, &p.src, "buf"),      "var: name");
        uint32_t attr_count = (uint32_t)(v->payload[1] & 0xFFFFFFFFu);
        uint32_t tpl_count  = (uint32_t)(v->payload[1] >> 32);
        uint32_t has_type   = (uint32_t)(v->payload[2] & 0xFFFFFFFFu);
        uint32_t has_init   = (uint32_t)(v->payload[2] >> 32);
        CHECK(attr_count == 2, "var: 2 attrs (@group, @binding)");
        CHECK(tpl_count  == 2, "var: 2 tpl args (storage, read_write)");
        CHECK(has_type   == 1, "var: has_type");
        CHECK(has_init   == 0, "var: no_init");
        /* children: [attr, attr, tpl1, tpl2, type] = 5 */
        CHECK(v->child_count == 5, "var: 5 children");
        done(&p);
    }
    /* — module-scope var with single template arg — */
    {
        P p; run(&p, "var<private> counter: u32 = 0u;");
        CHECK(p.ok, "var<private>: ok");
        const WGSLNode *v = p.ast.root->children[0];
        uint32_t tpl_count  = (uint32_t)(v->payload[1] >> 32);
        uint32_t has_init   = (uint32_t)(v->payload[2] >> 32);
        CHECK(tpl_count == 1, "var<private>: 1 tpl arg");
        CHECK(has_init  == 1, "var<private>: has_init");
        done(&p);
    }

    /* — module-scope const_assert — */
    {
        P p; run(&p, "const_assert 1 < 2;");
        CHECK(p.ok, "module const_assert: ok");
        const WGSLNode *c = p.ast.root->children[0];
        CHECK(c->kind == WGSL_NODE_DECL_CONST_ASSERT, "module const_assert: kind");
        CHECK(c->child_count == 1, "module const_assert: 1 child");
        done(&p);
    }

    /* — Error: attribute on alias — */
    {
        P p; run(&p, "@vertex alias Bad = i32;");
        CHECK(!p.ok,                          "attr on alias: rejected");
        CHECK(wgsl_diag_has_error(&p.diag),   "attr on alias: diag fired");
        done(&p);
    }

    /* — Multi-decl module — */
    {
        P p; run(&p,
            "alias UV = vec2<f32>;"
            "const SIZE: u32 = 64u;"
            "struct V { p: vec3<f32>, }"
            "@group(0) @binding(0) var<uniform> u: V;"
            "fn main() { _ = u.p; }");
        CHECK(p.ok, "multi: ok");
        CHECK(p.ast.root->child_count == 5, "multi: 5 decls");
        CHECK(p.ast.root->children[0]->kind == WGSL_NODE_DECL_ALIAS,    "multi: 0 alias");
        CHECK(p.ast.root->children[1]->kind == WGSL_NODE_DECL_CONST,    "multi: 1 const");
        CHECK(p.ast.root->children[2]->kind == WGSL_NODE_DECL_STRUCT,   "multi: 2 struct");
        CHECK(p.ast.root->children[3]->kind == WGSL_NODE_DECL_VAR,      "multi: 3 var");
        CHECK(p.ast.root->children[4]->kind == WGSL_NODE_DECL_FUNCTION, "multi: 4 fn");
        done(&p);
    }

    if (fail == 0) {
        printf("PASS  test_parser_globals\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_parser_globals  %d check(s) failed\n", fail);
    return 1;
}
