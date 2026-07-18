/**
 * Resolver, scope, and predeclared identifier tests.
 *
 * Covers:
 *   - predeclared scope: scalars / vec aliases / mat aliases / typegens /
 *     address spaces / access modes / a sample of builtin functions
 *   - module-scope forward references (resolve before declaration is OK)
 *   - module-scope redeclaration → error
 *   - shadowing predeclared at any scope → error
 *   - function-scope let / var / const binding + use
 *   - function parameters bind into fn scope
 *   - block scope: inner shadows outer; outer name still visible after `}`
 *   - for / while / loop introduce scopes
 *   - undeclared identifier → error
 *   - member-access RHS is *not* resolved (deferred to type checker)
 *   - attribute argument idents do not error on miss (e.g. `linear`,
 *     `vertex`, `position`)
 *   - templated_ident: leading ident is resolved, args are walked
 *   - §3.7.1: same code-point sequence only (NFC not used for compare)
 */
#include "internal/resolver.h"
#include "internal/parser.h"
#include "internal/lexer.h"
#include "internal/types.h"
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
    WGSLTypeStore  types;
    WGSLResolver   res;
    int            parse_ok;
    int            resolve_ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    p->parse_ok = lex_ok &&
                  wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    wgsl_types_init(&p->types, &p->arena);
    p->resolve_ok = wgsl_resolve(&p->ast, &p->src, &p->arena,
                                 &p->diag, &p->types, &p->res);
}

static void done(P *p) {
    wgsl_resolver_destroy(&p->res);
    wgsl_types_destroy(&p->types);
    wgsl_diag_destroy(&p->diag);
    wgsl_source_destroy(&p->src);
    wgsl_arena_destroy(&p->arena);
}

static int has_error_with(const WGSLDiagBag *b, const char *needle) {
    for (size_t i = 0; i < b->count; i++) {
        if (strstr(b->items[i].message, needle)) return 1;
    }
    return 0;
}

static char *build_large_struct_source(uint32_t fields, int duplicate_last) {
    size_t cap = 64u + (size_t)fields * 32u;
    char *src = (char *)malloc(cap);
    if (!src) return NULL;
    size_t pos = 0;
    int n = snprintf(src + pos, cap - pos, "struct S {\n");
    if (n < 0 || (size_t)n >= cap - pos) { free(src); return NULL; }
    pos += (size_t)n;
    for (uint32_t i = 0; i < fields; i++) {
        uint32_t name_ix = (duplicate_last && i + 1u == fields) ? 0u : i;
        n = snprintf(src + pos, cap - pos, "  f%u: i32%s\n",
                     (unsigned)name_ix, i + 1u < fields ? "," : "");
        if (n < 0 || (size_t)n >= cap - pos) { free(src); return NULL; }
        pos += (size_t)n;
    }
    n = snprintf(src + pos, cap - pos, "}\n");
    if (n < 0 || (size_t)n >= cap - pos) { free(src); return NULL; }
    return src;
}

/* Find the Nth EXPR_IDENT (depth-first, pre-order) in the tree.  -1 → not found. */
static const WGSLNode *find_ident_by_text(
    const WGSLNode *n, const WGSLSource *src, const char *want)
{
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t off = wgsl_node_name_span(n).offset;
        uint32_t len = wgsl_node_name_span(n).length;
        size_t wl = strlen(want);
        if (len == wl && memcmp(src->bytes + off, want, wl) == 0) {
            return n;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = find_ident_by_text(n->children[i], src, want);
        if (hit) return hit;
    }
    return NULL;
}

int main(void) {
    wgsl_utf8_init();

    /* Predeclared scope contents. */
    {
        P p; run(&p, "fn main() {}");
        CHECK(p.parse_ok && p.resolve_ok, "predecl: clean parse + resolve");

        const struct { const char *name; WGSLSymKind kind; } expect[] = {
            { "bool",       WGSL_SYM_PREDECLARED_TYPE       },
            { "i32",        WGSL_SYM_PREDECLARED_TYPE       },
            { "u32",        WGSL_SYM_PREDECLARED_TYPE       },
            { "f32",        WGSL_SYM_PREDECLARED_TYPE       },
            { "f16",        WGSL_SYM_PREDECLARED_TYPE       },
            { "vec3",       WGSL_SYM_PREDECLARED_TYPEGEN    },
            { "mat4x4",     WGSL_SYM_PREDECLARED_TYPEGEN    },
            { "atomic",     WGSL_SYM_PREDECLARED_TYPEGEN    },
            { "array",      WGSL_SYM_PREDECLARED_TYPEGEN    },
            { "ptr",        WGSL_SYM_PREDECLARED_TYPEGEN    },
            { "vec3f",      WGSL_SYM_PREDECLARED_TYPE_ALIAS },
            { "vec4u",      WGSL_SYM_PREDECLARED_TYPE_ALIAS },
            { "mat4x4f",    WGSL_SYM_PREDECLARED_TYPE_ALIAS },
            { "mat2x2h",    WGSL_SYM_PREDECLARED_TYPE_ALIAS },
            { "storage",    WGSL_SYM_PREDECLARED_ADDR_SPACE },
            { "uniform",    WGSL_SYM_PREDECLARED_ADDR_SPACE },
            { "workgroup",  WGSL_SYM_PREDECLARED_ADDR_SPACE },
            { "private",    WGSL_SYM_PREDECLARED_ADDR_SPACE },
            { "function",   WGSL_SYM_PREDECLARED_ADDR_SPACE },
            { "read",       WGSL_SYM_PREDECLARED_ACCESS_MODE},
            { "write",      WGSL_SYM_PREDECLARED_ACCESS_MODE},
            { "read_write", WGSL_SYM_PREDECLARED_ACCESS_MODE},
            { "sin",        WGSL_SYM_PREDECLARED_FN         },
            { "dot",        WGSL_SYM_PREDECLARED_FN         },
            { "textureSample", WGSL_SYM_PREDECLARED_FN      },
            { "atomicAdd",  WGSL_SYM_PREDECLARED_FN         },
            { "workgroupBarrier", WGSL_SYM_PREDECLARED_FN   },
        };
        for (size_t i = 0; i < sizeof expect / sizeof expect[0]; i++) {
            int found = 0;
            for (size_t j = 0; j < p.res.predeclared.count; j++) {
                WGSLSymbol *s = p.res.predeclared.items[j];
                if (s->name_len == strlen(expect[i].name) &&
                    memcmp(s->name, expect[i].name, s->name_len) == 0)
                {
                    if (s->kind == expect[i].kind) found = 1;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr,
                    "FAIL predecl: '%s' missing or wrong kind\n",
                    expect[i].name);
                fail += 1;
            }
        }
        /* Total predeclared cardinality: a smoke gate. */
        CHECK(p.res.predeclared.count >= 130,
              "predecl: at least 130 entries (5 scalars + ~35 typegens"
              " + 30 aliases + 7 addr + 3 access + ~130 fns)");
        done(&p);
    }

    /* Predeclared scalar lookup binds the right TypeInfo. */
    {
        P p; run(&p, "fn main() { let x: i32 = 0; }");
        CHECK(p.resolve_ok, "i32: resolve_ok");
        const WGSLNode *id = find_ident_by_text(p.ast.root, &p.src, "i32");
        CHECK(id != NULL, "i32: ident node found");
        if (id) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(id);
            CHECK(s != NULL, "i32: resolved");
            if (s) {
                CHECK(s->kind == WGSL_SYM_PREDECLARED_TYPE, "i32: kind");
                CHECK(s->type == p.res.types->t_i32, "i32: TypeInfo bound");
            }
        }
        done(&p);
    }

    /* Module-scope forward references. */
    {
        P p; run(&p,
            "fn caller() { _ = callee(1); }\n"
            "fn callee(x: i32) -> i32 { return x; }\n");
        CHECK(p.resolve_ok, "fwd-ref: resolve_ok");
        const WGSLNode *id = find_ident_by_text(p.ast.root, &p.src, "callee");
        CHECK(id != NULL, "fwd-ref: callee ident found");
        if (id) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(id);
            CHECK(s && s->kind == WGSL_SYM_FUNCTION, "fwd-ref: callee → FUNCTION");
        }
        done(&p);
    }

    /* Module-scope redeclaration -> error. */
    {
        P p; run(&p,
            "const FOO: i32 = 1;\n"
            "fn FOO() {}\n");
        CHECK(!p.resolve_ok, "redecl: must error");
        CHECK(has_error_with(&p.diag, "redeclaration"),
              "redecl: 'redeclaration' diagnostic");
        done(&p);
    }

    /* Shadow predeclared -> allowed (WGSL §5.1). */
    {
        P p; run(&p,
            "const i32: i32 = 0;\n");
        CHECK(p.resolve_ok, "shadow-pred: allowed");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let bool = 1; }\n");
        CHECK(p.resolve_ok, "shadow-pred-fn: allowed");
        done(&p);
    }

    /* Parameters bind into fn scope. */
    {
        P p; run(&p,
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n");
        CHECK(p.resolve_ok, "param: resolve_ok");
        const WGSLNode *use_a = find_ident_by_text(p.ast.root, &p.src, "a");
        const WGSLNode *use_b = find_ident_by_text(p.ast.root, &p.src, "b");
        CHECK(use_a && wgsl_node_resolved_symbol(use_a) &&
              wgsl_node_resolved_symbol(use_a)->kind == WGSL_SYM_PARAM,
              "param: a → PARAM");
        CHECK(use_b && wgsl_node_resolved_symbol(use_b) &&
              wgsl_node_resolved_symbol(use_b)->kind == WGSL_SYM_PARAM,
              "param: b → PARAM");
        done(&p);
    }

    /* decl_htab keys declarations by AST pointer, not spelling. */
    {
        P p; run(&p,
            "fn f(x: i32) -> i32 { return x; }\n"
            "fn g(x: i32) -> i32 { return x + 1; }\n");
        CHECK(p.resolve_ok, "decl_htab dup-name: resolve_ok");
        WGSLNode *f = p.ast.root && p.ast.root->child_count >= 1
            ? p.ast.root->children[0] : NULL;
        WGSLNode *g = p.ast.root && p.ast.root->child_count >= 2
            ? p.ast.root->children[1] : NULL;
        CHECK(f && g, "decl_htab dup-name: function decls found");
        if (f && g) {
            uint32_t ff = wgsl_fn_attr_count(f);
            uint32_t gf = wgsl_fn_attr_count(g);
            WGSLNode *fx = ff < f->child_count ? f->children[ff] : NULL;
            WGSLNode *gx = gf < g->child_count ? g->children[gf] : NULL;
            WGSLSymbol *fs = wgsl_resolver_symbol_for_decl(&p.res, f);
            WGSLSymbol *gs = wgsl_resolver_symbol_for_decl(&p.res, g);
            WGSLSymbol *fxs = wgsl_resolver_symbol_for_decl(&p.res, fx);
            WGSLSymbol *gxs = wgsl_resolver_symbol_for_decl(&p.res, gx);
            CHECK(fs && gs && fs != gs, "decl_htab dup-name: functions distinct");
            CHECK(fxs && gxs && fxs != gxs, "decl_htab dup-name: params distinct");
            CHECK(wgsl_resolver_index_for_symbol(&p.res, fxs) >= 0,
                  "decl_htab dup-name: first param index");
            CHECK(wgsl_resolver_index_for_symbol(&p.res, gxs) >= 0,
                  "decl_htab dup-name: second param index");
        }
        done(&p);
    }

    /* let / var bind in fn scope. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let x = 1;\n"
            "  var y = 2;\n"
            "  let z = x + y;\n"
            "  _ = z;\n"
            "}\n");
        CHECK(p.resolve_ok, "let/var: resolve_ok");
        const WGSLNode *use_x = find_ident_by_text(p.ast.root, &p.src, "x");
        const WGSLNode *use_y = find_ident_by_text(p.ast.root, &p.src, "y");
        const WGSLNode *use_z = find_ident_by_text(p.ast.root, &p.src, "z");
        CHECK(use_x && wgsl_node_resolved_symbol(use_x) &&
              wgsl_node_resolved_symbol(use_x)->kind == WGSL_SYM_LET,
              "let/var: x → LET");
        CHECK(use_y && wgsl_node_resolved_symbol(use_y) &&
              wgsl_node_resolved_symbol(use_y)->kind == WGSL_SYM_VAR,
              "let/var: y → VAR");
        CHECK(use_z && wgsl_node_resolved_symbol(use_z) != NULL,
              "let/var: z resolves");
        done(&p);
    }

    /* Use-before-decl in fn body -> undeclared error. */
    {
        P p; run(&p,
            "fn f() { _ = w; let w = 1; }\n");
        CHECK(!p.resolve_ok, "use-before-decl: must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "use-before-decl: 'undeclared' diagnostic");
        done(&p);
    }

    /* Block scope: inner shadows outer. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let x: i32 = 1;\n"
            "  { let x: i32 = 2; _ = x; }\n"
            "  _ = x;\n"
            "}\n");
        CHECK(p.resolve_ok, "block-shadow: resolve_ok");
        done(&p);
    }

    /* for-loop introduces a scope. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  for (var i = 0; i < 4; i = i + 1) { _ = i; }\n"
            "}\n");
        CHECK(p.resolve_ok, "for-scope: resolve_ok");
        done(&p);
    }

    /* — `i` from for-init is NOT visible after the loop — */
    {
        P p; run(&p,
            "fn f() {\n"
            "  for (var i = 0; i < 4; i = i + 1) { _ = i; }\n"
            "  _ = i;\n"
            "}\n");
        CHECK(!p.resolve_ok, "for-scope-leak: must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "for-scope-leak: undeclared 'i'");
        done(&p);
    }

    /* while / loop bodies are scopes. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  loop { let q = 1; if (q == 1) { break; } }\n"
            "}\n");
        CHECK(p.resolve_ok, "loop-scope: resolve_ok");
        done(&p);
    }

    /* Undeclared identifier. */
    {
        P p; run(&p, "fn f() { _ = ghost; }\n");
        CHECK(!p.resolve_ok, "undecl: must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "undecl: 'undeclared' diagnostic");
        done(&p);
    }

    /* Member-access RHS NOT resolved. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec3<f32>(0.0);\n"
            "  _ = v.x;\n"
            "}\n");
        CHECK(p.resolve_ok, "member-rhs: resolve_ok");
        const WGSLNode *use_v = find_ident_by_text(p.ast.root, &p.src, "v");
        CHECK(use_v && wgsl_node_resolved_symbol(use_v) &&
              wgsl_node_resolved_symbol(use_v)->kind == WGSL_SYM_LET,
              "member-rhs: v → LET");
        done(&p);
    }

    /* Attribute arg idents are silent on miss. */
    {
        P p; run(&p,
            "@vertex fn vmain(@builtin(vertex_index) vi: u32) "
            "-> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(p.resolve_ok,
              "attr-args: enumerants like 'vertex_index' don't error");
        done(&p);
    }

    /* Templated ident resolves leading name. */
    {
        P p; run(&p,
            "fn f() { let v: vec3<f32> = vec3<f32>(0.0); _ = v; }\n");
        CHECK(p.resolve_ok, "templ: resolve_ok");
        const WGSLNode *id = find_ident_by_text(p.ast.root, &p.src, "vec3");
        CHECK(id != NULL, "templ: 'vec3' ident found");
        if (id) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(id);
            CHECK(s && s->kind == WGSL_SYM_PREDECLARED_TYPEGEN,
                  "templ: vec3 → TYPEGEN");
        }
        done(&p);
    }

    /* Module-scope const referencing predeclared + decl. */
    {
        P p; run(&p,
            "const N: u32 = 4u;\n"
            "alias Buf = array<f32, N>;\n");
        CHECK(p.resolve_ok, "module-const: resolve_ok");
        const WGSLNode *use_n = find_ident_by_text(p.ast.root, &p.src, "N");
        CHECK(use_n && wgsl_node_resolved_symbol(use_n) &&
              wgsl_node_resolved_symbol(use_n)->kind == WGSL_SYM_CONST,
              "module-const: N → CONST");
        done(&p);
    }

    /* Struct member-types resolve, member names are namespaced. */
    {
        P p; run(&p,
            "struct Vertex {\n"
            "  pos: vec3<f32>,\n"
            "  uv:  vec2<f32>,\n"
            "}\n"
            "fn f(v: Vertex) -> vec3<f32> { return v.pos; }\n");
        CHECK(p.resolve_ok, "struct: resolve_ok");
        const WGSLNode *use_vt = find_ident_by_text(p.ast.root, &p.src, "Vertex");
        CHECK(use_vt && wgsl_node_resolved_symbol(use_vt) &&
              wgsl_node_resolved_symbol(use_vt)->kind == WGSL_SYM_STRUCT,
              "struct: Vertex → STRUCT");
        done(&p);
    }

    {
        char *src = build_large_struct_source(4096u, 0);
        CHECK(src != NULL, "struct hash: build unique source");
        if (src) {
            P p; run(&p, src);
            CHECK(p.resolve_ok, "struct hash: 4096 unique members");
            done(&p);
            free(src);
        }
    }
    {
        char *src = build_large_struct_source(4096u, 1);
        CHECK(src != NULL, "struct hash: build duplicate source");
        if (src) {
            P p; run(&p, src);
            CHECK(!p.resolve_ok, "struct hash: duplicate still errors");
            CHECK(has_error_with(&p.diag, "duplicate struct member 'f0'"),
                  "struct hash: resolver duplicate diag");
            done(&p);
            free(src);
        }
    }

    /* §5.2 function-scope resolution order (locks in spec behaviour;
     *    do NOT "fix" this to make case A an error — WGSL local scope begins
     *    AFTER the declaration, so an earlier use binds to the outer decl,
     *    it is not a JavaScript-style temporal-dead-zone error). */
    {
        /* Case A: outer `const x`, a later local `let x`.  The earlier
         * `let y = x` binds to the MODULE const (local not yet in scope). */
        P p; run(&p,
            "const x: i32 = 1;\n"
            "fn f() { let y = x; let x = 2; _ = y; _ = x; }\n");
        CHECK(p.resolve_ok, "§5.2 A: use-before-later-local resolves (no error)");
        CHECK(!has_error_with(&p.diag, "undeclared"), "§5.2 A: no undeclared error");
        const WGSLNode *xid = find_ident_by_text(p.ast.root, &p.src, "x");
        CHECK(xid != NULL, "§5.2 A: found `x` use");
        if (xid) {
            WGSLSymbol *s = wgsl_node_resolved_symbol(xid);
            CHECK(s && s->kind == WGSL_SYM_CONST,
                  "§5.2 A: `x` binds to module const, not the later local let");
        }
        done(&p);
    }
    {
        /* Case B: no outer `x`; using it before its only (local) declaration
         * is a use-before-declaration → undeclared. */
        P p; run(&p, "fn f() { let y = x; let x = 2; _ = y; _ = x; }\n");
        CHECK(!p.resolve_ok, "§5.2 B: use-before-decl (no outer) must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "§5.2 B: use-before-decl → undeclared identifier");
        done(&p);
    }
    {
        /* Case C: two declarations of the same name in one scope is illegal
         * (WGSL forbids same-scope redeclaration — sequential shadowing is
         * only allowed across nested blocks). */
        P p; run(&p, "fn f() { let x = 1; let x = x + 1; _ = x; }\n");
        CHECK(!p.resolve_ok, "§5.2 C: same-scope redeclaration must error");
        CHECK(has_error_with(&p.diag, "redeclaration"),
              "§5.2 C: same-scope redeclaration diagnostic");
        done(&p);
    }

    /* §3.7.1 Identifier Comparison (pinned CRD).
     * Same iff same sequence of code points. Spec Note: does *not*
     * permit Unicode normalization for comparison. U+00E9 (UTF-8 C3 A9)
     * vs U+0065 U+0301 (e + combining acute, 65 CC 81) are NFC-equal
     * but MUST be distinct WGSL identifiers. memcmp is correct. */
    {
        static const char src[] =
            "const \xC3\xA9: i32 = 1;\n"          /* U+00E9 */
            "const e\xCC\x81: i32 = 2;\n"       /* e + U+0301 */
            "fn f() { let _a = \xC3\xA9; let _b = e\xCC\x81; }\n";
        P p; run(&p, src);
        CHECK(p.parse_ok, "§3.7.1: NFC-distinct spellings parse");
        CHECK(p.resolve_ok,
              "§3.7.1: distinct code-point sequences → two symbols (no error)");
        CHECK(!has_error_with(&p.diag, "redeclaration"),
              "§3.7.1: must NOT treat NFC-equal spellings as redeclaration");
        CHECK(!has_error_with(&p.diag, "undeclared"),
              "§3.7.1: each spelling resolves to its own declaration");
        done(&p);
    }
    {
        /* Use of the other spelling must not bind across the pair. */
        static const char src[] =
            "const \xC3\xA9: i32 = 1;\n"
            "fn f() { let _x = e\xCC\x81; }\n";  /* NFD form undeclared */
        P p; run(&p, src);
        CHECK(p.parse_ok, "§3.7.1 cross-spell: parse");
        CHECK(!p.resolve_ok,
              "§3.7.1: NFD use does not resolve to NFC declaration");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "§3.7.1: cross-normalization lookup → undeclared");
        done(&p);
    }

    fprintf(stderr, "%s  test_resolver  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
