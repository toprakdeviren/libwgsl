/**
 * Phase 5 — Resolver + scope + predeclared identifiers.
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
 */
#include "internal/resolver.h"
#include "internal/parser.h"
#include "internal/lexer.h"
#include "internal/types.h"
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

/* Find the Nth EXPR_IDENT (depth-first, pre-order) in the tree.  -1 → not found. */
static const WGSLNode *find_ident_by_text(
    const WGSLNode *n, const WGSLSource *src, const char *want)
{
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(n->payload[0] >> 32);
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

    /* ── Predeclared scope contents ────────────────────────────── */
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

    /* ── Predeclared scalar lookup binds the right TypeInfo ──── */
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

    /* ── Module-scope forward references ───────────────────────── */
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

    /* ── Module-scope redeclaration → error ────────────────────── */
    {
        P p; run(&p,
            "const FOO: i32 = 1;\n"
            "fn FOO() {}\n");
        CHECK(!p.resolve_ok, "redecl: must error");
        CHECK(has_error_with(&p.diag, "redeclaration"),
              "redecl: 'redeclaration' diagnostic");
        done(&p);
    }

    /* ── Shadow predeclared → error ────────────────────────────── */
    {
        P p; run(&p,
            "const i32: i32 = 0;\n");
        CHECK(!p.resolve_ok, "shadow-pred: must error");
        CHECK(has_error_with(&p.diag, "shadows a predeclared"),
              "shadow-pred: diagnostic");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let bool = 1; }\n");
        CHECK(!p.resolve_ok, "shadow-pred-fn: must error");
        CHECK(has_error_with(&p.diag, "shadows a predeclared"),
              "shadow-pred-fn: diagnostic");
        done(&p);
    }

    /* ── Parameters bind into fn scope ─────────────────────────── */
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

    /* ── let / var bind in fn scope ────────────────────────────── */
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

    /* ── Use-before-decl in fn body → undeclared error ─────────── */
    {
        P p; run(&p,
            "fn f() { _ = w; let w = 1; }\n");
        CHECK(!p.resolve_ok, "use-before-decl: must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "use-before-decl: 'undeclared' diagnostic");
        done(&p);
    }

    /* ── Block scope: inner shadows outer ──────────────────────── */
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

    /* ── for-loop introduces a scope ───────────────────────────── */
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

    /* ── while / loop bodies are scopes ────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  loop { let q = 1; if (q == 1) { break; } }\n"
            "}\n");
        CHECK(p.resolve_ok, "loop-scope: resolve_ok");
        done(&p);
    }

    /* ── Undeclared identifier ──────────────────────────────────── */
    {
        P p; run(&p, "fn f() { _ = ghost; }\n");
        CHECK(!p.resolve_ok, "undecl: must error");
        CHECK(has_error_with(&p.diag, "undeclared"),
              "undecl: 'undeclared' diagnostic");
        done(&p);
    }

    /* ── Member-access RHS NOT resolved ─────────────────────────── */
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

    /* ── Attribute arg idents are silent on miss ───────────────── */
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

    /* ── Templated ident resolves leading name ─────────────────── */
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

    /* ── Module-scope const referencing predeclared + decl ─────── */
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

    /* ── Struct member-types resolve, member names are namespaced ─ */
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

    fprintf(stderr, "%s  test_resolver  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
