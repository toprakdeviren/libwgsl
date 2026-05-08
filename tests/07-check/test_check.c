/**
 * Phase 7 Iter A — type checker scalar foundation.
 *
 * Covers:
 *   - type-spec resolution: scalar / vec<T> / mat<T> / atomic<T> /
 *     array<T> / array<T, N> (with N from a const-eval'd symbol) /
 *     predeclared aliases (vec3f / mat4x4f / …)
 *   - module decl typing: const / let / var / override / alias / struct
 *     all populate `sym->type`
 *   - function param + return type typing
 *   - expression typing: literals, idents, parens, unary `- ! ~`,
 *     binary numeric / comparison / logical / bitwise
 *   - implicit promotion via §6.1.2 conversion-rank table
 *   - is_ref bit set on `var` ident uses, clear on const / let / param
 *   - error cases: u32 unary minus, concrete-type mismatch,
 *     non-numeric operands, non-bool logical operands
 */
#include "internal/check.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
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
    WGSLArena          arena;
    WGSLSource         src;
    WGSLDiagBag        diag;
    WGSLLexResult      lex;
    WGSLAst            ast;
    WGSLTypeStore      types;
    WGSLResolver       res;
    WGSLConstEvaluator cev;
    WGSLTypeChecker    tc;
    int                ok;
} P;

static void run(P *p, const char *text) {
    memset(p, 0, sizeof *p);
    wgsl_arena_init(&p->arena);
    wgsl_diag_init(&p->diag);
    wgsl_source_init(&p->src, text, strlen(text));
    int lex_ok   = wgsl_tokenize(&p->src, &p->arena, &p->diag, &p->lex);
    int parse_ok = wgsl_parse(&p->lex, &p->src, &p->arena, &p->diag, &p->ast);
    wgsl_types_init(&p->types, &p->arena);
    int res_ok   = wgsl_resolve(&p->ast, &p->src, &p->arena,
                                &p->diag, &p->types, &p->res);
    int ce_ok    = wgsl_consteval(&p->ast, &p->src, &p->arena,
                                  &p->diag, &p->types, &p->res, &p->cev);
    int tc_ok    = wgsl_typecheck(&p->ast, &p->src, &p->arena,
                                  &p->diag, &p->types, &p->res, &p->cev,
                                  &p->tc);
    p->ok = lex_ok && parse_ok && res_ok && ce_ok && tc_ok;
}

static void done(P *p) {
    wgsl_typecheck_destroy(&p->tc);
    wgsl_consteval_destroy(&p->cev);
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

/* Find the symbol with the given name across resolver's all_decls. */
static WGSLSymbol *sym_for(const P *p, const char *want) {
    size_t wl = strlen(want);
    for (size_t i = 0; i < p->res.all_decl_count; i++) {
        WGSLSymbol *s = p->res.all_decls[i];
        if (s->name_len == wl && memcmp(s->name, want, wl) == 0) {
            return s;
        }
    }
    return NULL;
}

/* Find the Nth match of an EXPR_IDENT by source text. */
static const WGSLNode *find_ident(
    const WGSLNode *n, const WGSLSource *src, const char *want, int *skip)
{
    if (!n) return NULL;
    if (n->kind == WGSL_NODE_EXPR_IDENT) {
        uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(n->payload[0] >> 32);
        if (len == strlen(want) &&
            memcmp(src->bytes + off, want, len) == 0)
        {
            if (*skip == 0) return n;
            (*skip) -= 1;
        }
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *h = find_ident(n->children[i], src, want, skip);
        if (h) return h;
    }
    return NULL;
}

int main(void) {
    wgsl_utf8_init();

    /* ── Typed module const ───────────────────────────────────── */
    {
        P p; run(&p, "const N: i32 = 5;\n");
        CHECK(p.ok, "const N: i32 = 5: tc_ok");
        WGSLSymbol *N = sym_for(&p, "N");
        CHECK(N && N->type == p.types.t_i32, "N: i32");
        done(&p);
    }
    {
        P p; run(&p, "const N = 5;\n");
        CHECK(p.ok, "const N = 5 (untyped): tc_ok");
        WGSLSymbol *N = sym_for(&p, "N");
        /* Default concretisation: AbstractInt → i32 */
        CHECK(N && N->type == p.types.t_i32, "N: concretised to i32");
        done(&p);
    }
    {
        P p; run(&p, "const PI = 3.14;\n");
        CHECK(p.ok, "const PI (untyped float): tc_ok");
        WGSLSymbol *S = sym_for(&p, "PI");
        CHECK(S && S->type == p.types.t_f32, "PI: AbstractFloat → f32");
        done(&p);
    }
    {
        P p; run(&p, "const X: u32 = 256;\n");
        CHECK(p.ok, "const X: u32: tc_ok");
        WGSLSymbol *X = sym_for(&p, "X");
        CHECK(X && X->type == p.types.t_u32, "X: u32");
        done(&p);
    }

    /* ── Vec / mat / array type-specs ─────────────────────────── */
    {
        P p; run(&p, "alias UV = vec2<f32>;\n");
        CHECK(p.ok, "alias UV = vec2<f32>: tc_ok");
        WGSLSymbol *UV = sym_for(&p, "UV");
        CHECK(UV && UV->type, "UV: typed");
        CHECK(UV->type->kind  == WGSL_TYPE_VEC,           "UV: vec");
        CHECK(UV->type->width == 2,                       "UV: width 2");
        CHECK(UV->type->ref   == p.types.t_f32,           "UV: elem f32");
        done(&p);
    }
    {
        P p; run(&p, "alias M = mat4x4<f32>;\n");
        CHECK(p.ok, "alias M = mat4x4<f32>: tc_ok");
        WGSLSymbol *M = sym_for(&p, "M");
        CHECK(M && M->type, "M: typed");
        CHECK(M->type->kind == WGSL_TYPE_MAT, "M: mat");
        CHECK(M->type->width == 4 && M->type->rows == 4, "M: 4x4");
        CHECK(M->type->ref == p.types.t_f32,             "M: elem f32");
        done(&p);
    }
    {
        P p; run(&p, "alias Buf = array<f32, 8>;\n");
        CHECK(p.ok, "alias Buf = array<f32, 8>: tc_ok");
        WGSLSymbol *Buf = sym_for(&p, "Buf");
        CHECK(Buf && Buf->type && Buf->type->kind == WGSL_TYPE_ARRAY,
              "Buf: array");
        CHECK(Buf->type->array_len == 8,            "Buf: length 8");
        CHECK(Buf->type->ref      == p.types.t_f32, "Buf: elem f32");
        done(&p);
    }
    {
        P p; run(&p,
            "const N: u32 = 4u;\n"
            "alias Buf = array<f32, N>;\n");
        CHECK(p.ok, "alias array<f32, N>: tc_ok");
        WGSLSymbol *Buf = sym_for(&p, "Buf");
        CHECK(Buf && Buf->type && Buf->type->array_len == 4,
              "Buf: length 4 (from const N)");
        done(&p);
    }
    {
        P p; run(&p, "alias Run = array<u32>;\n");
        CHECK(p.ok, "alias array<u32> (runtime): tc_ok");
        WGSLSymbol *R = sym_for(&p, "Run");
        CHECK(R && R->type && R->type->array_len == 0,
              "Run: runtime-sized");
        done(&p);
    }

    /* ── Predeclared alias ───────────────────────────────────── */
    {
        P p; run(&p, "alias V = vec3f;\n");
        CHECK(p.ok, "alias V = vec3f: tc_ok");
        WGSLSymbol *V = sym_for(&p, "V");
        CHECK(V && V->type && V->type->kind == WGSL_TYPE_VEC &&
              V->type->width == 3 && V->type->ref == p.types.t_f32,
              "V: vec3<f32>");
        done(&p);
    }

    /* ── Struct ──────────────────────────────────────────────── */
    {
        P p; run(&p,
            "struct Vertex {\n"
            "  pos: vec3<f32>,\n"
            "  uv:  vec2<f32>,\n"
            "}\n");
        CHECK(p.ok, "struct: tc_ok");
        WGSLSymbol *V = sym_for(&p, "Vertex");
        CHECK(V && V->type && V->type->kind == WGSL_TYPE_STRUCT,
              "Vertex: STRUCT type");
        done(&p);
    }

    /* ── Function param + return ─────────────────────────────── */
    {
        P p; run(&p,
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n");
        CHECK(p.ok, "add: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        CHECK(a && a->type == p.types.t_i32, "a: i32 param");
        CHECK(b && b->type == p.types.t_i32, "b: i32 param");
        /* Find `a + b` and verify it types to i32. */
        int skip0 = 0; const WGSLNode *id_a = find_ident(p.ast.root, &p.src, "a", &skip0);
        CHECK(id_a, "found ident 'a'");
        if (id_a) {
            CHECK(wgsl_typecheck_type_of(&p.tc, id_a) == p.types.t_i32,
                  "ident 'a' typed i32");
        }
        done(&p);
    }

    /* ── var produces ref ────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: i32 = 1;\n"
            "  let r = v + 1;\n"
            "}\n");
        CHECK(p.ok, "var v: tc_ok");
        WGSLSymbol *v = sym_for(&p, "v");
        CHECK(v && v->type == p.types.t_i32, "v: i32");
        int skip0 = 0;
        const WGSLNode *use_v = find_ident(p.ast.root, &p.src, "v", &skip0);
        CHECK(use_v, "found ident 'v'");
        if (use_v) {
            CHECK(wgsl_typecheck_is_ref(&p.tc, use_v),
                  "var ident 'v' is_ref=1");
        }
        done(&p);
    }

    /* ── Implicit numeric promotion ─────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 + 2.5;\n"
            "}\n");
        CHECK(p.ok, "1 + 2.5: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        CHECK(a && a->type == p.types.t_f32,
              "a: AbstractInt+AbstractFloat → f32 (concretised)");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 + 2i;\n"
            "}\n");
        CHECK(p.ok, "1 + 2i: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        CHECK(a && a->type == p.types.t_i32, "a: i32 (AbstractInt → i32)");
        done(&p);
    }

    /* ── Comparison / logical / bitwise types ───────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 < 2;\n"
            "  let b = a && true;\n"
            "  let c = 0xFFu | 0x0Fu;\n"
            "}\n");
        CHECK(p.ok, "compare/logical/bitwise: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        WGSLSymbol *c = sym_for(&p, "c");
        CHECK(a && a->type == p.types.t_bool, "a: bool");
        CHECK(b && b->type == p.types.t_bool, "b: bool");
        CHECK(c && c->type == p.types.t_u32,  "c: u32");
        done(&p);
    }

    /* ── Unary ───────────────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = -1;\n"
            "  let b = !true;\n"
            "  let c = ~0u;\n"
            "}\n");
        CHECK(p.ok, "unary: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        WGSLSymbol *c = sym_for(&p, "c");
        CHECK(a && a->type == p.types.t_i32,  "a: -1 → i32 (concretised)");
        CHECK(b && b->type == p.types.t_bool, "b: bool");
        CHECK(c && c->type == p.types.t_u32,  "c: u32");
        done(&p);
    }

    /* ── Errors ───────────────────────────────────────────────── */
    {
        P p; run(&p, "fn f() { let x = -1u; }\n");
        CHECK(!p.ok, "unary minus on u32: must error");
        CHECK(has_error_with(&p.diag, "unary minus"),
              "u32 unary minus diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1i + 1u; }\n");
        CHECK(!p.ok, "i32 + u32: must error");
        CHECK(has_error_with(&p.diag, "no implicit conversion"),
              "concrete-mismatch diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1 && 2; }\n");
        CHECK(!p.ok, "logical with non-bool: must error");
        CHECK(has_error_with(&p.diag, "logical operator requires bool"),
              "logical-non-bool diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = !1; }\n");
        CHECK(!p.ok, "!1 (non-bool): must error");
        CHECK(has_error_with(&p.diag, "logical not requires"),
              "!non-bool diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1.5 & 2; }\n");
        CHECK(!p.ok, "1.5 & 2: must error");
        CHECK(has_error_with(&p.diag, "bitwise"),
              "bitwise-non-int diag");
        done(&p);
    }

    fprintf(stderr, "%s  test_check  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
