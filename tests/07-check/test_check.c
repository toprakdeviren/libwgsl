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
 *   - structural ref<AS,T,AM> type + is_ref bit on `var` ident uses,
 *     clear on const / let / param
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
    if (!p->ok) {
        for (size_t i = 0; i < p->diag.count; i++) {
            fprintf(stderr, "DIAG: %s\n", p->diag.items[i].message);
        }
    }
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

static const WGSLNode *find_kind(const WGSLNode *n, WGSLNodeKind kind) {
    if (!n) return NULL;
    if (n->kind == kind) return n;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *h = find_kind(n->children[i], kind);
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
        CHECK(N && N->type == p.types.t_abstract_int, "N: AbstractInt");
        done(&p);
    }
    {
        P p; run(&p, "const PI = 3.14;\n");
        CHECK(p.ok, "const PI (untyped float): tc_ok");
        WGSLSymbol *S = sym_for(&p, "PI");
        CHECK(S && S->type == p.types.t_abstract_float, "PI: AbstractFloat");
        done(&p);
    }
    {
        P p; run(&p, "const X: u32 = 256;\n");
        CHECK(p.ok, "const X: u32: tc_ok");
        WGSLSymbol *X = sym_for(&p, "X");
        CHECK(X && X->type == p.types.t_u32, "X: u32");
        done(&p);
    }
    {
        P p; run(&p,
            "alias U = u32;\n"
            "const X: U = -1;\n");
        CHECK(!p.ok, "const X: alias u32 = -1: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "alias-typed const range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const C = 5;\n"
            "fn f() { let x = C; }\n");
        CHECK(p.ok, "let from untyped const: tc_ok");
        WGSLSymbol *x = sym_for(&p, "x");
        CHECK(x && x->type == p.types.t_i32,
              "let x = const AbstractInt materializes to i32");
        done(&p);
    }

    /* ── Forward-ref module const / var ───────────────────────── */
    {
        P p; run(&p,
            "const A = B;\n"
            "const B: i32 = 5;\n");
        CHECK(p.ok, "const A = B (forward ref): tc_ok");
        WGSLSymbol *A = sym_for(&p, "A");
        CHECK(A && A->type == p.types.t_i32, "A: i32 (from forward ref B)");
        done(&p);
    }
    {
        P p; run(&p,
            "const A = B;\n"
            "const B = A;\n");
        CHECK(!p.ok, "cyclic const: must error");
        CHECK(has_error_with(&p.diag, "cyclic dependency"),
              "cyclic dependency diagnostic");
        done(&p);
    }
    {
        /* §7.3 — `var<private>` initializer must be a const- or
         * override-expression; referencing another `var` is rejected
         * (this test used to assert acceptance pre-§7.3 init gate). */
        P p; run(&p,
            "var<private> X = Y;\n"
            "var<private> Y = 10u;\n");
        CHECK(!p.ok, "var<private> X = (var) Y: must error");
        CHECK(has_error_with(&p.diag, "var<private> initializer must be a const- or override-expression"),
              "var<private> non-const init diag");
        done(&p);
    }
    {
        /* §7.3 — `var<private>` with a const-expr init is OK; with an
         * override-ident is also OK (override-expression). */
        P p; run(&p,
            "const K: u32 = 42;\n"
            "override O: u32 = 1;\n"
            "var<private> X: u32 = K;\n"
            "var<private> Y: u32 = O + K;\n"
            "var<private> Z: u32 = 7u;\n");
        CHECK(p.ok, "var<private> with const/override init: ok");
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
            WGSLTypeInfo *vt = wgsl_typecheck_type_of(&p.tc, use_v);
            CHECK(vt && vt->kind == WGSL_TYPE_REF &&
                  vt->ref == p.types.t_i32 &&
                  (vt->flags & 0xFFu) == WGSL_AS_FUNCTION &&
                  ((vt->flags >> 8) & 0xFFu) == WGSL_ACCESS_READ_WRITE,
                  "var ident 'v' type is ref<function, i32, read_write>");
        }
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: vec3<f32>;\n"
            "  v.x = 1.0;\n"
            "}\n");
        CHECK(p.ok, "single swizzle ref structural type: tc_ok");
        const WGSLNode *m = find_kind(p.ast.root, WGSL_NODE_EXPR_MEMBER);
        CHECK(m, "found member expression");
        if (m) {
            WGSLTypeInfo *mt = wgsl_typecheck_type_of(&p.tc, m);
            CHECK(mt && mt->kind == WGSL_TYPE_REF &&
                  mt->ref == p.types.t_f32 &&
                  (mt->flags & 0xFFu) == WGSL_AS_FUNCTION &&
                  ((mt->flags >> 8) & 0xFFu) == WGSL_ACCESS_READ_WRITE,
                  "single swizzle type is ref<function, f32, read_write>");
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

    /* ── Texture / sampler type-specs (§6.5) ─────────────────── */
    {
        /* Bare `sampler` resolves to a real type, not "is not a type". */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n");
        CHECK(p.ok, "var s : sampler: tc_ok");
        CHECK(!has_error_with(&p.diag, "is not a type"),
              "sampler must not trip 'is not a type'");
        WGSLSymbol *s = sym_for(&p, "s");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_SAMPLER,
              "s: typed as sampler");
        CHECK(s && s->type && s->type->width == 0,
              "s: non-comparison sampler");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n");
        CHECK(p.ok, "var sc : sampler_comparison: tc_ok");
        WGSLSymbol *s = sym_for(&p, "sc");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_SAMPLER &&
              s->type->width == 1,
              "sc: comparison sampler");
        done(&p);
    }
    {
        /* Sampled texture: texture_2d<f32>. */
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n");
        CHECK(p.ok, "var t : texture_2d<f32>: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "texture_2d<f32> must not trip 'has no type yet'");
        WGSLSymbol *s = sym_for(&p, "t");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE,
              "t: typed as texture");
        CHECK(s && s->type && s->type->width == WGSL_TEX_DIM_2D,
              "t: dim = 2d");
        CHECK(s && s->type && s->type->ref == p.types.t_f32,
              "t: sampled elem = f32");
        done(&p);
    }
    {
        /* Invalid sampled texture: texture_2d<bool>. */
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<bool>;\n");
        CHECK(!p.ok, "texture_2d<bool>: must error");
        CHECK(has_error_with(&p.diag, "sampled texture type must be f32, i32, or u32"),
              "texture_2d<bool> diagnostic");
        done(&p);
    }
    {
        /* ptr type specifier */
        P p; run(&p, "fn f(x: ptr<function, i32>) {}\n");
        CHECK(p.ok, "ptr<function, i32> parameter: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "ptr param must not trip 'has no type yet'");
        WGSLSymbol *s = sym_for(&p, "x");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_PTR,
              "x: typed as ptr");
        CHECK(s && s->type && (s->type->flags & 0xFF) == WGSL_AS_FUNCTION,
              "x: address space function");
        done(&p);
    }
    {
        /* Address-of and indirection */
        P p; run(&p, "fn f() { var i: i32; let p = &i; let v = *p; }\n");
        CHECK(p.ok, "address-of and indirection: tc_ok");
        WGSLSymbol *sp = sym_for(&p, "p");
        CHECK(sp && sp->type && sp->type->kind == WGSL_TYPE_PTR, "p is ptr");
        WGSLSymbol *sv = sym_for(&p, "v");
        CHECK(sv && sv->type && sv->type->kind == WGSL_TYPE_I32, "v is i32");
        done(&p);
    }
    {
        /* §11.2 — pointer args must match address space, store type,
         * and access mode, not just the outer `ptr` kind. */
        P p; run(&p,
            "var<private> g: i32;\n"
            "fn take_fn(p: ptr<function, i32>) {}\n"
            "fn f() { take_fn(&g); }\n");
        CHECK(!p.ok, "ptr arg AS mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_fn'"),
              "ptr AS mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn take_i32(p: ptr<function, i32>) {}\n"
            "fn f() { var u: u32; take_i32(&u); }\n");
        CHECK(!p.ok, "ptr arg store-type mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_i32'"),
              "ptr store-type mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<storage, read> s: i32;\n"
            "fn take_rw(p: ptr<storage, i32, read_write>) {}\n"
            "fn f() { take_rw(&s); }\n");
        CHECK(!p.ok, "ptr arg access-mode mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_rw'"),
              "ptr access-mode mismatch diag");
        done(&p);
    }
    {
        /* Assignment to non-reference */
        P p; run(&p, "fn f() { let x = 1; x = 2; }\n");
        CHECK(!p.ok, "assign to let: must error");
        CHECK(has_error_with(&p.diag, "left-hand side must be a reference"),
              "assign to value diag");
        done(&p);
    }
    {
        /* Assignment to read-only reference */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<uniform> u: S;\nfn f() { u.a = 1; }\n");
        CHECK(!p.ok, "assign to uniform: must error");
        CHECK(has_error_with(&p.diag, "cannot assign to read-only reference"),
              "assign to read-only diag");
        done(&p);
    }
    {
        /* Pointer in struct */
        P p; run(&p, "struct S { p: ptr<function, i32> }\n");
        CHECK(!p.ok, "ptr in struct: must error");
        CHECK(has_error_with(&p.diag, "struct member type cannot be a ptr"),
              "ptr in struct diag");
        done(&p);
    }
    {
        /* Override-sized arrays cannot be nested in structs. */
        P p; run(&p, "override N: u32;\nstruct S { a: array<u32, N> }\n");
        CHECK(!p.ok, "override-sized array in struct: must error");
        CHECK(has_error_with(&p.diag, "must not contain an override-sized array"),
              "override array struct-member diag");
        done(&p);
    }
    /* ── §7 Variable and Value Declarations ──────────────────────── */
    {
        /* §7 (general): var must have type or initializer */
        P p; run(&p, "fn f() { var x; }\n");
        CHECK(!p.ok, "var without type or init: must error");
        CHECK(has_error_with(&p.diag, "must specify a type or an initializer"),
              "var no type/init diag");
        done(&p);
    }
    {
        /* §7 (general): init type must be convertible to declared type */
        P p; run(&p, "fn f() { var x: i32 = 1.5; }\n");
        CHECK(!p.ok, "i32 = 1.5: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "i32 = float mismatch diag");
        done(&p);
    }
    {
        /* §7 (general): composite init compatibility is structural:
         * same `vec` kind alone is not enough. */
        P p; run(&p, "fn f() { let x: vec2<bool> = vec2<u32>(1u, 2u); }\n");
        CHECK(!p.ok, "vec2<bool> = vec2<u32>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "vec element mismatch diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x: vec2<i32> = vec3<i32>(1, 2, 3); }\n");
        CHECK(!p.ok, "vec2<i32> = vec3<i32>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "vec width mismatch diag");
        done(&p);
    }
    {
        /* §7.2.2: override must be concrete scalar */
        P p; run(&p, "override X: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);\n");
        CHECK(!p.ok, "override vec3: must error");
        CHECK(has_error_with(&p.diag, "override type must be a scalar"),
              "override vec3 diag");
        done(&p);
    }
    {
        /* §7.2.2: @id(N) must be in [0, 65535] */
        P p; run(&p, "@id(70000) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(70000): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id range diag");
        done(&p);
    }
    {
        /* §7.3: var<storage, write> forbidden */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<storage, write> s: S;\n");
        CHECK(!p.ok, "storage write: must error");
        CHECK(has_error_with(&p.diag, "do not support 'write' access mode"),
              "storage write diag");
        done(&p);
    }
    {
        /* §7.3: var template list accepts address space and access only. */
        P p; run(&p, "@group(0) @binding(0) var<storage, read, read> x: i32;\n");
        CHECK(!p.ok, "var with extra template arg: must error");
        CHECK(has_error_with(&p.diag, "at most address space and access mode"),
              "var template arity diag");
        done(&p);
    }
    {
        /* §7.3: access mode only for storage */
        P p; run(&p, "var<private, read> x: i32;\n");
        CHECK(!p.ok, "private read: must error");
        CHECK(has_error_with(&p.diag, "access mode can only be specified for 'storage'"),
              "private access mode diag");
        done(&p);
    }
    {
        /* §7.3: initializer disallowed for var<uniform> */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<uniform> u: S = S(1);\n");
        CHECK(!p.ok, "uniform with init: must error");
        CHECK(has_error_with(&p.diag, "cannot have an initializer"),
              "uniform init diag");
        done(&p);
    }
    {
        /* §7.3: handle address-space variables cannot have initializers. */
        P p; run(&p,
            "@group(0) @binding(0) var s0: sampler;\n"
            "@group(0) @binding(1) var s1: sampler = s0;\n");
        CHECK(!p.ok, "sampler with init: must error");
        CHECK(has_error_with(&p.diag, "cannot have an initializer"),
              "handle init diag");
        done(&p);
    }
    {
        /* §7.3: var<function> at module scope forbidden */
        P p; run(&p, "var<function> x: i32 = 0;\n");
        CHECK(!p.ok, "var<function> at module: must error");
        CHECK(has_error_with(&p.diag, "can only appear inside a function body"),
              "var<function> module scope diag");
        done(&p);
    }
    {
        /* §7.3: var<private> inside function forbidden */
        P p; run(&p, "fn f() { var<private> x: i32 = 0; }\n");
        CHECK(!p.ok, "var<private> in function: must error");
        CHECK(has_error_with(&p.diag, "must be at module scope"),
              "var<private> fn scope diag");
        done(&p);
    }
    {
        /* §7.3: texture var inside function forbidden */
        P p; run(&p, "fn f() { var t: texture_2d<f32>; }\n");
        CHECK(!p.ok, "texture in fn: must error");
        CHECK(has_error_with(&p.diag, "texture and sampler variables must be at module scope"),
              "texture fn scope diag");
        done(&p);
    }
    {
        /* Depth texture: no template args. */
        P p; run(&p,
            "@group(0) @binding(0) var d : texture_depth_2d;\n");
        CHECK(p.ok, "var d : texture_depth_2d: tc_ok");
        WGSLSymbol *s = sym_for(&p, "d");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE &&
              s->type->width == WGSL_TEX_DIM_DEPTH_2D,
              "d: depth_2d");
        done(&p);
    }
    {
        /* texture_external: no template args, predeclared shape. */
        P p; run(&p,
            "@group(0) @binding(0) var e : texture_external;\n");
        CHECK(p.ok, "var e : texture_external: tc_ok");
        WGSLSymbol *s = sym_for(&p, "e");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE &&
              s->type->width == WGSL_TEX_DIM_EXTERNAL,
              "e: external");
        done(&p);
    }
    {
        /* Storage texture: <texel_format, access>. */
        P p; run(&p,
            "@group(0) @binding(0) "
            "var st : texture_storage_2d<rgba8unorm, write>;\n");
        CHECK(p.ok, "var st : texture_storage_2d<rgba8unorm, write>: tc_ok");
        CHECK(!has_error_with(&p.diag, "undeclared identifier 'rgba8unorm'"),
              "rgba8unorm must be recognised");
        WGSLSymbol *s = sym_for(&p, "st");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE,
              "st: typed as texture");
        CHECK(s && s->type && s->type->width == WGSL_TEX_DIM_STORAGE_2D,
              "st: dim = storage_2d");
        CHECK(s && s->type && s->type->array_len == WGSL_TEX_FORMAT_RGBA8UNORM,
              "st: format = rgba8unorm");
        CHECK(s && s->type && s->type->rows == WGSL_ACCESS_WRITE,
              "st: access = write");
        done(&p);
    }
    {
        /* End-to-end: a textured fragment shader argument list must
         * not trip "has no type yet" diagnostics. */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "@fragment fn fs(@location(0) uv : vec2<f32>) "
            "-> @location(0) vec4<f32> {\n"
            "  return textureSampleLevel(t, s, uv, 0.0);\n"
            "}\n");
        CHECK(p.ok, "textured fragment: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "textured-fragment must not trip 'has no type yet'");
        CHECK(!has_error_with(&p.diag, "is not a type"),
              "textured-fragment must not trip 'is not a type'");
        done(&p);
    }
    {
        /* Texture / sampler types are interned: same shape → same ptr. */
        P p; run(&p,
            "@group(0) @binding(0) var a : texture_2d<f32>;\n"
            "@group(0) @binding(1) var b : texture_2d<f32>;\n");
        CHECK(p.ok, "two texture_2d<f32>: tc_ok");
        WGSLSymbol *A = sym_for(&p, "a");
        WGSLSymbol *B = sym_for(&p, "b");
        CHECK(A && B && A->type && A->type == B->type,
              "texture_2d<f32> interned");
        done(&p);
    }
    {
        /* Generated texture metadata now validates coordinate shapes. */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_1d<f32>;\n"
            "fn f() { _ = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureSample(texture_1d, vec2 coords): must error");
        CHECK(has_error_with(&p.diag, "coordinates"),
              "textureSample coordinate shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureSample(texture_2d, vec2 coords): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_f32,
              "textureSample(texture_2d<f32>) -> vec4<f32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<u32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureSample(texture_2d<u32>): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureSample'"),
              "integer textureSample overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<i32>;\n"
            "fn f() { let r = textureSampleLevel(t, s, vec2<f32>(0.0), 0.0); }\n");
        CHECK(!p.ok, "textureSampleLevel(texture_2d<i32>): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureSampleLevel'"),
              "integer textureSampleLevel overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r: f32 = textureSampleCompare(d, sc, vec2<f32>(0.0), 0.5); }\n");
        CHECK(p.ok, "textureSampleCompare(depth_2d): returns f32");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var st : texture_storage_2d<rgba8uint, write>;\n"
            "fn f() { textureStore(st, vec2<i32>(0), vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureStore rgba8uint with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "value argument"),
              "textureStore value shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(component, texture_2d): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_f32,
              "textureGather(texture_2d<f32>) -> vec4<f32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<i32>;\n"
            "fn f() { let r: vec4<i32> = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(texture_2d<i32>) -> vec4<i32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<u32>;\n"
            "fn f() { let r: vec4<u32> = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(texture_2d<u32>) -> vec4<u32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureGather(4, t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureGather component 4: must error");
        CHECK(has_error_with(&p.diag, "component value"),
              "textureGather component range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r = textureGather(d, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(depth texture): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r = textureGatherCompare(d, sc, vec2<f32>(0.0), 0.5); }\n");
        CHECK(p.ok, "textureGatherCompare(depth texture): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0), vec2<i32>(7, -8)); }\n");
        CHECK(p.ok, "textureSample const offset in range: ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0), vec2<i32>(8, 0)); }\n");
        CHECK(!p.ok, "textureSample offset out of range: must error");
        CHECK(has_error_with(&p.diag, "offset component"),
              "texture offset range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f(o : vec2<i32>) { let r = textureSample(t, s, vec2<f32>(0.0), o); }\n");
        CHECK(!p.ok, "textureSample non-const offset: must error");
        CHECK(has_error_with(&p.diag, "offset must be a const-expression"),
              "texture offset const diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_multisampled_2d<i32>;\n"
            "fn f() { let r = textureLoad(t, vec2<i32>(0), 0); }\n");
        CHECK(p.ok, "textureLoad(multisampled_2d<i32>): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_i32,
              "textureLoad(texture_multisampled_2d<i32>) -> vec4<i32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureLoad(t, vec2<i32>(0), 0.0); }\n");
        CHECK(!p.ok, "textureLoad float mip level: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad exact-row no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_cube<f32>;\n"
            "fn f() { let r = textureLoad(t, vec3<i32>(0), 0); }\n");
        CHECK(!p.ok, "textureLoad(texture_cube): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad cube no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_3d<f32>;\n"
            "fn f() { let r = textureDimensions(t, 0); }\n");
        CHECK(p.ok, "textureDimensions(texture_3d, level): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 3 && r->type->ref == p.types.t_u32,
              "textureDimensions(texture_3d) -> vec3<u32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r: vec2<bool> = textureDimensions(t, 0); }\n");
        CHECK(!p.ok, "textureDimensions vec2<u32> -> vec2<bool>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "textureDimensions declared vec mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureDimensions(t, 0.0); }\n");
        CHECK(!p.ok, "textureDimensions float level: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureDimensions'"),
              "textureDimensions exact-row no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var st : texture_storage_2d<rgba8uint, write>;\n"
            "fn f() { let r = textureLoad(st, vec2<i32>(0)); }\n");
        CHECK(!p.ok, "textureLoad(write-only storage texture): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad storage access exact-row diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { let r: f32 = subgroupShuffle(0, 0); }\n");
        CHECK(!p.ok, "subgroupShuffle abstract int result -> f32: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "subgroupShuffle concrete result mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { let r: i32 = subgroupShuffle(0, 0); }\n");
        CHECK(p.ok, "subgroupShuffle abstract int result materializes to i32");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { _ = subgroupShuffleXor(0, i32(0)); }\n");
        CHECK(!p.ok, "subgroupShuffleXor i32 mask: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'subgroupShuffleXor'"),
              "subgroupShuffleXor i32 mask no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const bad: u32 = 128u;\n"
            "fn f() { _ = subgroupShuffleUp(0, bad); }\n");
        CHECK(!p.ok, "subgroupShuffleUp const delta 128: must error");
        CHECK(has_error_with(&p.diag, "second argument must be in [0, 128)"),
              "subgroupShuffleUp range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "var<private> lane: u32;\n"
            "fn f() { _ = subgroupBroadcast(0, lane); }\n");
        CHECK(!p.ok, "subgroupBroadcast runtime id: must error");
        CHECK(has_error_with(&p.diag, "second argument must be a const-expression"),
              "subgroupBroadcast const id diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const lane = 4;\n"
            "fn f() { _ = quadBroadcast(0, lane); }\n");
        CHECK(!p.ok, "quadBroadcast lane 4: must error");
        CHECK(has_error_with(&p.diag, "second argument must be in [0, 4)"),
              "quadBroadcast range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const lane = 127u;\n"
            "fn f() { _ = subgroupBroadcast(0, lane); }\n");
        CHECK(p.ok, "subgroupBroadcast const lane 127: ok");
        done(&p);
    }

    /* ── §6.1.2 / §6.1.3 — Overload resolution ────────────────── *
     *
     * Pins the §6.2.1 spec example shape: `log2(32)` ranks the
     * AbstractFloat overload (rank 5) above the f32/f16 overloads
     * (ranks 6/7), so the call returns AbstractFloat — which the
     * surrounding let then concretizes to f32.
     */
    {
        P p; run(&p, "fn f() { let r = log2(32); }\n");
        CHECK(p.ok, "log2(32): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "log2(32) → AbstractFloat → f32 via let");
        done(&p);
    }
    {
        /* `log2(32f)` — only the f32 overload is feasible. */
        P p; run(&p, "fn f() { let r = log2(32f); }\n");
        CHECK(p.ok, "log2(32f): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32, "log2(32f) → f32");
        done(&p);
    }
    {
        /* `log2(1u)` — no float overload accepts u32 (i32→f32 is also
         * infeasible per §6.1.2).  Must error. */
        P p; run(&p, "fn f() { let r = log2(1u); }\n");
        CHECK(!p.ok, "log2(1u): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "log2(1u) no-overload diag");
        done(&p);
    }
    {
        /* §6.2.1 example 2: `1 + 2.5` style — applied to a builtin.
         * `min(1, 1.5)` ranks AbstractFloat (5, 0) above f32 (6, 1)
         * and f16 (7, 2).  AbstractFloat wins → let concretizes to f32. */
        P p; run(&p, "fn f() { let r = min(1, 1.5); }\n");
        CHECK(p.ok, "min(1, 1.5): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "min(1, 1.5) → AbstractFloat → f32");
        done(&p);
    }
    {
        /* `max(1, 2)` — both AbstractInt, lowest rank is AbstractInt
         * (0, 0).  Let then concretizes AbstractInt → i32. */
        P p; run(&p, "fn f() { let r = max(1, 2); }\n");
        CHECK(p.ok, "max(1, 2): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32,
              "max(1, 2) → AbstractInt → i32");
        done(&p);
    }
    {
        /* `min(1u, 1)` — only the u32 overload is feasible
         * (AbstractInt→u32 is rank 4, u32→u32 is rank 0; the
         * AbstractInt overload requires u32→AbstractInt which is
         * infeasible). */
        P p; run(&p, "fn f() { let r = min(1u, 1); }\n");
        CHECK(p.ok, "min(1u, 1): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_u32, "min(1u, 1) → u32");
        done(&p);
    }
    {
        /* Float-only family rejects integer concrete arg. */
        P p; run(&p, "fn f() { let r = cos(1u); }\n");
        CHECK(!p.ok, "cos(1u): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "cos(1u) no-overload diag");
        done(&p);
    }
    {
        /* Integer-only family rejects float arg. */
        P p; run(&p, "fn f() { let r = reverseBits(1.0); }\n");
        CHECK(!p.ok, "reverseBits(1.0): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "reverseBits(1.0) no-overload diag");
        done(&p);
    }
    {
        /* Numeric family on a vec carries the shape through. */
        P p; run(&p, "fn f() { let r = abs(vec3<f32>(1.0)); }\n");
        CHECK(p.ok, "abs(vec3<f32>): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 3 && r->type->ref == p.types.t_f32,
              "abs(vec3<f32>) → vec3<f32>");
        done(&p);
    }
    {
        /* Reduction builtins: BR_ARG0_ELEM combined with overload
         * resolution.  `length(vec3<f32>(1))` picks T=f32, then the
         * BR_ARG0_ELEM rule yields f32 (the element scalar). */
        P p; run(&p, "fn f() { let r = length(vec3<f32>(1.0)); }\n");
        CHECK(p.ok, "length(vec3<f32>): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "length(vec3<f32>) → f32 (scalar element)");
        done(&p);
    }
    {
        /* §17 exact rows: sign excludes unsigned integer T. */
        P p; run(&p, "fn f() { let r = sign(1u); }\n");
        CHECK(!p.ok, "sign(u32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "sign(u32) no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: normalize is vector-only, not scalar float. */
        P p; run(&p, "fn f() { let r = normalize(1.0); }\n");
        CHECK(!p.ok, "normalize(f32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "normalize scalar no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: determinant only accepts square matrices. */
        P p; run(&p, "fn f() { let r = determinant(mat2x3<f32>()); }\n");
        CHECK(!p.ok, "determinant(mat2x3): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "determinant rectangular no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: transpose flips C/R in the return shape. */
        P p; run(&p, "fn f() { let r = transpose(mat2x3<f32>()); }\n");
        CHECK(p.ok, "transpose(mat2x3): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_MAT &&
              r->type->width == 3 && r->type->rows == 2 &&
              r->type->ref == p.types.t_f32,
              "transpose(mat2x3<f32>) → mat3x2<f32>");
        done(&p);
    }
    {
        /* §17 exact rows: dot4I8Packed returns i32, not u32. */
        P p; run(&p,
            "fn f() { let r: i32 = dot4I8Packed(1u, 2u); }\n");
        CHECK(p.ok, "dot4I8Packed return i32: tc_ok");
        done(&p);
    }
    {
        /* §17.6 exact rows: derivative builtins accept f32, not f16. */
        P p; run(&p,
            "enable f16;\n"
            "fn f() { let r = dpdx(1.0h); }\n");
        CHECK(!p.ok, "dpdx(f16): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "dpdx(f16) no-overload diag");
        done(&p);
    }
    {
        /* Step-2 smoke: a non-const concrete arg forces concretization.
         * `min(1, v)` with `var v: f32 = 0.0;` resolves to T=f32 (the
         * AbstractFloat candidate is eliminated by step 1, since
         * f32→AbstractFloat is infeasible). */
        P p; run(&p,
            "fn f() {\n"
            "  var v: f32 = 0.0;\n"
            "  let r = min(1, v);\n"
            "}\n");
        CHECK(p.ok, "min(1, v) non-const: tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "min(1, var f32) → f32");
        done(&p);
    }
    {
        /* Const-expression ident participates as const-expr — picks the
         * AbstractInt overload just like a literal would. */
        P p; run(&p,
            "const C: i32 = 1;\n"
            "fn f() { let r = max(C, 2); }\n");
        CHECK(p.ok, "max(const C, 2): tc_ok");
        /* C is a CONST symbol with i32 type, so arg0 has rank table
         * (i32→i32)=0, (AbstractInt→i32)=3; the AbstractInt overload
         * is eliminated by step 1 since i32→AbstractInt is infeasible.
         * Result: i32. */
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32, "max(const i32, 2) → i32");
        done(&p);
    }
    {
        /* `clamp(0, lo, hi)` over abstract literals — ternary picks
         * AbstractInt with rank vector (0, 0, 0). */
        P p; run(&p, "fn f() { let r = clamp(0, 1, 2); }\n");
        CHECK(p.ok, "clamp(0, 1, 2): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32,
              "clamp(0,1,2) → AbstractInt → i32");
        done(&p);
    }
    {
        /* Mixing concrete float and concrete int with no automatic
         * conversion between them — must error. */
        P p; run(&p, "fn f() { let r = min(1.0f, 1u); }\n");
        CHECK(!p.ok, "min(f32, u32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "min(f32, u32) no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = mix(vec2<f32>(0.0), vec2<f32>(1.0), 0.5); }\n");
        CHECK(p.ok, "mix(vec2, vec2, scalar): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 2 && r->type->ref == p.types.t_f32,
              "mix scalar factor preserves vector result");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = mix(vec2<f32>(0.0), vec2<f32>(1.0), vec3<f32>(0.5)); }\n");
        CHECK(!p.ok, "mix(vec2, vec2, vec3 factor): must error");
        CHECK(has_error_with(&p.diag, "mix factor"),
              "mix factor shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = ldexp(vec2<f32>(1.0), vec2<i32>(1)); }\n");
        CHECK(p.ok, "ldexp(vec2<f32>, vec2<i32>): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = ldexp(vec2<f32>(1.0), 1); }\n");
        CHECK(!p.ok, "ldexp(vec2<f32>, scalar exponent): must error");
        CHECK(has_error_with(&p.diag, "exponent argument"),
              "ldexp exponent shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { var v: f32; let r = ldexp(v, i32(129)); }\n");
        CHECK(!p.ok, "ldexp(f32, 129): must error");
        CHECK(has_error_with(&p.diag, "exponent too large"),
              "ldexp f32 exponent domain diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable f16;\n"
            "fn f() { var v: vec2<f16>; let r = ldexp(v, vec2<i32>(17)); }\n");
        CHECK(!p.ok, "ldexp(vec2<f16>, 17): must error");
        CHECK(has_error_with(&p.diag, "exponent too large"),
              "ldexp f16 exponent domain diag");
        done(&p);
    }
    {
        P p; run(&p, "var<private> X = min(1, 2);\n");
        CHECK(p.ok, "@const builtin call is accepted in private var init");
        done(&p);
    }

    /* ── §6.2.1 — Abstract → concrete overflow at narrowing ──── *
     *
     * The §6.1.2 conversion table says "AbstractInt → i32 is identity
     * if the value is in i32, error otherwise."  Same for u32.  These
     * tests exercise that range check for `let` / `var` / `override`
     * (the `const` path already runs through the consteval module pass
     * and was tested separately in test_consteval_module).
     */
    {
        P p; run(&p, "fn f() { let x: i32 = 2147483648; }\n");
        CHECK(!p.ok, "let x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x: i32 = 2147483648; }\n");
        CHECK(!p.ok, "var x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "var i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "var<private> x: i32 = 2147483648;\n");
        CHECK(!p.ok, "var<private> x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "var<private> i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "override X: i32 = 2147483648;\n");
        CHECK(!p.ok, "override X: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "override i32 overflow diag");
        done(&p);
    }
    {
        /* Negative AbstractInt → u32 must error. */
        P p; run(&p, "fn f() { let x: u32 = -1; }\n");
        CHECK(!p.ok, "let x: u32 = -1: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let u32 negative diag");
        done(&p);
    }
    {
        /* AbstractInt > UINT32_MAX → u32 must error. */
        P p; run(&p, "fn f() { let x: u32 = 4294967296; }\n");
        CHECK(!p.ok, "let x: u32 = 2^32: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let u32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 2147483648i; }\n");
        CHECK(!p.ok, "suffixed i32 literal 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range for i32"),
              "suffixed i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 4294967296u; }\n");
        CHECK(!p.ok, "suffixed u32 literal 2^32: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "suffixed u32 overflow diag");
        done(&p);
    }
    {
        /* Computed const-expression that overflows materializes
         * via the binary op then the i32 conversion — must error. */
        P p; run(&p, "fn f() { let x: i32 = 1073741824 * 2; }\n");
        CHECK(!p.ok, "let x: i32 = 2^30 * 2: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let i32 computed overflow diag");
        done(&p);
    }
    {
        /* Boundary values that DO fit must remain accepted. */
        P p; run(&p,
            "fn f() {\n"
            "  let a: i32 =  2147483647;\n"
            "  let b: i32 = -2147483648;\n"
            "  let c: u32 =  4294967295;\n"
            "}\n");
        CHECK(p.ok, "i32/u32 boundary values: tc_ok");
        done(&p);
    }
    {
        /* Runtime initializers (non-const) MUST NOT trigger spurious
         * range diagnostics: the helper rolls back consteval's
         * "not a constant" complaints. */
        P p; run(&p, "fn f(p: i32) { let x: i32 = p + 1; }\n");
        CHECK(p.ok, "runtime fn-param init: tc_ok");
        CHECK(!has_error_with(&p.diag, "out of range"),
              "runtime init: no spurious range diag");
        CHECK(!has_error_with(&p.diag, "not a constant"),
              "runtime init: no leaked 'not a constant' diag");
        done(&p);
    }

    /* ── §6.2.4 — Per-suffix float literal range gates ──────── *
     *
     * `f` / `h` suffixes pin the literal to a concrete float type;
     * a value that doesn't fit the binding's IEEE range is a shader-
     * generation error.  strtod's ERANGE branch only catches double-
     * precision overflow (~1e308); the per-suffix check closes the
     * f32 (FLT_MAX, ~3.4e38) and f16 (65504) gates.
     */
    {
        P p; run(&p, "enable f16;\nfn f() { let x = 65504.0h; }\n");
        CHECK(p.ok, "1.0h at f16 max: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "enable f16;\nfn f() { let x = 65505.0h; }\n");
        CHECK(!p.ok, "65505.0h: must error");
        CHECK(has_error_with(&p.diag, "out of range for f16"),
              "h-suffix overflow diag");
        done(&p);
    }
    {
        /* `const` path must NOT double-emit (module pass + type
         * checker both visit the literal; CONST_EVALED flag dedup). */
        P p; run(&p, "enable f16;\nconst x = 1e10h;\n");
        CHECK(!p.ok, "const 1e10h: must error");
        int cnt = 0;
        for (size_t k = 0; k < p.diag.count; k++) {
            if (strstr(p.diag.items[k].message, "out of range for f16")) cnt++;
        }
        CHECK(cnt == 1, "h-suffix overflow: single emit (no double-fire)");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 3.4e38f; }\n");
        CHECK(p.ok, "3.4e38f: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1e40f; }\n");
        CHECK(!p.ok, "1e40f: must error");
        CHECK(has_error_with(&p.diag, "out of range for f32"),
              "f-suffix overflow diag");
        done(&p);
    }

    /* ── §6.2.8 — Atomic-typed expression is illegal ─────────── *
     *
     * "An expression must not evaluate to an atomic type."  Reading
     * an atomic var as a value (`let x = a;`) is the canonical
     * non-builtin path; atomicLoad(&a) is the only legal way.
     */
    {
        P p; run(&p,
            "var<workgroup> a: atomic<i32>;\n"
            "fn f() { let x = a; }\n");
        CHECK(!p.ok, "let x = atomic_var: must error");
        CHECK(has_error_with(&p.diag, "atomic type"),
              "atomic-in-expr diag");
        done(&p);
    }
    {
        /* The atomicLoad path remains legal — pointer arg via `&`. */
        P p; run(&p,
            "var<workgroup> a: atomic<i32>;\n"
            "fn f() { let x = atomicLoad(&a); }\n");
        CHECK(p.ok, "atomicLoad(&a): tc_ok");
        done(&p);
    }

    /* ── §6.2.9 — Override-sized array element count ────────── *
     *
     * `array<E, N>` where N is an override declaration is legal
     * (per §6.2.9 + §14.4.5 AS placement).  The length isn't
     * const-evaluable, so the type checker treats the array as
     * runtime-sized for v1 layout purposes; the spec's identifier-
     * equality refinement for two distinct overrides is a known v1.x
     * gap (see missing.md).
     */
    {
        P p; run(&p,
            "override N: u32;\n"
            "var<workgroup> wg: array<f32, N>;\n");
        CHECK(p.ok, "var<workgroup> array<f32, override N>: tc_ok");
        CHECK(!has_error_with(&p.diag, "not a constant"),
              "override-sized array: no 'not a constant' diag");
        CHECK(!has_error_with(&p.diag, "greater than zero"),
              "override-sized array: no spurious 'must be > 0' diag");
        done(&p);
    }
    {
        /* A let/var with a non-const, non-override length must still
         * error — only override symbols get the size-source pass. */
        P p; run(&p, "var<private> v: u32 = 4u;\n"
                     "var<private> a: array<f32, v>;\n");
        CHECK(!p.ok, "array sized by `var`: must error");
        CHECK(has_error_with(&p.diag, "greater than zero") ||
              has_error_with(&p.diag, "not a constant"),
              "var-sized array rejection diag");
        done(&p);
    }

    /* ── §6.2.11 / §6.2.12 / §6.2.13 — Composite predicates ──── *
     *
     * The new predicates feed the let / param / return gates so a
     * non-constructible type in those positions produces a clear,
     * single-source diagnostic instead of cascading "cannot convert"
     * complaints.
     */
    {
        /* Function return type must be constructible. */
        P p; run(&p, "fn f() -> array<f32> { return; }\n");
        CHECK(!p.ok, "fn f() -> array<f32>: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "fn return non-constructible diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() -> texture_2d<f32> { return; }\n");
        CHECK(!p.ok, "fn f() -> texture_2d<f32>: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "fn return texture diag");
        done(&p);
    }
    {
        /* Function param type must be constructible / ptr / tex / sampler. */
        P p; run(&p, "fn f(a: array<f32>) {}\n");
        CHECK(!p.ok, "fn(a: array<f32>): must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "fn param runtime-array diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(a: atomic<i32>) {}\n");
        CHECK(!p.ok, "fn(a: atomic<i32>): must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "fn param atomic diag");
        done(&p);
    }
    {
        /* Param positives — ptr / texture / sampler still legal. */
        P p; run(&p,
            "fn f(p: ptr<function, i32>, t: texture_2d<f32>, s: sampler) {}\n");
        CHECK(p.ok, "ptr / texture / sampler params: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<handle, u32>) {}\n");
        CHECK(!p.ok, "ptr<handle, u32> param: must error");
        CHECK(has_error_with(&p.diag, "ptr address space"),
              "ptr handle AS diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<private, u32, read>) {}\n");
        CHECK(!p.ok, "ptr<private, u32, read> param: must error");
        CHECK(has_error_with(&p.diag, "ptr access mode"),
              "ptr private access-mode diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<storage, u32, write>;\n");
        CHECK(!p.ok, "ptr<storage, u32, write>: must error");
        CHECK(has_error_with(&p.diag, "read' or 'read_write"),
              "ptr storage write access-mode diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<private, 1>;\n");
        CHECK(!p.ok, "ptr literal store type: must error");
        CHECK(has_error_with(&p.diag, "ptr store type must be a type"),
              "ptr literal store type diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<private, atomic<u32>>;\n");
        CHECK(!p.ok, "ptr<private, atomic<u32>>: must error");
        CHECK(has_error_with(&p.diag, "atomic requires the workgroup or storage"),
              "ptr atomic AS pairing diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<workgroup, array<i32>>;\n");
        CHECK(!p.ok, "ptr<workgroup, runtime array>: must error");
        CHECK(has_error_with(&p.diag, "runtime-sized array requires"),
              "ptr runtime array AS pairing diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<private, clamp>) {}\n");
        CHECK(!p.ok, "ptr<private, clamp> param: must error");
        CHECK(has_error_with(&p.diag, "not a type"),
              "ptr builtin-as-store-type diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<function, texture_external>) {}\n");
        CHECK(!p.ok, "ptr<function, texture_external> param: must error");
        CHECK(has_error_with(&p.diag, "ptr store type must be storable"),
              "ptr texture store-type diag");
        done(&p);
    }
    {
        /* Fixed-size array param: constructible, legal. */
        P p; run(&p, "fn f(a: array<f32, 4>) {}\n");
        CHECK(p.ok, "array<f32, 4> param: tc_ok");
        done(&p);
    }
    {
        /* Struct deep-walk: a struct containing a runtime-sized array
         * is not constructible, so it cannot be a return type. */
        P p; run(&p,
            "struct S { a: array<f32> }\n"
            "fn f() -> S { return; }\n");
        CHECK(!p.ok, "struct with runtime array as return: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "struct-runtime-array return diag");
        done(&p);
    }
    {
        /* Struct deep-walk: a struct containing an atomic is not
         * constructible, so it cannot be a function param. */
        P p; run(&p,
            "struct S { a: atomic<i32> }\n"
            "fn f(s: S) {}\n");
        CHECK(!p.ok, "struct with atomic as param: must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "struct-with-atomic param diag");
        done(&p);
    }
    {
        /* Nested struct of constructibles — legal. */
        P p; run(&p,
            "struct Inner { x: f32 }\n"
            "struct Outer { a: Inner, b: vec3<f32> }\n"
            "fn f(o: Outer) -> Outer { return o; }\n");
        CHECK(p.ok, "nested constructible struct param/return: tc_ok");
        done(&p);
    }
    {
        /* `let x: array<f32>` (runtime-sized): non-constructible. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> b: array<f32>;\n"
            "fn f() { let x: array<f32> = b; }\n");
        CHECK(!p.ok, "let x: array<f32>: must error");
        CHECK(has_error_with(&p.diag,
                  "'let' type must be a concrete constructible"),
              "let runtime-array diag");
        done(&p);
    }

    /* ── §6.3 — Enumerants are template parameters only ─────── *
     *
     * Address-space, access-mode, and texel-format enumerants live in
     * the predeclared scope so name resolution finds them — but using
     * one as a value (let init / arithmetic / etc.) is forbidden per
     * spec §6.3.  Previously this surfaced as "has no type yet"; now
     * a labelled diagnostic names the enumeration.
     */
    {
        P p; run(&p, "fn f() { let x = read_write; }\n");
        CHECK(!p.ok, "let x = read_write: must error");
        CHECK(has_error_with(&p.diag,
                  "access mode enumerant 'read_write' can only be used "
                  "as a template parameter"),
              "access-mode enumerant value diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = storage; }\n");
        CHECK(!p.ok, "let x = storage: must error");
        CHECK(has_error_with(&p.diag,
                  "address space enumerant 'storage'"),
              "address-space enumerant value diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = rgba8unorm; }\n");
        CHECK(!p.ok, "let x = rgba8unorm: must error");
        CHECK(has_error_with(&p.diag,
                  "texel format enumerant 'rgba8unorm'"),
              "texel-format enumerant value diag");
        done(&p);
    }
    {
        /* Arithmetic on enumerant — still flagged on the leaf. */
        P p; run(&p, "fn f() { let x = read + 1; }\n");
        CHECK(!p.ok, "read + 1: must error");
        CHECK(has_error_with(&p.diag, "access mode enumerant 'read'"),
              "enumerant-in-expr diag");
        done(&p);
    }
    {
        /* Template-arg usage stays legal. */
        P p; run(&p,
            "@group(0) @binding(0) "
            "var t: texture_storage_2d<rgba8unorm, write>;\n");
        CHECK(p.ok, "texture_storage_2d<rgba8unorm, write>: tc_ok");
        CHECK(!has_error_with(&p.diag, "enumerant"),
              "template-arg enumerant: no enumerant-value diag");
        done(&p);
    }

    /* ── §7 — declaration / decl-attr conformance round-up ────── */
    {
        /* §7.2.2 @id range — now wired through consteval so the
         * literal's span is read correctly (the previous version
         * pulled offset/length out of LITERAL_INT's payload[0],
         * which actually carries suffix bits). */
        P p; run(&p, "@id(70000) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(70000): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id range diag");
        done(&p);
    }
    {
        /* §7.2.2 @id boundary — 65535 must be accepted. */
        P p; run(&p, "@id(65535) override X: i32 = 1;\n");
        CHECK(p.ok, "@id(65535): tc_ok");
        done(&p);
    }
    {
        /* §7.2.2 @id uniqueness across overrides. */
        P p; run(&p,
            "@id(1) override A: i32 = 1;\n"
            "@id(1) override B: i32 = 2;\n");
        CHECK(!p.ok, "duplicate @id(1): must error");
        CHECK(has_error_with(&p.diag, "must be unique across overrides"),
              "@id uniqueness diag");
        done(&p);
    }
    {
        /* §12.8 @id must be non-negative; previously a bare `@id(-1)`
         * silently slipped past the typecker. */
        P p; run(&p, "@id(-1) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(-1): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id negative diag");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing another override is OK. */
        P p; run(&p,
            "override A: i32 = 1;\n"
            "override B: i32 = A * 2 + 5;\n");
        CHECK(p.ok, "override init referencing override: ok");
        done(&p);
    }
    {
        P p; run(&p, "override X;\n");
        CHECK(!p.ok, "override without type or initializer: must error");
        CHECK(has_error_with(&p.diag, "must specify a type or an initializer"),
              "override missing type/init diag");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing const is OK. */
        P p; run(&p,
            "const K: i32 = 7;\n"
            "override B: i32 = K + 1;\n");
        CHECK(p.ok, "override init referencing const: ok");
        done(&p);
    }
    {
        /* §17.9 — pack4x8snorm requires vec4<f32>. */
        P p; run(&p,
            "fn f() { let x = pack4x8snorm(vec4<i32>(0)); }\n");
        CHECK(!p.ok, "pack4x8snorm with vec4<i32>: must error");
        CHECK(has_error_with(&p.diag, "'pack4x8snorm' requires a vec4<f32> argument"),
              "pack4x8snorm shape diag");
        done(&p);
    }
    {
        /* §17.9 — pack4xI8 requires vec4<i32>. */
        P p; run(&p,
            "fn f() { let x = pack4xI8(vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "pack4xI8 with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "'pack4xI8' requires a vec4<i32> argument"),
              "pack4xI8 shape diag");
        done(&p);
    }
    {
        /* §17.9 — pack2x16float requires vec2<f32>. */
        P p; run(&p,
            "fn f() { let x = pack2x16float(vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "pack2x16float with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "'pack2x16float' requires a vec2<f32> argument"),
              "pack2x16float shape diag");
        done(&p);
    }
    {
        /* §17.10 — unpack4xU8 requires u32. */
        P p; run(&p,
            "fn f() { let x = unpack4xU8(1.5); }\n");
        CHECK(!p.ok, "unpack4xU8 with f32: must error");
        CHECK(has_error_with(&p.diag, "'unpack4xU8' requires a u32 argument"),
              "unpack4xU8 shape diag");
        done(&p);
    }
    {
        /* §17.9/10 happy paths. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = pack4x8snorm(vec4<f32>(0.5));\n"
            "  let b = pack4xI8(vec4<i32>(1));\n"
            "  let c = pack4xU8(vec4<u32>(2u));\n"
            "  let d = pack2x16float(vec2<f32>(0.25));\n"
            "  let e = unpack4x8snorm(1u);\n"
            "  let g = unpack4xI8(2u);\n"
            "  let h = unpack2x16float(3u);\n"
            "}\n");
        CHECK(p.ok, "pack/unpack happy paths: ok");
        done(&p);
    }
    {
        /* §17.10 — return types are correct (regression: unpack4xI8
         * was BR_VEC4F before, now BR_VEC4I). */
        P p; run(&p,
            "fn f() {\n"
            "  let v: vec4<i32> = unpack4xI8(1u);\n"
            "  let u: vec4<u32> = unpack4xU8(2u);\n"
            "  let w: vec2<f32> = unpack2x16snorm(3u);\n"
            "}\n");
        CHECK(p.ok, "unpack return-type fixes: ok");
        done(&p);
    }
    {
        /* §17.8 — atomic ptr arg validation: passing a non-atomic var
         * is rejected. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: i32;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let r = atomicAdd(&a, 1);\n"
            "}\n");
        CHECK(!p.ok, "atomicAdd on non-atomic: must error");
        CHECK(has_error_with(&p.diag, "requires a pointer to atomic<T>"),
              "atomicAdd ptr-arg diag");
        done(&p);
    }
    {
        /* §17.8 — value-arg type must match atomic's element T. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  atomicStore(&a, 1.5);\n"
            "}\n");
        CHECK(!p.ok, "atomicStore u32 with f32: must error");
        CHECK(has_error_with(&p.diag, "value argument has type"),
              "atomicStore type diag");
        done(&p);
    }
    {
        /* §17.8 — atomicLoad arg-count: 1 arg expected. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a, 1u);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad with extra arg: must error");
        CHECK(has_error_with(&p.diag, "atomicLoad' expects 1 argument"),
              "atomicLoad arity diag");
        done(&p);
    }
    {
        /* §17.8 — atomic builtins require read_write atomic pointers. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad on storage/read atomic: must error");
        CHECK(has_error_with(&p.diag, "ptr<storage, atomic<T>, read_write>"),
              "atomic storage-read access diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic pointer store types are only
         * instantiable in workgroup or storage address spaces. */
        P p; run(&p,
            "fn f(p: ptr<private, atomic<i32>>) {\n"
            "  let x = atomicLoad(p);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad on ptr<private, atomic<i32>>: must error");
        CHECK(has_error_with(&p.diag, "atomic requires the workgroup or storage"),
              "atomic private pointer diag");
        done(&p);
    }
    {
        /* §6.2.9 — fixed-size arrays of different lengths are
         * structurally distinct (regression — was previously trusted
         * via the kind-equal fallthrough in `init_type_mismatch`). */
        P p; run(&p,
            "var<private> x: array<f32, 4>;\n"
            "var<private> y: array<f32, 8>;\n"
            "fn f() { x = y; }\n");
        CHECK(!p.ok, "array<f32,4> = array<f32,8>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted to the assignment target"),
              "fixed-size array mismatch diag");
        done(&p);
    }
    {
        /* §6.2.13 — override-sized arrays are only valid as workgroup
         * memory views, not private/function/storage store types. */
        P p; run(&p,
            "override N: u32 = 4;\n"
            "var<private> x: array<f32, N>;\n");
        CHECK(!p.ok, "var<private> array<f32,N>: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array placement diag");
        done(&p);
    }
    {
        /* The same placement rule applies through aliases. */
        P p; run(&p,
            "override N: u32 = 4;\n"
            "alias T = array<f32, N>;\n"
            "fn f() { var x: T; }\n");
        CHECK(!p.ok, "function var alias to override array: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array alias placement diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override N: u32 = 4;\n"
            "fn f(p: ptr<private, array<u32, N>>) { }\n");
        CHECK(!p.ok, "ptr<private, override array>: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array ptr placement diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override N: u32 = 4;\n"
            "fn f() { let x = array<u32, N>(); }\n");
        CHECK(!p.ok, "construct override-sized array: must error");
        CHECK(has_error_with(&p.diag, "cannot be constructed"),
              "override array constructor diag");
        done(&p);
    }
    {
        /* §6.2.9 — override-sized arrays are shader-valid even when
         * the default value is not a valid fixed array count; pipeline
         * specialization can supply the generation-time length. */
        P p; run(&p,
            "override N: i32 = 0;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "array<f32, override=0>: shader-valid");
        done(&p);
    }
    {
        /* §6.2.9 — override default > 0: ok. */
        P p; run(&p,
            "override N: i32 = 4;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "override default > 0: ok");
        done(&p);
    }
    {
        /* §6.2.9 — no default → check skipped (pipeline-time value). */
        P p; run(&p,
            "override N: u32;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "override no default: ok (skip static check)");
        done(&p);
    }
    {
        /* §6.2.10 — duplicate struct member name. */
        P p; run(&p,
            "struct S { x: i32, x: f32 }\n");
        CHECK(!p.ok, "dup struct member name: must error");
        CHECK(has_error_with(&p.diag,
            "struct member name 'x' must be unique within the struct"),
              "dup-member-name diag");
        done(&p);
    }
    {
        /* §17.8 — happy paths covering Load / Store / RMW / CX. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<i32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a);\n"
            "  atomicStore(&a, 1);\n"
            "  let y = atomicAdd(&a, 1);\n"
            "  let z = atomicExchange(&a, 5);\n"
            "  let w = atomicCompareExchangeWeak(&a, 0, 1);\n"
            "}\n");
        CHECK(p.ok, "atomic happy paths: ok");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing a `let`/`var` is rejected. */
        P p; run(&p,
            "var<private> v: i32 = 3;\n"
            "override B: i32 = v + 1;\n");
        CHECK(!p.ok, "override init referencing var: must error");
        CHECK(has_error_with(&p.diag, "override initializer must be an override-expression"),
              "override-expr diag");
        done(&p);
    }
    {
        /* §7.2.1 — `const` type must be constructible.  Use a struct
         * containing a runtime-sized array (deep non-constructible) to
         * avoid the pre-existing atomic-init crash path. */
        P p; run(&p,
            "struct S { a: array<f32> }\n"
            "const c: S;\n");
        CHECK(!p.ok, "const non-constructible: must error");
        CHECK(has_error_with(&p.diag, "'const' type must be constructible"),
              "const constructible diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic in non-shared AS is rejected. */
        P p; run(&p, "var<private> a: atomic<i32>;\n");
        CHECK(!p.ok, "var<private> atomic: must error");
        CHECK(has_error_with(&p.diag,
                  "atomic store type requires the workgroup or storage"),
              "atomic-AS pairing diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic nested in an array has the same AS
         * restriction. */
        P p; run(&p, "var<private> a: array<atomic<i32>, 4>;\n");
        CHECK(!p.ok, "var<private> array<atomic<i32>>: must error");
        CHECK(has_error_with(&p.diag,
                  "atomic store type requires the workgroup or storage"),
              "array-atomic-AS pairing diag");
        done(&p);
    }
    {
        /* Atomic in workgroup AS is legal. */
        P p; run(&p, "var<workgroup> a: atomic<i32>;\n");
        CHECK(p.ok, "var<workgroup> atomic: tc_ok");
        done(&p);
    }
    {
        /* Storage atomics require read_write access. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: atomic<u32>;\n");
        CHECK(!p.ok, "var<storage> atomic default access: must error");
        CHECK(has_error_with(&p.diag, "requires read_write access mode"),
              "storage atomic access-mode diag");
        done(&p);
    }
    {
        /* Runtime-sized arrays are only valid in storage store types. */
        P p; run(&p, "var<workgroup> a: array<i32>;\n");
        CHECK(!p.ok, "var<workgroup> runtime array: must error");
        CHECK(has_error_with(&p.diag,
                  "runtime-sized array store type requires the storage"),
              "runtime-array-AS diag");
        done(&p);
    }
    {
        /* The runtime-sized-array restriction is deep through structs. */
        P p; run(&p,
            "struct S { a: array<i32> }\n"
            "var<private> s: S;\n");
        CHECK(!p.ok, "var<private> struct with runtime array: must error");
        CHECK(has_error_with(&p.diag,
                  "runtime-sized array store type requires the storage"),
              "struct-runtime-array-AS diag");
        done(&p);
    }
    {
        /* §12.2 / §12.7 — @group / @binding on a private var is
         * rejected (these attributes are for resource variables only). */
        P p; run(&p, "@group(0) var<private> x: i32;\n");
        CHECK(!p.ok, "@group on var<private>: must error");
        CHECK(has_error_with(&p.diag,
                  "@group is only allowed on resource variables"),
              "@group-on-non-resource diag");
        done(&p);
    }
    {
        /* @group / @binding on a workgroup var likewise rejected. */
        P p; run(&p, "@binding(0) var<workgroup> wg: i32;\n");
        CHECK(!p.ok, "@binding on var<workgroup>: must error");
        CHECK(has_error_with(&p.diag,
                  "@binding is only allowed on resource variables"),
              "@binding-on-non-resource diag");
        done(&p);
    }
    {
        /* @group / @binding on a uniform / storage / texture var stays
         * legal (the long-standing happy path). */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "struct S { x: f32 }\n"
            "@group(0) @binding(1) var<uniform> u: S;\n"
            "@group(0) @binding(2) var<storage> s: array<f32>;\n");
        CHECK(p.ok, "@group/@binding on resources: tc_ok");
        done(&p);
    }

    /* ── §8 — expression-level gates ─────────────────────────── */
    {
        /* §8.5.1 — mixed swizzle letterings rejected. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v.xrgb; }\n");
        CHECK(!p.ok, "v.xrgb: must error");
        CHECK(has_error_with(&p.diag, "mixed swizzle"),
              "mixed-swizzle diag");
        done(&p);
    }
    {
        /* §8.5.1 — vec const-index out of range. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[5]; }\n");
        CHECK(!p.ok, "v[5] on vec3: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "vec out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.1 — vector index must be an integer scalar. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[1.0]; }\n");
        CHECK(!p.ok, "v[1.0] on vec3: must error");
        CHECK(has_error_with(&p.diag, "expected integer scalar"),
              "vec float-index diag");
        done(&p);
    }
    {
        /* §8.5.1 — bool is not a valid vector index type. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[true]; }\n");
        CHECK(!p.ok, "v[true] on vec3: must error");
        CHECK(has_error_with(&p.diag, "expected integer scalar"),
              "vec bool-index diag");
        done(&p);
    }
    {
        /* §8.5.2 — mat const-index out of range. */
        P p; run(&p, "fn f() { var m: mat3x2<f32>; let x = m[5]; }\n");
        CHECK(!p.ok, "m[5] on mat3x2: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "mat out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.3 — array const-index out of range. */
        P p; run(&p, "fn f() { var a: array<i32,3>; let x = a[5]; }\n");
        CHECK(!p.ok, "a[5] on array<i32,3>: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "array out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.3 — negative const-index on array. */
        P p; run(&p, "fn f() { var a: array<i32,3>; let x = a[-1]; }\n");
        CHECK(!p.ok, "a[-1]: must error");
        CHECK(has_error_with(&p.diag, "negative"),
              "negative-index diag");
        done(&p);
    }
    {
        /* Runtime-sized arrays still reject negative const indices. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: array<i32>;\n"
            "fn f() { let x = a[-1]; }\n");
        CHECK(!p.ok, "runtime array a[-1]: must error");
        CHECK(has_error_with(&p.diag, "negative"),
              "runtime array negative-index diag");
        done(&p);
    }
    {
        /* Runtime index must NOT trip the static range check. */
        P p; run(&p,
            "fn f(i: u32) { var a: array<i32,3>; let x = a[i]; }\n");
        CHECK(p.ok, "runtime-index a[i]: tc_ok");
        CHECK(!has_error_with(&p.diag, "out of range"),
              "runtime index: no static range diag");
        done(&p);
    }
    {
        /* §8.6 — vec<bool> component-wise `&` / `|` logical ops allowed. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let c = a & b;\n"
            "  let d = a | b;\n"
            "}\n");
        CHECK(p.ok, "vec<bool> &|: tc_ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let e = a ^ b;\n"
            "}\n");
        CHECK(!p.ok, "vec<bool> ^ : must error");
        done(&p);
    }
    {
        /* §8.6 — integer bitwise ops do not scalar-broadcast to vectors. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<i32>(0, 0);\n"
            "  let b = 0;\n"
            "  let c = a & b;\n"
            "}\n");
        CHECK(!p.ok, "vec2<i32> & scalar: must error");
        CHECK(has_error_with(&p.diag, "matching shapes"),
              "bitwise scalar-vector diag");
        done(&p);
    }
    {
        /* §8.6 — shifts still integer-only. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let c = a << b;\n"
            "}\n");
        CHECK(!p.ok, "vec<bool> << : must error");
        CHECK(has_error_with(&p.diag, "integer"),
              "vec<bool> shift diag");
        done(&p);
    }
    {
        /* §8.7 — div by zero at const-eval. */
        P p; run(&p, "fn f() { let x: i32 = 5 / 0; }\n");
        CHECK(!p.ok, "5/0 const-eval: must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "div-by-zero diag");
        done(&p);
    }
    {
        /* §8.7 — mod by zero at const-eval. */
        P p; run(&p, "fn f() { let x: i32 = 5 % 0; }\n");
        CHECK(!p.ok, "5%0 const-eval: must error");
        CHECK(has_error_with(&p.diag, "modulo by zero"),
              "mod-by-zero diag");
        done(&p);
    }
    {
        /* Runtime div-by-zero must NOT fire at type-check time
         * (the consteval noise filter rolls back "not a constant"
         * for runtime initializers). */
        P p; run(&p, "fn f(x: i32, y: i32) { let z = x / y; }\n");
        CHECK(p.ok, "runtime x/y: tc_ok");
        CHECK(!has_error_with(&p.diag, "division by zero"),
              "runtime div: no spurious diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 1; x /= 0; }\n");
        CHECK(!p.ok, "compound division by const zero: must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "compound div-by-zero diag");
        done(&p);
    }
    {
        /* §17.5.52 — dot(vecN<AbstractInt>) evaluates as AbstractInt
         * and must reject intermediate products outside the i64 domain. */
        P p; run(&p,
            "const x = dot(vec2(9223372036854775807, 9223372036854775807), "
            "vec2(2, 2));\n");
        CHECK(!p.ok, "abstract-int dot overflow: must error");
        CHECK(has_error_with(&p.diag, "overflow"),
              "abstract-int dot overflow diag");
        done(&p);
    }
    {
        /* §8.9 — shift amount out of range at const-eval. */
        P p; run(&p, "fn f() { let x = 1u << 33u; }\n");
        CHECK(!p.ok, "1u << 33u: must error");
        CHECK(has_error_with(&p.diag, "shift amount"),
              "shift-range diag");
        done(&p);
    }
    {
        /* §8.9 — a constant shift amount is checked even when the
         * shifted value is runtime. */
        P p; run(&p, "fn f() { var v: i32 = 0; let x = v << 32u; }\n");
        CHECK(!p.ok, "runtime lhs << 32u: must error");
        CHECK(has_error_with(&p.diag, "shift amount"),
              "runtime shift const amount diag");
        done(&p);
    }
    {
        /* §8.9 — AbstractInt right shift by >= 64 is defined and
         * saturates in const-eval; left shift still range-checks. */
        P p; run(&p, "const x = 1 >> 64;\n");
        CHECK(p.ok, "abstract-int 1 >> 64: ok");
        done(&p);
    }
    {
        /* §8.9 — shifts are only defined for scalar/vector operands,
         * not arrays even when their element type is integer. */
        P p; run(&p,
            "var<private> a: array<u32, 4>;\n"
            "fn f() { let x = a << a; }\n");
        CHECK(!p.ok, "array << array: must error");
        CHECK(has_error_with(&p.diag, "shift operands"),
              "array shift diag");
        done(&p);
    }
    {
        /* §15.7.3 finite-math — AbstractFloat overflow producing inf
         * is rejected (1e200 * 1e200 ≈ 1e400 > 1.8e308). */
        P p; run(&p, "fn f() { let x: f32 = 1e200 * 1e200; }\n");
        CHECK(!p.ok, "1e200 * 1e200: must error");
        CHECK(has_error_with(&p.diag, "infinity"),
              "fp inf diag");
        done(&p);
    }
    {
        /* §17.5.53 — pow(0, y) is only defined for positive y in
         * const-eval validation. */
        P p; run(&p, "const x = pow(0.0, 0.0);\n");
        CHECK(!p.ok, "pow(0, 0): must error");
        CHECK(has_error_with(&p.diag, "zero base"),
              "pow zero-base diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { _ = atanh(1f); }\n");
        CHECK(!p.ok, "atanh(1f) const builtin domain: must error");
        CHECK(has_error_with(&p.diag, "builtin 'atanh' domain error"),
              "atanh runtime-literal domain diag");
        done(&p);
    }
    {
        /* §17.5.16 — clamp() rejects constant low/high bounds with
         * low > high even when the clamped value is runtime. */
        P p; run(&p,
            "fn f() { var v: f32; let x = clamp(v, 1.0, 0.0); }\n");
        CHECK(!p.ok, "clamp runtime value with low > high: must error");
        CHECK(has_error_with(&p.diag, "low must be <= high"),
              "clamp low/high diag");
        done(&p);
    }
    {
        /* §17.5.62 — smoothstep() rejects equal constant edges even
         * when x is runtime. */
        P p; run(&p,
            "fn f() { var x: f32; let y = smoothstep(0.0, 0.0, x); }\n");
        CHECK(!p.ok, "smoothstep equal edges: must error");
        CHECK(has_error_with(&p.diag, "edge0 and edge1 must differ"),
              "smoothstep edge diag");
        done(&p);
    }
    {
        /* §17.5.56 — normalize() squares components during const-eval;
         * those intermediate operations must stay inside the result type. */
        P p; run(&p,
            "const x = normalize(vec2<f32>("
            "3.4028234663852886e+38f, 3.4028234663852886e+38f));\n");
        CHECK(!p.ok, "normalize f32 intermediate overflow: must error");
        CHECK(has_error_with(&p.diag, "infinity") ||
              has_error_with(&p.diag, "non-finite"),
              "normalize f32 overflow diag");
        done(&p);
    }
    {
        /* §8.9 — shift amount must be u32 (or vec<u32>); `1u << 2i`
         * was previously rejected as "wrong reason" (binary numeric). */
        P p; run(&p, "fn f() { let x = 1u << 2i; }\n");
        CHECK(!p.ok, "1u << 2i: must error");
        CHECK(has_error_with(&p.diag, "shift amount must be u32"),
              "shift amount type diag");
        done(&p);
    }
    {
        /* §8.9 — abstract-int shift amount converts to u32: ok. */
        P p; run(&p, "fn f() { let x = 1u << 2; }\n");
        CHECK(p.ok, "1u << 2: ok");
        done(&p);
    }
    {
        /* §8.9 — vector LHS with scalar RHS is rejected. */
        P p; run(&p,
            "fn f() { let v = vec2<u32>(1u, 2u); let x = v << 1u; }\n");
        CHECK(!p.ok, "vec << scalar: must error");
        CHECK(has_error_with(&p.diag, "shift operand shapes must match"),
              "shift shape diag");
        done(&p);
    }
    {
        /* §8.1 — short-circuit at const-eval: RHS skipped when LHS
         * already decides the result.  The RHS would otherwise trip
         * div-by-zero. */
        P p; run(&p,
            "const X: bool = false && (5 / 0 == 0);\n");
        CHECK(p.ok, "false && (5/0==0): ok via short-circuit");
        CHECK(!has_error_with(&p.diag, "division by zero"),
              "no spurious div-by-zero");
        done(&p);
    }
    {
        /* §8.1 — || mirror: true || (5/0==0) skips RHS. */
        P p; run(&p,
            "const X: bool = true || (5 / 0 == 0);\n");
        CHECK(p.ok, "true || (5/0==0): ok via short-circuit");
        done(&p);
    }
    {
        /* §8.1 — non-short-circuit path still walks RHS. */
        P p; run(&p,
            "const X: bool = true && (5 / 0 == 0);\n");
        CHECK(!p.ok, "true && (5/0==0): RHS evaluated, must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "div-by-zero from RHS");
        done(&p);
    }
    {
        /* §8.10 — argument count mismatch on user-fn call. */
        P p; run(&p, "fn g(a: i32, b: i32) {} fn f() { g(1); }\n");
        CHECK(!p.ok, "g(1) of g(i32,i32): must error");
        CHECK(has_error_with(&p.diag, "argument count mismatch"),
              "arg-count diag");
        done(&p);
    }
    {
        /* §8.10 — argument type mismatch (concrete f32 → concrete i32
         * has no automatic conversion). */
        P p; run(&p, "fn g(a: i32) {} fn f() { g(1.5f); }\n");
        CHECK(!p.ok, "g(1.5f): must error");
        CHECK(has_error_with(&p.diag, "argument 1"),
              "arg-type diag");
        done(&p);
    }
    {
        /* §8.10 — happy path: argument that converts via abstract→i32
         * stays legal. */
        P p; run(&p, "fn g(a: i32) {} fn f() { g(1); }\n");
        CHECK(p.ok, "g(1) of g(i32): tc_ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let abs = 4;\n"
            "  _ = abs(1);\n"
            "}\n");
        CHECK(!p.ok, "shadowed builtin callee is not callable");
        CHECK(has_error_with(&p.diag, "not callable"),
              "shadowed builtin callee diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let vec2 = 4;\n"
            "  _ = vec2<f32>(1.0, 2.0);\n"
            "}\n");
        CHECK(!p.ok, "shadowed type generator callee is not callable");
        CHECK(has_error_with(&p.diag, "not callable") ||
              has_error_with(&p.diag, "not a type"),
              "shadowed typegen callee diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> sampler: f32;\n"
            "@group(0) @binding(0) var s: sampler;\n");
        CHECK(!p.ok, "shadowed handle type is not a type");
        CHECK(has_error_with(&p.diag, "not a type"),
              "shadowed handle type diag");
        done(&p);
    }
    {
        /* §8.13 — `&v.x` (vector component) rejected. */
        P p; run(&p, "fn f() { var v: vec3<f32>; let p = &v.x; }\n");
        CHECK(!p.ok, "&v.x: must error");
        CHECK(has_error_with(&p.diag, "vector component"),
              "addr-of-vec-component diag");
        done(&p);
    }
    {
        /* §8.13 — `&t` on a handle-AS texture rejected. */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "fn f() { let p = &t; }\n");
        CHECK(!p.ok, "&t (texture): must error");
        CHECK(has_error_with(&p.diag, "handle"),
              "addr-of-handle diag");
        done(&p);
    }

    /* ── §9 — statement-level conformance ─────────────────────── */
    {
        /* §9.2.1 — RHS type vs LHS store type. */
        P p; run(&p, "fn f() { var x: i32 = 0; x = 1.5; }\n");
        CHECK(!p.ok, "x = 1.5 with i32 store type: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "assign type-mismatch diag");
        done(&p);
    }
    {
        /* §9.2.2 — phony RHS must be constructible / ptr / texture / sampler. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: array<f32>;\n"
            "fn f() { _ = a; }\n");
        CHECK(!p.ok, "_ = runtime-array: must error");
        CHECK(has_error_with(&p.diag, "constructible"),
              "phony-rhs diag");
        done(&p);
    }
    {
        /* Phony positives — scalar / pointer arg / texture all legal. */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "fn f() { _ = 1; _ = t; }\n");
        CHECK(p.ok, "_ = scalar / texture: tc_ok");
        done(&p);
    }
    {
        /* §9.3 — `++` requires concrete integer scalar. */
        P p; run(&p, "fn f() { var a: f32 = 0.0; a++; }\n");
        CHECK(!p.ok, "a++ on f32: must error");
        CHECK(has_error_with(&p.diag, "integer"),
              "inc-on-float diag");
        done(&p);
    }
    {
        /* §9.4.2 — switch selector must be concrete integer scalar. */
        P p; run(&p,
            "fn f() { var x: f32 = 0.0; switch x { default: {} } }\n");
        CHECK(!p.ok, "switch f32 selector: must error");
        CHECK(has_error_with(&p.diag, "selector must be a concrete integer"),
              "switch f32 diag");
        done(&p);
    }
    {
        /* §9.4.2 — duplicate default reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { default: {} default: {} } }\n");
        CHECK(!p.ok, "switch two defaults: must error");
        CHECK(has_error_with(&p.diag, "more than one default"),
              "two-defaults diag");
        done(&p);
    }
    {
        /* §9.4.2 — missing default reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0; switch x { case 0: {} } }\n");
        CHECK(!p.ok, "switch missing default: must error");
        CHECK(has_error_with(&p.diag, "exactly one default"),
              "missing-default diag");
        done(&p);
    }
    {
        /* §9.4.2 — duplicate case selectors reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0: {} case 0: {} default: {} } }\n");
        CHECK(!p.ok, "switch duplicate case: must error");
        CHECK(has_error_with(&p.diag, "duplicate case selector"),
              "duplicate-case diag");
        done(&p);
    }
    {
        /* §9.4.2 — concrete case selectors must have a common type. */
        P p; run(&p,
            "fn f() { switch 1 { case i32(1): {} case 2u: {} default: {} } }\n");
        CHECK(!p.ok, "switch mixed concrete case types: must error");
        CHECK(has_error_with(&p.diag, "earlier case selector type"),
              "mixed concrete case type diag");
        done(&p);
    }
    {
        /* Happy switch path: distinct cases + default. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0: {} case 1, 2: {} default: {} } }\n");
        CHECK(p.ok, "well-formed switch: tc_ok");
        done(&p);
    }
    {
        /* §9.4.2 — case selector must be a const-expression.  A `let`
         * is not a const-expression in WGSL. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  let k: i32 = 1;\n"
            "  switch x { case k: {} default: {} } }\n");
        CHECK(!p.ok, "non-const case selector: must error");
        CHECK(has_error_with(&p.diag, "case selector must be a const-expression"),
              "non-const case diag");
        done(&p);
    }
    {
        /* §9.4.2 — `const` selector is a const-expression: ok. */
        P p; run(&p,
            "const K: i32 = 1;\n"
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case K: {} default: {} } }\n");
        CHECK(p.ok, "const case selector: ok");
        done(&p);
    }
    {
        /* §9.4.9 — `return` inside continuing is rejected. */
        P p; run(&p, "fn f() { loop { continuing { return; } } }\n");
        CHECK(!p.ok, "return in continuing: must error");
        CHECK(has_error_with(&p.diag, "'return' is not allowed inside"),
              "return-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.9 — `continue` inside continuing is rejected. */
        P p; run(&p, "fn f() { loop { continuing { continue; } } }\n");
        CHECK(!p.ok, "continue in continuing: must error");
        CHECK(has_error_with(&p.diag, "'continue' is not allowed inside"),
              "continue-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.6 — `break` inside continuing (no inner loop/switch)
         * would exit the loop's continuing block. */
        P p; run(&p,
            "fn f() {\n"
            "  loop { continuing { if (true) { break; } } }\n"
            "}\n");
        CHECK(!p.ok, "break in continuing: must error");
        CHECK(has_error_with(&p.diag, "'break' is not allowed to exit"),
              "break-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.6 — `break` inside a switch inside continuing is OK
         * (it targets the switch, not the outer loop). */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      switch 1 { case 0: { break; } default: { break; } }\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "break in switch in continuing: ok");
        done(&p);
    }
    {
        /* §9.4.6 — `break_if` must be the final statement of a
         * continuing block.  Anything after it makes the parser emit
         * "expected '}'" rather than silently accepting. */
        P p; run(&p,
            "fn f() { loop { continuing { break if true; let x = 1; } } }\n");
        CHECK(!p.ok, "break_if not last in continuing: must error");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { loop { continuing { break if 1i; } } }\n");
        CHECK(!p.ok, "break_if non-bool condition: must error");
        CHECK(has_error_with(&p.diag, "'break if' condition must evaluate to bool"),
              "break_if condition type diag");
        done(&p);
    }
    {
        /* §7.3 — var<private> store type must be storable; texture
         * types are excluded. */
        P p; run(&p, "var<private> t: texture_2d<f32>;\n");
        CHECK(!p.ok, "var<private> texture_2d: must error");
        CHECK(has_error_with(&p.diag, "var<private> store type must be storable"),
              "var<private> texture diag");
        done(&p);
    }
    {
        /* §9.4.8 / §9.7 — `continue` inside an inner loop inside a
         * continuing block targets the inner loop, not the continuing.
         * Must NOT be flagged.  The inner loop has its own exit edge
         * via `if (true) { break; }`. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      loop {\n"
            "        if (true) { break; }\n"
            "        continue;\n"
            "      }\n"
            "      break if true;\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continue in inner loop inside continuing: ok");
        CHECK(!has_error_with(&p.diag, "'continue' is not allowed"),
              "no continue-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.2 — two `default` keywords in the same case-selector
         * list (`case 0, default, default, 2:`) is invalid. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0, default, default, 2: { } }\n"
            "}\n");
        CHECK(!p.ok, "two defaults in one list: must error");
        CHECK(has_error_with(&p.diag, "'default' must not appear more than once in the same case"),
              "double-default-in-list diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      loop {\n"
            "        if (true) { break; }\n"
            "        return;\n"
            "      }\n"
            "      break if true;\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(!p.ok, "return in inner loop inside continuing: must error");
        CHECK(has_error_with(&p.diag, "'return' is not allowed inside a continuing block"),
              "return-in-inner-loop-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.8 — continue before a decl used in continuing. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    if (true) { continue; }\n"
            "    let x = 1;\n"
            "    continuing { let y = x; break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(!p.ok, "continue before referenced decl: must error");
        CHECK(has_error_with(&p.diag, "would re-enter the loop's continuing block"),
              "continue-past-decl diag");
        done(&p);
    }
    {
        /* §9.4.8 — continue AFTER the decl is fine. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    let x = 1;\n"
            "    if (true) { continue; }\n"
            "    continuing { let y = x; break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continue after decl: ok");
        done(&p);
    }
    {
        /* §9.4.8 — continuing that doesn't reference any loop-local
         * decl: continues are unrestricted. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    if (true) { continue; }\n"
            "    let x = 1;\n"
            "    continuing { break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continuing references no loop-local decl: ok");
        done(&p);
    }
    {
        /* §9.4.10 — void fn with `return expr;` rejected. */
        P p; run(&p, "fn f() { return 1; }\n");
        CHECK(!p.ok, "void return expr: must error");
        CHECK(has_error_with(&p.diag, "void function"),
              "void-return-expr diag");
        done(&p);
    }
    {
        /* §9.4.10 — non-void fn with bare `return;` rejected. */
        P p; run(&p, "fn f() -> i32 { return; }\n");
        CHECK(!p.ok, "bare return in non-void fn: must error");
        CHECK(has_error_with(&p.diag, "requires an expression"),
              "bare-return diag");
        done(&p);
    }
    {
        /* §9.4.10 — return value type mismatch. */
        P p; run(&p, "fn f() -> i32 { return 1.5; }\n");
        CHECK(!p.ok, "return type mismatch: must error");
        CHECK(has_error_with(&p.diag, "does not match function"),
              "return-type-mismatch diag");
        done(&p);
    }
    {
        /* §9.4.10 — happy path: matching return types. */
        P p; run(&p, "fn f() -> i32 { return 0; }\n");
        CHECK(p.ok, "return with matching type: tc_ok");
        done(&p);
    }
    {
        /* §9.5 — call statement discards `@must_use` result. */
        P p; run(&p,
            "@must_use fn g() -> i32 { return 0; }\n"
            "fn f() { g(); }\n");
        CHECK(!p.ok, "@must_use call discarded: must error");
        CHECK(has_error_with(&p.diag, "@must_use"),
              "must_use diag");
        done(&p);
    }
    {
        /* §9.5 — builtin `@must_use` comes from generated metadata. */
        P p; run(&p, "fn f() { sin(1.0); }\n");
        CHECK(!p.ok, "builtin @must_use call discarded: must error");
        CHECK(has_error_with(&p.diag, "@must_use"),
              "builtin must_use diag");
        done(&p);
    }
    {
        /* §9.5 — consuming the result is legal. */
        P p; run(&p,
            "@must_use fn g() -> i32 { return 0; }\n"
            "fn f() { let _x = g(); }\n");
        CHECK(p.ok, "@must_use call consumed: tc_ok");
        done(&p);
    }

    /* ── §11 — Function restrictions ──────────────────────────── */
    {
        /* §11.4 — entry points must never be call targets. */
        P p; run(&p,
            "@compute @workgroup_size(1) fn main_cs() {}\n"
            "fn other() { main_cs(); }\n");
        CHECK(!p.ok, "calling @compute entry: must error");
        CHECK(has_error_with(&p.diag,
                  "cannot call entry point 'main_cs'"),
              "entry-point-call diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@vertex fn vs() -> @builtin(position) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n"
            "fn other() { let x = vs(); }\n");
        CHECK(!p.ok, "calling @vertex entry: must error");
        CHECK(has_error_with(&p.diag, "vertex"),
              "entry-point-call vertex diag");
        done(&p);
    }
    {
        /* §11.4 — vertex shader must return @builtin(position).
         * Direct return-attribute case. */
        P p; run(&p,
            "@vertex fn vs() -> @location(0) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n");
        CHECK(!p.ok, "vertex w/o @position: must error");
        CHECK(has_error_with(&p.diag, "@builtin(position)"),
              "vertex-no-position diag");
        done(&p);
    }
    {
        /* §11.4 — happy path: @vertex with @builtin(position) on the
         * return type. */
        P p; run(&p,
            "@vertex fn vs() -> @builtin(position) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n");
        CHECK(p.ok, "vertex w/ @position: tc_ok");
        done(&p);
    }
    {
        /* §11.4 — struct return: walks members for @builtin(position). */
        P p; run(&p,
            "struct VsOut {\n"
            "  @builtin(position) pos: vec4<f32>,\n"
            "  @location(0) color: vec3<f32>,\n"
            "}\n"
            "@vertex fn vs() -> VsOut {\n"
            "  return VsOut(vec4<f32>(0.0), vec3<f32>(1.0));\n"
            "}\n");
        CHECK(p.ok, "vertex returning struct with @position: tc_ok");
        done(&p);
    }
    {
        /* §11.4 — struct return without @builtin(position) member. */
        P p; run(&p,
            "struct VsOut {\n"
            "  @location(0) color: vec3<f32>,\n"
            "}\n"
            "@vertex fn vs() -> VsOut {\n"
            "  return VsOut(vec3<f32>(1.0));\n"
            "}\n");
        CHECK(!p.ok, "vertex returning struct w/o @position: must error");
        CHECK(has_error_with(&p.diag, "@builtin(position)"),
              "vertex-struct-no-position diag");
        done(&p);
    }
    {
        /* §11.4 — @fragment and @compute are NOT subject to the
         * vertex-position rule. */
        P p; run(&p,
            "@fragment fn fs() -> @location(0) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n"
            "@compute @workgroup_size(1) fn cs() {}\n");
        CHECK(p.ok, "fragment + compute w/o @position: tc_ok");
        done(&p);
    }

    /* ── §12 — Attributes ─────────────────────────────────────── */
    /* The conformance round-up below mostly drives `validate.c`'s
     * attribute table — but it uses the same `run` harness so a
     * regression in any of the new rules surfaces here. */
    {
        /* §12 general — duplicate attribute reject. */
        WGSLResult *r = wgsl_check(
            "@vertex fn vs(@location(0) @location(1) v: f32) -> "
            "@builtin(position) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location(0) @location(1): must error");
        const char *needle = "must not be specified more than once";
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, needle)) { hit = 1; break; }
        }
        CHECK(hit, "duplicate-attr diag");
        wgsl_free(r);
    }
    {
        /* §12.1 @align positive. */
        WGSLResult *r = wgsl_check("struct S { @align(-4) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@align(-4): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@align value must be positive")) { hit = 1; break; }
        }
        CHECK(hit, "@align-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.1 @align power of 2. */
        WGSLResult *r = wgsl_check("struct S { @align(7) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@align(7): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "power of 2")) { hit = 1; break; }
        }
        CHECK(hit, "@align-power2 diag");
        wgsl_free(r);
    }
    {
        /* §12.7 @group non-negative. */
        WGSLResult *r = wgsl_check(
            "@group(-1) @binding(0) var<storage> b: array<f32>;\n");
        CHECK(!wgsl_ok(r), "@group(-1): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "non-negative")) { hit = 1; break; }
        }
        CHECK(hit, "@group-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.9 @interpolate without @location. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@interpolate(flat) v: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@interpolate w/o @location: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@interpolate")) { hit = 1; break; }
        }
        CHECK(hit, "@interpolate-no-location diag");
        wgsl_free(r);
    }
    {
        /* §12.10 @invariant only on @builtin(position). */
        WGSLResult *r = wgsl_check(
            "struct S {\n"
            "  @builtin(position) p: vec4<f32>,\n"
            "  @location(0) @invariant v: f32,\n"
            "}\n"
            "@vertex fn vs() -> S { return S(vec4<f32>(0), 0.0); }\n");
        CHECK(!wgsl_ok(r), "@invariant on non-position: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@invariant")) { hit = 1; break; }
        }
        CHECK(hit, "@invariant-pos diag");
        wgsl_free(r);
    }
    {
        /* §12.12 @must_use on void fn. */
        WGSLResult *r = wgsl_check("@must_use fn g() {}\n");
        CHECK(!wgsl_ok(r), "@must_use on void: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@must_use")) { hit = 1; break; }
        }
        CHECK(hit, "@must_use-void diag");
        wgsl_free(r);
    }
    {
        /* §12.13 @size positive. */
        WGSLResult *r = wgsl_check("struct S { @size(-4) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@size(-4): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@size value must be positive")) { hit = 1; break; }
        }
        CHECK(hit, "@size-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(0). */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(0) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(0): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@workgroup_size argument must be a positive")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-zero diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(1.0) — float forbidden. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1.0) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(1.0): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "floating-point")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-float diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(1i, 1u) — same-type rule. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1i, 1u) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(1i, 1u): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "same type")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-mixed diag");
        wgsl_free(r);
    }
    {
        /* §12.14 happy path — all-AbstractInt accepted. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(8, 8, 1) fn cs() {}\n");
        CHECK(wgsl_ok(r), "@workgroup_size(8, 8, 1): tc_ok");
        wgsl_free(r);
    }

    /* ── §13 — Entry-point conformance ────────────────────────── */
    {
        /* §13.2 — @compute must not have a return type. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs() -> i32 { return 0; }\n");
        CHECK(!wgsl_ok(r), "@compute return: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "must not have a return")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "compute-return diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — @builtin matrix: stage / direction / type. */
        WGSLResult *r = wgsl_check(
            "@vertex fn vs(@builtin(vertex_index) i: i32) -> "
            "@builtin(position) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "vertex_index as i32: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "requires type u32")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-type diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — @position on a compute param rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs(@builtin(position) p: vec4<f32>) {}\n");
        CHECK(!wgsl_ok(r), "@position on compute: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "not valid on this shader stage")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-stage diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — duplicate @builtin on one entry point rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs("
            "@builtin(global_invocation_id) a: vec3<u32>, "
            "@builtin(global_invocation_id) b: vec3<u32>) {}\n");
        CHECK(!wgsl_ok(r), "dup @builtin: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "more than once")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-dup diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — @location with bool type rejected. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) b: bool) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location bool: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "numeric scalar or vector")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "location-bool diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — bare entry-point param with no @location /
         * @builtin rejected. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(v: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "bare entry param: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@location or @builtin")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "bare-entry-param diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — @location on compute entry rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs(@location(0) v: vec3<f32>) {}\n");
        CHECK(!wgsl_ok(r), "@location on compute: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "compute")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "compute-user-IO diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — duplicate @location across params. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) a: f32, @location(0) b: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "dup @location params: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "duplicate @location")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "dup-location diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — input vs output share @location is allowed. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) a: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(wgsl_ok(r), "input/output @location(0) overlap: tc_ok");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — @location on a struct-typed param rejected. */
        WGSLResult *r = wgsl_check(
            "struct D { x: f32 }\n"
            "@fragment fn fs(@location(0) in: D) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location on struct param: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "struct-typed entry parameter")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "location-on-struct-param diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — nested IO struct rejected. */
        WGSLResult *r = wgsl_check(
            "struct Inner { @location(0) v: f32 }\n"
            "struct Outer { i: Inner }\n"
            "@fragment fn fs(in: Outer) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "nested IO struct: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "cannot nest")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "nested-IO diag");
        wgsl_free(r);
    }
    {
        /* CTS parity: integer-typed inter-stage user-defined IO requires
         * explicit @interpolate(flat). */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) v: u32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "integer IO w/o flat: must error");
        wgsl_free(r);
    }
    {
        /* §13.3.1.4 — happy path: integer IO with @interpolate(flat). */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) @interpolate(flat) v: u32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(wgsl_ok(r), "integer IO w/ flat: tc_ok");
        wgsl_free(r);
    }
    {
        /* §13.3.2 — duplicate (group, binding) pair. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<storage> a: array<f32>;\n"
            "@group(0) @binding(0) var<storage> b: array<f32>;\n"
            "@compute @workgroup_size(1) fn main() {\n"
            "  _ = arrayLength(&a); _ = arrayLength(&b);\n"
            "}\n");
        CHECK(!wgsl_ok(r), "dup binding: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "duplicate resource binding")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "dup-binding diag");
        wgsl_free(r);
    }

    /* ── §14 — Memory layout (uniform AS constraints) ─────────── */
    {
        /* §14.4.5 — uniform AS forbids runtime-sized array. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<f32>;\n");
        CHECK(!wgsl_ok(r), "uniform runtime-array: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "runtime-sized")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "uniform-rta diag");
        wgsl_free(r);
    }
    {
        /* With uniform_buffer_standard_layout support enabled, uniform AS
         * arrays use standard layout and no 16-byte stride bump is needed. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<f32, 8>;\n");
        CHECK(wgsl_ok(r), "uniform array<f32, N>: ok under standard layout");
        wgsl_free(r);
    }
    {
        /* §14.4.5 happy path: uniform array of vec4<f32> (stride 16). */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<vec4<f32>, 4>;\n");
        CHECK(wgsl_ok(r), "uniform array<vec4<f32>, N>: tc_ok");
        wgsl_free(r);
    }

    /* ── §17 — Built-in function guards (selected fixes) ─────── */
    {
        /* §17.2 — bitcast<T> requires T to be a numeric scalar/vec. */
        WGSLResult *r = wgsl_check("fn f() { let x = bitcast<bool>(1u); }\n");
        CHECK(!wgsl_ok(r), "bitcast<bool>: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message,
                       "bitcast<T> requires T to be a concrete numeric")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "bitcast-non-numeric diag");
        wgsl_free(r);
    }
    {
        /* §17.2 — bitcast<u32>(f32) stays legal. */
        WGSLResult *r = wgsl_check(
            "fn f() { let x = bitcast<u32>(1.5f); }\n");
        CHECK(wgsl_ok(r), "bitcast<u32>(f32): tc_ok");
        wgsl_free(r);
    }
    {
        /* §17.2 — bitcast source must also be a concrete numeric
         * scalar/vector. */
        WGSLResult *r = wgsl_check(
            "fn f() { let x = bitcast<i32>(true); }\n");
        CHECK(!wgsl_ok(r), "bitcast<i32>(bool): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message,
                       "bitcast source type must be a concrete numeric")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "bitcast-bad-source diag");
        wgsl_free(r);
    }
    {
        /* §17.2 — f16 bitcasts are only legal in 32-bit word packs
         * (vec2h / vec4h), not scalar f16 or vec3h. */
        WGSLResult *r = wgsl_check(
            "enable f16;\n"
            "fn f() { var src: vec3h; let x = bitcast<u32>(src); }\n");
        CHECK(!wgsl_ok(r), "bitcast<u32>(vec3h): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message,
                       "32-bit word representation")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "bitcast-vec3h diag");
        wgsl_free(r);
    }
    {
        /* §17.2 — vec2h packs into one 32-bit scalar. */
        WGSLResult *r = wgsl_check(
            "enable f16;\n"
            "fn f() { var src: u32; let x = bitcast<vec2h>(src); }\n");
        CHECK(wgsl_ok(r), "bitcast<vec2h>(u32): tc_ok");
        wgsl_free(r);
    }
    {
        /* §17.4 — arrayLength on a local fixed-size array rejected. */
        WGSLResult *r = wgsl_check(
            "fn f() { var a: array<f32, 3>; let x = arrayLength(&a); }\n");
        CHECK(!wgsl_ok(r), "arrayLength on local: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message,
                       "arrayLength requires a pointer to a runtime-sized")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "arrayLength-local diag");
        wgsl_free(r);
    }
    {
        /* §17.4 — arrayLength happy path through storage buffer. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<storage> a: array<f32>;\n"
            "fn f() -> u32 { return arrayLength(&a); }\n");
        CHECK(wgsl_ok(r), "arrayLength on storage: tc_ok");
        wgsl_free(r);
    }
    {
        /* §17.11 — workgroupUniformLoad requires ptr<workgroup,T,read_write>. */
        WGSLResult *r = wgsl_check(
            "var<workgroup> a: u32;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x: u32 = workgroupUniformLoad(&a);\n"
            "}\n");
        CHECK(wgsl_ok(r), "workgroupUniformLoad on workgroup var: tc_ok");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check(
            "var<private> a: u32;\n"
            "fn f() { let x = workgroupUniformLoad(&a); }\n");
        CHECK(!wgsl_ok(r), "workgroupUniformLoad on private var: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message,
                       "no matching overload for 'workgroupUniformLoad'")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "workgroupUniformLoad-private no-overload diag");
        wgsl_free(r);
    }
    {
        WGSLResult *r = wgsl_check(
            "override N = 4u;\n"
            "var<workgroup> a: array<u32, N>;\n"
            "fn f() { let x = workgroupUniformLoad(&a); }\n");
        CHECK(!wgsl_ok(r), "workgroupUniformLoad override array: must error");
        wgsl_free(r);
    }
    {
        /* §17.8 — atomicCompareExchangeWeak's synthetic result struct
         * is constructible and `let` can bind it. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<storage,read_write> a: atomic<i32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let r = atomicCompareExchangeWeak(&a, 0, 1);\n"
            "  let v = r.old_value;\n"
            "  let x = r.exchanged;\n"
            "}\n");
        CHECK(wgsl_ok(r), "atomicCompareExchangeWeak + let: tc_ok");
        wgsl_free(r);
    }

    /* ── §18 — Grammar shapes ─────────────────────────────────── */
    {
        /* §18 — leading attribute list permitted on statements
         * (compound / if / for / while / loop / switch).  The
         * attributes are parsed-and-discarded for now; the meaningful
         * shape (`@diagnostic(off, derivative_uniformity)` on a
         * compound block) parses without error. */
        WGSLResult *r = wgsl_check(
            "fn f() {\n"
            "  @diagnostic(off, derivative_uniformity) {\n"
            "    let x = 0;\n"
            "  }\n"
            "  @diagnostic(off, derivative_uniformity) if (true) {\n"
            "    let y = 0;\n"
            "  }\n"
            "}\n");
        CHECK(wgsl_ok(r), "@diagnostic on compound + if: tc_ok");
        wgsl_free(r);
    }
    {
        /* §18 / §9.4.2 — `default` mid case-selector list is a legal
         * selector form; the clause counts as the switch's default. */
        WGSLResult *r = wgsl_check(
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0, default, 2: { } }\n"
            "}\n");
        CHECK(wgsl_ok(r),
              "case 0, default, 2: counts as the default clause");
        wgsl_free(r);
    }
    {
        /* §18 — case clause with optional colon. */
        WGSLResult *r = wgsl_check(
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0 { } default { } }\n"
            "}\n");
        CHECK(wgsl_ok(r), "case / default without ':': tc_ok");
        wgsl_free(r);
    }
    {
        /* §18 — trailing comma in @enable / @workgroup_size args /
         * struct member list / fn param list. */
        WGSLResult *r = wgsl_check(
            "enable f16,;\n"
            "struct S { a: i32, b: f32, }\n"
            "@compute @workgroup_size(1, 2, 3,) fn cs(@builtin(local_invocation_index) i: u32,) {}\n");
        CHECK(wgsl_ok(r), "trailing commas across the grammar: tc_ok");
        wgsl_free(r);
    }
    /* §9.4.1 — `if` condition must be bool. */
    {
        P p; run(&p, "fn f() { if 1 { } }\n");
        CHECK(!p.ok, "if 1: must error");
        CHECK(has_error_with(&p.diag,
            "'if' condition must evaluate to bool"),
              "if non-bool: diag");
        done(&p);
    }
    /* §5 / §7.3 — forward-ref resolution of a module-scope `var<storage>`
     * while walking a fn body must NOT trigger the "must be at module
     * scope" gate.  Regression: the `tc->in_function = 1` flag set by
     * `type_function` leaked into `type_decl_var`'s scope-placement
     * check when `type_ident` lazily forwarded an as-yet-untyped var
     * back into `type_module_decl`. */
    {
        /* `tbl` is used by `k()`'s body BEFORE the module-decl walk gets
         * to it.  The forward-ref path through `type_ident` triggers the
         * bug if it doesn't save/restore `in_function`. */
        P p; run(&p,
            "@compute @workgroup_size(1)\n"
            "fn k() { let x = tbl[0u]; }\n"
            "@group(0) @binding(0) var<storage, read> tbl: array<u32>;\n");
        CHECK(p.ok, "forward-ref to var<storage>: no false 'must be at module scope'");
        CHECK(!has_error_with(&p.diag, "must be at module scope"),
              "forward-ref to var<storage>: scope diag suppressed");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { if 1.5 { } }\n");
        CHECK(!p.ok, "if 1.5: must error");
        CHECK(has_error_with(&p.diag,
            "'if' condition must evaluate to bool"),
              "if float: diag");
        done(&p);
    }
    {
        /* `else if` cond also gated. */
        P p; run(&p, "fn f() { if true { } else if 2u { } }\n");
        CHECK(!p.ok, "else if 2u: must error");
        CHECK(has_error_with(&p.diag,
            "'if' condition must evaluate to bool"),
              "else if non-bool: diag");
        done(&p);
    }
    {
        /* legal: cond is bool. */
        P p; run(&p, "fn f() { if true { } else if false { } else { } }\n");
        CHECK(p.ok, "bool cond chain: tc_ok");
        done(&p);
    }
    /* §9.4.3 — `while` condition must be bool. */
    {
        P p; run(&p, "fn f() { while 1 { break; } }\n");
        CHECK(!p.ok, "while 1: must error");
        CHECK(has_error_with(&p.diag,
            "'while' condition must evaluate to bool"),
              "while non-bool: diag");
        done(&p);
    }
    /* §9.4.4 — `for` condition must be bool. */
    {
        P p; run(&p, "fn f() { for (var i = 0; i; i = i + 1) { } }\n");
        CHECK(!p.ok, "for(...; i; ...): must error");
        CHECK(has_error_with(&p.diag,
            "'for' condition must evaluate to bool"),
              "for non-bool: diag");
        done(&p);
    }
    {
        /* legal: bool cond + empty cond (`for (;;)`) both pass. */
        P p; run(&p,
            "fn f() {\n"
            "  for (var i = 0; i < 10; i = i + 1) { }\n"
            "  for (;;) { break; }\n"
            "}\n");
        CHECK(p.ok, "for with bool cond + for(;;): tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { true; }\n");
        CHECK(!p.ok, "plain expression statement: must error");
        CHECK(has_error_with(&p.diag, "expression statement must be a function call"),
              "plain expression statement diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { for (;;let i = 1) { break; } }\n");
        CHECK(!p.ok, "for update declaration: must error");
        CHECK(has_error_with(&p.diag, "for update must not be a declaration"),
              "for update declaration diag");
        done(&p);
    }
    /* §17 — `frexp(T)` / `modf(T)` return synthetic struct types
     * `__frexp_result<T>` { fract: T, exp: I32-shape } and
     * `__modf_result<T>` { fract: T, whole: T }. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let r = frexp(1.5);\n"
            "  let x: f32 = r.fract;\n"
            "  let e: i32 = r.exp;\n"
            "}\n");
        CHECK(p.ok, "frexp scalar fract/exp typed correctly");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let r = frexp(vec3<f32>(1.5));\n"
            "  let x: vec3<f32> = r.fract;\n"
            "  let e: vec3<i32> = r.exp;\n"
            "}\n");
        CHECK(p.ok, "frexp vec fract/exp typed correctly");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let r = modf(1.5);\n"
            "  let a: f32 = r.fract;\n"
            "  let b: f32 = r.whole;\n"
            "}\n");
        CHECK(p.ok, "modf scalar fract/whole typed correctly");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = frexp(1.5); let x = r.banana; }\n");
        CHECK(!p.ok, "frexp bad field: must error");
        CHECK(has_error_with(&p.diag,
            "__frexp_result has no field 'banana'"),
              "frexp bad-field diag");
        done(&p);
    }

    /* §6.2.9 — array element type must have a generation-fixed footprint
     * (no runtime-sized array as an element of either a fixed-size or a
     * runtime-sized array). */
    {
        P p; run(&p, "alias A = array<array<f32>, 4>;\n");
        CHECK(!p.ok, "array<array<f32>, 4>: must error");
        CHECK(has_error_with(&p.diag,
            "generation-fixed footprint"),
              "fixed-array of runtime-array: CFF diag");
        done(&p);
    }
    {
        P p; run(&p, "alias A = array<array<f32>>;\n");
        CHECK(!p.ok, "array<array<f32>>: must error");
        CHECK(has_error_with(&p.diag,
            "generation-fixed footprint"),
              "runtime-array of runtime-array: CFF diag");
        done(&p);
    }
    {
        P p; run(&p, "alias A = array<4>;\n");
        CHECK(!p.ok, "array<4>: must error");
        CHECK(has_error_with(&p.diag, "array element type must be a type"),
              "array bad element-type diag");
        done(&p);
    }
    {
        /* legal: fixed-size array of fixed-size array. */
        P p; run(&p, "alias A = array<array<f32, 4>, 8>;\n");
        CHECK(p.ok, "fixed-array of fixed-array: tc_ok");
        done(&p);
    }

    fprintf(stderr, "%s  test_check  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
