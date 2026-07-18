/**
 * Composite expression type-checking: constructors, member access,
 * indexing, vec/mat operators, and user-function return types.
 *
 * Covers:
 *   - constructors: vec*<T>(args), aliased forms (vec3f, mat4x4f),
 *     scalar conversion (i32(x), f32(x))
 *   - member access:
 *       - vec swizzle .x / .xyz / .rgba (length 1 → scalar; 2-4 → vec)
 *       - struct field
 *       - swizzle range guard, invalid swizzle char
 *   - indexing: vec[i] → scalar, mat[i] → vec<rows>, array[i] → elem
 *   - binary ops on vec/mat:
 *       - vec + vec → vec
 *       - vec * scalar → vec (broadcast)
 *       - mat + mat → mat
 *       - mat * vec → vec (multiplicative dispatch)
 *       - mat * mat → mat
 *       - vec comparison → vec<bool>
 *   - user-defined function call → return type from sym->type
 *   - error: function name used as value
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

static WGSLSymbol *sym_for(const P *p, const char *want) {
    size_t wl = strlen(want);
    for (size_t i = 0; i < p->res.all_decl_count; i++) {
        WGSLSymbol *s = p->res.all_decls[i];
        if (s->name_len == wl && memcmp(s->name, want, wl) == 0) return s;
    }
    return NULL;
}

int main(void) {
    wgsl_utf8_init();

    /* Vec constructor (templated form). */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec3<f32>(1.0, 2.0, 3.0);\n"
            "}\n");
        CHECK(p.ok, "vec3<f32>(...): tc_ok");
        WGSLSymbol *v = sym_for(&p, "v");
        CHECK(v && v->type && v->type->kind == WGSL_TYPE_VEC &&
              v->type->width == 3 && v->type->ref == p.types.t_f32,
              "v: vec3<f32>");
        done(&p);
    }

    /* Vec constructor (predeclared alias form). */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec4f(0.0);\n"
            "}\n");
        CHECK(p.ok, "vec4f(0.0): tc_ok");
        WGSLSymbol *v = sym_for(&p, "v");
        CHECK(v && v->type && v->type->kind == WGSL_TYPE_VEC &&
              v->type->width == 4 && v->type->ref == p.types.t_f32,
              "v: vec4<f32>");
        done(&p);
    }

    /* Scalar conversion. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1.5;\n"
            "  let b = i32(a);\n"
            "}\n");
        CHECK(p.ok, "i32(a): tc_ok");
        WGSLSymbol *b = sym_for(&p, "b");
        CHECK(b && b->type == p.types.t_i32, "b: i32");
        done(&p);
    }

    /* Mat constructor. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let m = mat4x4f(1.0);\n"
            "}\n");
        CHECK(p.ok, "mat4x4f(1.0): tc_ok");
        WGSLSymbol *m = sym_for(&p, "m");
        CHECK(m && m->type && m->type->kind == WGSL_TYPE_MAT &&
              m->type->width == 4 && m->type->rows == 4 &&
              m->type->ref == p.types.t_f32,
              "m: mat4x4<f32>");
        done(&p);
    }

    /* Constructor partial-eval range checks. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: i32;\n"
            "  let x = vec2(68719476735, v);\n"
            "}\n");
        CHECK(!p.ok, "vec constructor const component overflow: must error");
        CHECK(has_error_with(&p.diag, "out of range for i32"),
              "vec constructor overflow diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: u32;\n"
            "  let x = array(v, -1);\n"
            "}\n");
        CHECK(!p.ok, "array constructor const component overflow: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "array constructor overflow diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct S { x: u32, y: u32 }\n"
            "fn f() {\n"
            "  var v: u32;\n"
            "  let x = S(v, -1);\n"
            "}\n");
        CHECK(!p.ok, "struct constructor const member overflow: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "struct constructor overflow diag");
        done(&p);
    }

    /* Vec swizzle. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec4f(1.0, 2.0, 3.0, 4.0);\n"
            "  let s1 = v.x;\n"
            "  let s3 = v.xyz;\n"
            "  let s4 = v.wzyx;\n"
            "  let r2 = v.rg;\n"
            "}\n");
        CHECK(p.ok, "swizzle: tc_ok");
        WGSLSymbol *s1 = sym_for(&p, "s1");
        WGSLSymbol *s3 = sym_for(&p, "s3");
        WGSLSymbol *s4 = sym_for(&p, "s4");
        WGSLSymbol *r2 = sym_for(&p, "r2");
        CHECK(s1 && s1->type == p.types.t_f32, "s1: f32 (length-1)");
        CHECK(s3 && s3->type && s3->type->kind == WGSL_TYPE_VEC &&
              s3->type->width == 3, "s3: vec3<f32>");
        CHECK(s4 && s4->type && s4->type->kind == WGSL_TYPE_VEC &&
              s4->type->width == 4, "s4: vec4<f32>");
        CHECK(r2 && r2->type && r2->type->kind == WGSL_TYPE_VEC &&
              r2->type->width == 2, "r2: vec2<f32> (rgba alias set)");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec3f(0.0);\n"
            "  let bad = v.w;\n"            /* w on vec3 is out of range */
            "}\n");
        CHECK(!p.ok, "vec3.w: must error");
        CHECK(has_error_with(&p.diag, "out of range") ||
              has_error_with(&p.diag, "swizzle"),
              "vec3.w: range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec4f(0.0);\n"
            "  let bad = v.q;\n"
            "}\n");
        CHECK(!p.ok, "v.q invalid char: must error");
        CHECK(has_error_with(&p.diag, "swizzle") ||
              has_error_with(&p.diag, "invalid"),
              "swizzle invalid-char diag");
        done(&p);
    }

    /* Struct field access. */
    {
        P p; run(&p,
            "struct V { pos: vec3<f32>, uv: vec2<f32> }\n"
            "fn f(v: V) -> vec3<f32> { return v.pos; }\n");
        CHECK(p.ok, "v.pos: tc_ok");
        /* Find the return statement's expression */
        done(&p);
    }
    {
        P p; run(&p,
            "struct V { pos: vec3<f32> }\n"
            "fn f(v: V) -> f32 { return v.zoo; }\n");
        CHECK(!p.ok, "v.zoo: must error");
        CHECK(has_error_with(&p.diag, "no field"),
              "missing-field diag");
        done(&p);
    }

    /* Indexing. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let v = vec4f(0.0);\n"
            "  let s = v[0];\n"
            "}\n");
        CHECK(p.ok, "vec[i]: tc_ok");
        WGSLSymbol *s = sym_for(&p, "s");
        CHECK(s && s->type == p.types.t_f32, "s: f32 (vec[i])");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let m = mat4x4f(1.0);\n"
            "  let col = m[2];\n"
            "}\n");
        CHECK(p.ok, "mat[i]: tc_ok");
        WGSLSymbol *col = sym_for(&p, "col");
        CHECK(col && col->type && col->type->kind == WGSL_TYPE_VEC &&
              col->type->width == 4 && col->type->ref == p.types.t_f32,
              "col: vec4<f32>");
        done(&p);
    }
    {
        P p; run(&p,
            "alias Buf = array<f32, 8>;\n"
            "fn f(a: Buf) -> f32 { return a[3]; }\n");
        CHECK(p.ok, "array[i]: tc_ok");
        done(&p);
    }

    /* Vec/mat binary ops. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec3f(1.0);\n"
            "  let b = vec3f(2.0);\n"
            "  let c = a + b;\n"
            "  let d = a * 2.0;\n"
            "  let e = 3.0 * a;\n"
            "}\n");
        CHECK(p.ok, "vec arith: tc_ok");
        WGSLSymbol *c = sym_for(&p, "c");
        WGSLSymbol *d = sym_for(&p, "d");
        WGSLSymbol *e = sym_for(&p, "e");
        CHECK(c && c->type && c->type->kind == WGSL_TYPE_VEC &&
              c->type->width == 3, "c: vec3 (vec+vec)");
        CHECK(d && d->type && d->type->kind == WGSL_TYPE_VEC &&
              d->type->width == 3, "d: vec3 (vec*scalar)");
        CHECK(e && e->type && e->type->kind == WGSL_TYPE_VEC &&
              e->type->width == 3, "e: vec3 (scalar*vec)");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let m = mat4x4f(1.0);\n"
            "  let v = vec4f(1.0);\n"
            "  let r = m * v;\n"           /* mat<4,4> * vec<4> → vec<4> */
            "  let mm = m * m;\n"           /* mat<4,4> * mat<4,4> → mat<4,4> */
            "  let mb = m + m;\n"
            "}\n");
        CHECK(p.ok, "mat*vec / mat*mat / mat+mat: tc_ok");
        WGSLSymbol *r  = sym_for(&p, "r");
        WGSLSymbol *mm = sym_for(&p, "mm");
        WGSLSymbol *mb = sym_for(&p, "mb");
        CHECK(r  && r->type  && r->type->kind  == WGSL_TYPE_VEC &&
              r->type->width == 4, "r: vec4 (mat*vec)");
        CHECK(mm && mm->type && mm->type->kind == WGSL_TYPE_MAT &&
              mm->type->width == 4 && mm->type->rows == 4,
              "mm: mat4x4 (mat*mat)");
        CHECK(mb && mb->type && mb->type->kind == WGSL_TYPE_MAT,
              "mb: mat (mat+mat)");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec3f(1.0);\n"
            "  let b = vec3f(2.0);\n"
            "  let c = a < b;\n"
            "}\n");
        CHECK(p.ok, "vec compare: tc_ok");
        WGSLSymbol *c = sym_for(&p, "c");
        CHECK(c && c->type && c->type->kind == WGSL_TYPE_VEC &&
              c->type->width == 3 && c->type->ref == p.types.t_bool,
              "c: vec3<bool>");
        done(&p);
    }

    /* User function call return type. */
    {
        P p; run(&p,
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n"
            "fn caller() {\n"
            "  let r = add(1, 2);\n"
            "}\n");
        CHECK(p.ok, "user-fn call: tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32, "r: i32 (fn ret type)");
        done(&p);
    }

    /* Function name as value -> error. */
    {
        P p; run(&p,
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n"
            "fn caller() {\n"
            "  let f = add;\n"
            "}\n");
        CHECK(!p.ok, "fn-as-value: must error");
        CHECK(has_error_with(&p.diag, "is not a value"),
              "fn-as-value diag");
        done(&p);
    }

    /* Index into ref'd vec preserves ref-ness. */
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v = vec3f(0.0);\n"
            "  let s = v.x;\n"            /* length-1 swizzle on ref → ref scalar */
            "}\n");
        CHECK(p.ok, "var v swizzle: tc_ok");
        done(&p);
    }

    fprintf(stderr, "%s  test_check_composite  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
