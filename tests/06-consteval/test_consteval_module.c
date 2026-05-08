/**
 * Phase 6 Iter B — module + fn-scope const declarations + const_assert.
 *
 * Covers:
 *   - module-scope `const NAME = expr;` evaluates and caches the value
 *   - typed module const: `const N: i32 = 5;` (target-driven materialise)
 *   - typed const range guard: `const N: u32 = -1;` errors
 *   - module const referenced in a later const: `const B = A * 2;`
 *   - `const_assert` literal pass/fail
 *   - `const_assert` referencing a module const
 *   - `const_assert` with non-bool condition errors
 *   - `const_assert` referencing a non-const ident errors ("not a constant")
 *   - fn-scope `const X = …;` bound for use inside the same fn body
 *   - fn-scope const referencing a module const
 *   - `override` decl walked without exploding (Iter B no-op)
 */
#include "internal/consteval.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
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
    int                parse_ok;
    int                resolve_ok;
    int                ce_ok;
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
    p->ce_ok = wgsl_consteval(&p->ast, &p->src, &p->arena,
                              &p->diag, &p->types, &p->res, &p->cev);
}

static void done(P *p) {
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

/* Find the bound value for a module-scope decl whose declared name == `want`. */
static const WGSLValue *value_of(const P *p, const char *want) {
    for (size_t i = 0; i < p->res.module.count; i++) {
        WGSLSymbol *s = p->res.module.items[i];
        if (s->name_len == strlen(want) &&
            memcmp(s->name, want, s->name_len) == 0)
        {
            return wgsl_consteval_value_of(&p->cev, s);
        }
    }
    return NULL;
}

/* Find the bound value for a const declared inside the (only) fn body
 * by walking the resolver's all_decls. */
static const WGSLValue *fn_value_of(const P *p, const char *want) {
    for (size_t i = 0; i < p->res.all_decl_count; i++) {
        WGSLSymbol *s = p->res.all_decls[i];
        if (s->name_len == strlen(want) &&
            memcmp(s->name, want, s->name_len) == 0)
        {
            const WGSLValue *v = wgsl_consteval_value_of(&p->cev, s);
            if (v) return v;
        }
    }
    return NULL;
}

int main(void) {
    wgsl_utf8_init();

    /* ── Module-scope const: untyped ───────────────────────────── */
    {
        P p; run(&p, "const N = 4;\n");
        CHECK(p.ce_ok, "const N = 4: ce_ok");
        const WGSLValue *v = value_of(&p, "N");
        CHECK(v && v->kind == WGSL_VAL_INT, "N: kind INT");
        CHECK(v && v->u.i == 4,             "N: value 4");
        /* Untyped → default concretisation: AbstractInt → i32 */
        CHECK(v && v->type == p.types.t_i32, "N: i32 after concretize");
        done(&p);
    }

    /* ── Module-scope typed const ─────────────────────────────── */
    {
        P p; run(&p, "const N: u32 = 256;\n");
        CHECK(p.ce_ok, "typed const u32: ce_ok");
        const WGSLValue *v = value_of(&p, "N");
        CHECK(v && v->u.i == 256 && v->type == p.types.t_u32,
              "N: 256u32");
        done(&p);
    }

    /* ── Typed const, abstract literal materialised to f32 ───── */
    {
        P p; run(&p, "const PI: f32 = 314;\n");   /* 314 → 314.0f */
        CHECK(p.ce_ok, "typed const f32: ce_ok");
        const WGSLValue *v = value_of(&p, "PI");
        CHECK(v && v->kind == WGSL_VAL_FLOAT && v->type == p.types.t_f32,
              "PI: float f32");
        CHECK(v && v->u.f == 314.0, "PI: value 314.0");
        done(&p);
    }

    /* ── Typed const range error ──────────────────────────────── */
    {
        P p; run(&p, "const N: u32 = -1;\n");
        CHECK(!p.ce_ok, "u32 = -1: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "u32 = -1: out-of-range diag");
        done(&p);
    }

    /* ── Module const referenced by a later module const ─────── */
    {
        P p; run(&p,
            "const A = 4;\n"
            "const B = A * 2 + 1;\n");
        CHECK(p.ce_ok, "B = A * 2 + 1: ce_ok");
        const WGSLValue *vb = value_of(&p, "B");
        CHECK(vb && vb->u.i == 9, "B = 9");
        done(&p);
    }

    /* ── const_assert literal pass ────────────────────────────── */
    {
        P p; run(&p, "const_assert true;\n");
        CHECK(p.ce_ok, "const_assert true: ce_ok");
        done(&p);
    }
    {
        P p; run(&p, "const_assert 1 + 2 == 3;\n");
        CHECK(p.ce_ok, "const_assert 1+2==3: ce_ok");
        done(&p);
    }

    /* ── const_assert literal fail ────────────────────────────── */
    {
        P p; run(&p, "const_assert 1 == 2;\n");
        CHECK(!p.ce_ok, "const_assert 1==2: must error");
        CHECK(has_error_with(&p.diag, "const_assert failed"),
              "const_assert: failure diag");
        done(&p);
    }

    /* ── const_assert referencing a module const ─────────────── */
    {
        P p; run(&p,
            "const WG: u32 = 256u;\n"
            "const_assert WG == 256u;\n");
        CHECK(p.ce_ok, "const_assert WG==256u: ce_ok");
        done(&p);
    }

    /* ── const_assert referencing a non-const (fn param) errors ─ */
    {
        P p; run(&p,
            "fn f(x: i32) {\n"
            "  const_assert x == 0;\n"
            "}\n");
        CHECK(!p.ce_ok, "fn-param ref in const_assert: must error");
        CHECK(has_error_with(&p.diag, "is not a constant"),
              "fn-param ref: 'is not a constant' diag");
        done(&p);
    }

    /* ── const_assert with non-bool condition errors ─────────── */
    {
        P p; run(&p, "const_assert 5;\n");
        CHECK(!p.ce_ok, "const_assert 5: must error");
        CHECK(has_error_with(&p.diag, "must evaluate to bool"),
              "non-bool const_assert: diag");
        done(&p);
    }

    /* ── Fn-scope const + const_assert ────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  const X = 3;\n"
            "  const Y = X * 2;\n"
            "  const_assert Y == 6;\n"
            "}\n");
        CHECK(p.ce_ok, "fn-scope const: ce_ok");
        const WGSLValue *vy = fn_value_of(&p, "Y");
        CHECK(vy && vy->u.i == 6, "fn Y == 6");
        done(&p);
    }

    /* ── Fn-scope const referencing a module const ───────────── */
    {
        P p; run(&p,
            "const N: u32 = 4u;\n"
            "fn f() {\n"
            "  const M = N + 1u;\n"
            "  const_assert M == 5u;\n"
            "}\n");
        CHECK(p.ce_ok, "fn-scope referencing module const: ce_ok");
        done(&p);
    }

    /* ── Override decl: walked without erroring (Iter B no-op) ─ */
    {
        P p; run(&p,
            "override SCALE: f32 = 1.0;\n"
            "const_assert true;\n");
        CHECK(p.ce_ok, "override + const_assert: ce_ok");
        const WGSLValue *vs = value_of(&p, "SCALE");
        CHECK(vs == NULL,
              "override: no cached value in v1 (deferred to pipeline-time)");
        done(&p);
    }

    /* ── Module const without init must already have failed at parse;
     *    verify the const-evaluator gracefully handles it. ─── */
    {
        P p; run(&p, "const N;\n");
        /* The parser will already have emitted a diag.  We just check
         * that consteval doesn't crash. */
        (void)p.ce_ok;
        done(&p);
    }

    /* ── Mixing types in a typed const ────────────────────────── */
    {
        P p; run(&p, "const N: i32 = 1u + 1u;\n");
        /* Unsigned + unsigned can't materialise to i32 (no implicit
         * concrete-to-concrete conversion). */
        CHECK(!p.ce_ok, "i32 = u32 expr: must error");
        done(&p);
    }

    fprintf(stderr, "%s  test_consteval_module  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
