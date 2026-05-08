/**
 * Phase 6 Iter A — const-expression evaluator: scalar folding.
 *
 * Covers:
 *   - integer literal parse: decimal, hex, suffixes (`i`, `u`, none)
 *   - float literal parse: decimal, hex float, suffixes (`f`, `h`, none)
 *   - bool literal: `true` / `false`
 *   - unary: + - ! ~ on appropriate operand kinds
 *   - binary arith: + - * / % on int / float
 *   - binary compare: == != < <= > >= producing bool
 *   - binary logical: && || on bool
 *   - binary bitwise: & | ^ << >> on int (and & | ^ on bool)
 *   - paren grouping
 *   - implicit conversions: AbstractInt + AbstractFloat → AbstractFloat;
 *     Abstract + i32 → i32; Abstract + u32 → u32
 *   - errors: u32 unary minus; division/modulo by zero; concrete-type
 *     mismatch (i32 + u32); literal range overflow
 *   - materialise: default tower (Abstract* → i32/f32) + target-driven
 */
#include "internal/consteval.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/utf8.h"

#include <math.h>
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

/* ── Per-test scaffold ──────────────────────────────────────────── */

typedef struct {
    WGSLArena          arena;
    WGSLSource         src;
    WGSLDiagBag        diag;
    WGSLLexResult      lex;
    WGSLAst            ast;
    WGSLTypeStore      types;
    WGSLConstEvaluator cev;
    char               wrapped[512];
} TestCtx;

/* Wrap `<expr>` in a fn-scope `let` so we can parse a single expression
 * via the regular parser path; then return the init expression. */
static WGSLNode *parse_expr(TestCtx *t, const char *expr) {
    memset(t, 0, sizeof *t);
    snprintf(t->wrapped, sizeof t->wrapped,
             "fn _t() { let _v = %s; }\n", expr);
    wgsl_arena_init(&t->arena);
    wgsl_diag_init(&t->diag);
    wgsl_source_init(&t->src, t->wrapped, strlen(t->wrapped));
    if (!wgsl_tokenize(&t->src, &t->arena, &t->diag, &t->lex)) return NULL;
    if (!wgsl_parse(&t->lex, &t->src, &t->arena, &t->diag, &t->ast)) return NULL;
    wgsl_types_init(&t->types, &t->arena);
    wgsl_consteval_init(&t->cev, &t->arena, &t->diag, &t->types, &t->src, NULL);

    /* AST shape:
     *   tu → fn → compound → let (DECL_LET)
     *   let.children = [type? (none here), init]
     * payload[1] = has_type (0 here). */
    if (!t->ast.root || t->ast.root->child_count == 0) return NULL;
    WGSLNode *fn = t->ast.root->children[0];
    if (!fn || fn->child_count == 0) return NULL;
    WGSLNode *body = fn->children[fn->child_count - 1];   /* last child = body */
    if (!body || body->child_count == 0) return NULL;
    WGSLNode *let = body->children[0];
    if (!let || let->child_count == 0) return NULL;
    /* DECL_LET has [type?][init].  has_type bit lives at payload[1]. */
    int has_type = (int)(let->payload[1] & 1u);
    return let->children[has_type ? 1 : 0];
}

static void cleanup(TestCtx *t) {
    wgsl_consteval_destroy(&t->cev);
    wgsl_types_destroy(&t->types);
    wgsl_diag_destroy(&t->diag);
    wgsl_source_destroy(&t->src);
    wgsl_arena_destroy(&t->arena);
}

static int eval_to(TestCtx *t, const char *expr, WGSLValue *out) {
    WGSLNode *n = parse_expr(t, expr);
    if (!n) {
        fprintf(stderr, "    parse failed for: %s\n", expr);
        return 0;
    }
    return wgsl_consteval_expr(&t->cev, n, out);
}

/* ── Convenience checkers ───────────────────────────────────────── */

static void check_int(const char *expr, int64_t want, WGSLTypeKind want_t) {
    TestCtx t; WGSLValue v;
    if (!eval_to(&t, expr, &v)) {
        fprintf(stderr, "FAIL  '%s' did not eval\n", expr); fail += 1;
        cleanup(&t); return;
    }
    if (v.kind != WGSL_VAL_INT || v.u.i != want ||
        (want_t != WGSL_TYPE_INVALID && v.type->kind != (uint16_t)want_t))
    {
        fprintf(stderr, "FAIL  '%s' = (kind=%u, type=%u, i=%lld), want (INT, %u, %lld)\n",
                expr, v.kind, v.type ? v.type->kind : 0, (long long)v.u.i,
                (unsigned)want_t, (long long)want);
        fail += 1;
    }
    cleanup(&t);
}

static void check_float(const char *expr, double want, WGSLTypeKind want_t) {
    TestCtx t; WGSLValue v;
    if (!eval_to(&t, expr, &v)) {
        fprintf(stderr, "FAIL  '%s' did not eval\n", expr); fail += 1;
        cleanup(&t); return;
    }
    if (v.kind != WGSL_VAL_FLOAT ||
        (want_t != WGSL_TYPE_INVALID && v.type->kind != (uint16_t)want_t) ||
        fabs(v.u.f - want) > 1e-9)
    {
        fprintf(stderr, "FAIL  '%s' = (kind=%u, type=%u, f=%g), want (FLOAT, %u, %g)\n",
                expr, v.kind, v.type ? v.type->kind : 0, v.u.f,
                (unsigned)want_t, want);
        fail += 1;
    }
    cleanup(&t);
}

static void check_bool(const char *expr, int want) {
    TestCtx t; WGSLValue v;
    if (!eval_to(&t, expr, &v)) {
        fprintf(stderr, "FAIL  '%s' did not eval\n", expr); fail += 1;
        cleanup(&t); return;
    }
    if (v.kind != WGSL_VAL_BOOL || (int)v.u.b != !!want) {
        fprintf(stderr, "FAIL  '%s' = (kind=%u, b=%d), want (BOOL, %d)\n",
                expr, v.kind, (int)v.u.b, want);
        fail += 1;
    }
    cleanup(&t);
}

static void check_eval_fails(const char *expr, const char *want_substr) {
    TestCtx t; WGSLValue v;
    int ok = eval_to(&t, expr, &v);
    if (ok) {
        fprintf(stderr,
                "FAIL  '%s' should have errored, got (kind=%u, i=%lld f=%g)\n",
                expr, v.kind, (long long)v.u.i, v.u.f);
        fail += 1;
    } else {
        int matched = 0;
        for (size_t i = 0; i < t.diag.count; i++) {
            if (strstr(t.diag.items[i].message, want_substr)) {
                matched = 1; break;
            }
        }
        if (!matched) {
            fprintf(stderr,
                "FAIL  '%s' errored but no diagnostic with '%s'\n",
                expr, want_substr);
            for (size_t i = 0; i < t.diag.count; i++) {
                fprintf(stderr, "      diag[%zu]: %s\n",
                        i, t.diag.items[i].message);
            }
            fail += 1;
        }
    }
    cleanup(&t);
}

int main(void) {
    wgsl_utf8_init();

    /* ── Integer literals ──────────────────────────────────────── */
    check_int("0",        0,    WGSL_TYPE_ABSTRACT_INT);
    check_int("42",       42,   WGSL_TYPE_ABSTRACT_INT);
    check_int("42i",      42,   WGSL_TYPE_I32);
    check_int("42u",      42,   WGSL_TYPE_U32);
    check_int("0xff",     255,  WGSL_TYPE_ABSTRACT_INT);
    check_int("0xFFu",    255,  WGSL_TYPE_U32);
    check_int("0X10",     16,   WGSL_TYPE_ABSTRACT_INT);

    /* ── Float literals ────────────────────────────────────────── */
    check_float("1.0",    1.0,   WGSL_TYPE_ABSTRACT_FLOAT);
    check_float("1.5f",   1.5,   WGSL_TYPE_F32);
    check_float("2.0h",   2.0,   WGSL_TYPE_F16);
    check_float("1e3",    1000., WGSL_TYPE_ABSTRACT_FLOAT);
    check_float("1.5e2f", 150.0, WGSL_TYPE_F32);
    check_float("0x1p4",  16.0,  WGSL_TYPE_ABSTRACT_FLOAT);

    /* ── Bool literals ─────────────────────────────────────────── */
    check_bool("true",  1);
    check_bool("false", 0);

    /* ── Unary ─────────────────────────────────────────────────── */
    check_int  ("-5",    -5,   WGSL_TYPE_ABSTRACT_INT);
    check_int  ("-5i",   -5,   WGSL_TYPE_I32);
    check_float("-1.5",  -1.5, WGSL_TYPE_ABSTRACT_FLOAT);
    check_int  ("~0",    ~0,   WGSL_TYPE_ABSTRACT_INT);
    check_int  ("~0u",   (int64_t)(uint32_t)~0u, WGSL_TYPE_U32);
    check_bool ("!true", 0);
    check_bool ("!false",1);

    /* ── Binary arithmetic ─────────────────────────────────────── */
    check_int  ("1 + 2",          3,   WGSL_TYPE_ABSTRACT_INT);
    check_int  ("1i + 2i",        3,   WGSL_TYPE_I32);
    check_int  ("4u - 1u",        3,   WGSL_TYPE_U32);
    check_int  ("3 * 4",          12,  WGSL_TYPE_ABSTRACT_INT);
    check_int  ("10 / 3",         3,   WGSL_TYPE_ABSTRACT_INT);
    check_int  ("10 % 3",         1,   WGSL_TYPE_ABSTRACT_INT);
    check_int  ("(-10) / 3",      -3,  WGSL_TYPE_ABSTRACT_INT);
    check_int  ("(-10) % 3",      -1,  WGSL_TYPE_ABSTRACT_INT);
    check_float("1.5 + 2.5",      4.0, WGSL_TYPE_ABSTRACT_FLOAT);
    check_float("3.0 * 2.0f",     6.0, WGSL_TYPE_F32);
    check_float("9.0 / 4.0",      2.25,WGSL_TYPE_ABSTRACT_FLOAT);

    /* ── Implicit promotion ────────────────────────────────────── */
    /* AbstractInt + AbstractFloat → AbstractFloat */
    check_float("1 + 2.5",        3.5, WGSL_TYPE_ABSTRACT_FLOAT);
    /* AbstractInt + i32 → i32 */
    check_int  ("1 + 2i",         3,   WGSL_TYPE_I32);
    /* AbstractInt + u32 → u32 (positive only) */
    check_int  ("1 + 2u",         3,   WGSL_TYPE_U32);
    /* AbstractInt + f32 → f32 */
    check_float("1 + 2.5f",       3.5, WGSL_TYPE_F32);

    /* ── Comparisons ───────────────────────────────────────────── */
    check_bool("1 == 1",        1);
    check_bool("1 != 1",        0);
    check_bool("1 < 2",         1);
    check_bool("2 <= 2",        1);
    check_bool("3 > 2",         1);
    check_bool("3 >= 3",        1);
    check_bool("1.5 < 2.0",     1);
    check_bool("true == true",  1);
    check_bool("true == false", 0);

    /* ── Logical / bitwise ─────────────────────────────────────── */
    check_bool("true && false", 0);
    check_bool("true || false", 1);
    check_int ("0xff & 0x0f",   0x0f, WGSL_TYPE_ABSTRACT_INT);
    check_int ("0xf0 | 0x0f",   0xff, WGSL_TYPE_ABSTRACT_INT);
    check_int ("0xff ^ 0x0f",   0xf0, WGSL_TYPE_ABSTRACT_INT);
    check_int ("1 << 4",        16,   WGSL_TYPE_ABSTRACT_INT);
    check_int ("16 >> 2",       4,    WGSL_TYPE_ABSTRACT_INT);

    /* ── Paren grouping ────────────────────────────────────────── */
    check_int  ("(1 + 2) * 3",      9,    WGSL_TYPE_ABSTRACT_INT);
    check_float("(1.0 + 2.0) / 2.0",1.5,  WGSL_TYPE_ABSTRACT_FLOAT);

    /* ── Errors ────────────────────────────────────────────────── */
    check_eval_fails("-1u",         "u32");
    check_eval_fails("1 / 0",       "division by zero");
    check_eval_fails("1 % 0",       "modulo by zero");
    check_eval_fails("1.0 / 0.0",   "division by zero");
    check_eval_fails("1i + 1u",     "no implicit conversion");
    /* §6.2.3: -(-INT32_MIN) is not representable in i32.  Construct
     * the value via `1i << 31` (which truncates to INT32_MIN at i32
     * width), then attempt the negation. */
    check_eval_fails("-(1i << 31)", "not representable in i32");

    /* ── Materialise: default tower ────────────────────────────── */
    {
        TestCtx t; WGSLValue v;
        if (eval_to(&t, "1 + 2", &v)) {
            CHECK(v.type == t.types.t_abstract_int, "abstract int before materialise");
            CHECK(wgsl_consteval_materialize(&t.cev, &v, NULL, NULL),
                  "default materialise OK");
            CHECK(v.type == t.types.t_i32, "AbstractInt → i32");
            CHECK(v.u.i == 3,              "value preserved");
        } else { fail += 1; }
        cleanup(&t);
    }
    {
        TestCtx t; WGSLValue v;
        if (eval_to(&t, "1.5 + 2.5", &v)) {
            CHECK(v.type == t.types.t_abstract_float, "abstract float before materialise");
            CHECK(wgsl_consteval_materialize(&t.cev, &v, NULL, NULL),
                  "default materialise OK");
            CHECK(v.type == t.types.t_f32, "AbstractFloat → f32");
            CHECK(v.u.f == 4.0,            "value preserved");
        } else { fail += 1; }
        cleanup(&t);
    }
    /* AbstractInt → u32 positive OK. */
    {
        TestCtx t; WGSLValue v;
        if (eval_to(&t, "5", &v)) {
            CHECK(wgsl_consteval_materialize(&t.cev, &v, t.types.t_u32, NULL),
                  "AbstractInt → u32 (positive)");
            CHECK(v.type == t.types.t_u32, "type → u32");
            CHECK(v.u.i == 5,              "value preserved");
        } else { fail += 1; }
        cleanup(&t);
    }
    /* AbstractInt negative → u32 must FAIL. */
    {
        TestCtx t; WGSLValue v;
        if (eval_to(&t, "-3", &v)) {
            int ok = wgsl_consteval_materialize(&t.cev, &v, t.types.t_u32, NULL);
            CHECK(!ok, "AbstractInt negative → u32 must fail");
        } else { fail += 1; }
        cleanup(&t);
    }
    /* AbstractInt → f32 OK (numeric tower). */
    {
        TestCtx t; WGSLValue v;
        if (eval_to(&t, "7", &v)) {
            CHECK(wgsl_consteval_materialize(&t.cev, &v, t.types.t_f32, NULL),
                  "AbstractInt → f32");
            CHECK(v.type == t.types.t_f32, "type → f32");
            CHECK(v.kind == WGSL_VAL_FLOAT, "kind switched to FLOAT");
            CHECK(v.u.f == 7.0,            "int → float value");
        } else { fail += 1; }
        cleanup(&t);
    }

    fprintf(stderr, "%s  test_consteval  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
