/**
 * Phase 8 Iter A — validator: cycle detection family.
 *
 * Covers:
 *   - call-graph: self-recursion, mutual recursion → "recursion" diag
 *   - acyclic call graph (caller→callee→leaf) → ok
 *   - struct value cycle: self-field, mutual fields, array-of-struct
 *     wrapping → "struct cycle" diag
 *   - acyclic struct nesting → ok
 *   - alias cycle: A=B / B=A → "alias cycle" diag
 *   - acyclic alias chain → ok
 *   - const init cycle: A = A + 1 → "const cycle" diag
 *   - override init cycle → "override cycle" diag
 */
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/consteval.h"
#include "internal/check.h"
#include "internal/validate.h"
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
    WGSLValidator      val;
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
    int val_ok   = wgsl_validate(&p->ast, &p->src, &p->arena,
                                 &p->diag, &p->types, &p->res, &p->cev,
                                 &p->tc, &p->val);
    p->ok = lex_ok && parse_ok && res_ok && ce_ok && tc_ok && val_ok;
}

static void done(P *p) {
    wgsl_validator_destroy(&p->val);
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

int main(void) {
    wgsl_utf8_init();

    /* ── Call-graph cycles ─────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() { f(); }\n");
        CHECK(!p.ok, "self-recursion: must error");
        CHECK(has_error_with(&p.diag, "recursion"),
              "self-recursion: 'recursion' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn a() { b(); }\n"
            "fn b() { a(); }\n");
        CHECK(!p.ok, "mutual recursion: must error");
        CHECK(has_error_with(&p.diag, "recursion"),
              "mutual recursion: 'recursion' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn leaf() -> i32 { return 1; }\n"
            "fn middle() -> i32 { return leaf(); }\n"
            "fn top() -> i32 { return middle() + leaf(); }\n");
        CHECK(p.ok, "acyclic call graph: ok");
        done(&p);
    }

    /* ── Struct cycles ─────────────────────────────────────────── */
    {
        P p; run(&p,
            "struct S { x: S }\n");
        CHECK(!p.ok, "self-recursive struct: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "self-recursive struct: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct A { b: B }\n"
            "struct B { a: A }\n");
        CHECK(!p.ok, "mutually-recursive structs: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "mutual structs: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct S { arr: array<S, 4> }\n");
        CHECK(!p.ok, "array<S,N> field cycle: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "array-of-struct cycle: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct Inner { v: vec3<f32> }\n"
            "struct Outer { i: Inner, w: f32 }\n");
        CHECK(p.ok, "acyclic struct nesting: ok");
        done(&p);
    }

    /* ── Alias cycles ──────────────────────────────────────────── */
    {
        P p; run(&p,
            "alias A = B;\n"
            "alias B = A;\n");
        CHECK(!p.ok, "alias cycle: must error");
        CHECK(has_error_with(&p.diag, "alias cycle"),
              "alias cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "alias A = u32;\n"
            "alias B = A;\n"
            "alias C = B;\n");
        CHECK(p.ok, "alias chain: ok");
        done(&p);
    }

    /* ── Const cycles ──────────────────────────────────────────── */
    {
        P p; run(&p,
            "const A: i32 = A + 1;\n");
        CHECK(!p.ok, "self-cycle const: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "const cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const A: i32 = B + 1;\n"
            "const B: i32 = A * 2;\n");
        CHECK(!p.ok, "mutual const cycle: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "mutual const cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const A: i32 = 2;\n"
            "const B: i32 = A + 1;\n"
            "const C: i32 = A * B;\n");
        CHECK(p.ok, "acyclic const chain: ok");
        done(&p);
    }

    /* ── Override cycles ───────────────────────────────────────── */
    {
        P p; run(&p,
            "override A: f32 = B + 1.0;\n"
            "override B: f32 = A * 2.0;\n");
        CHECK(!p.ok, "mutual override cycle: must error");
        CHECK(has_error_with(&p.diag, "override cycle"),
              "override cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override A: f32 = 1.0;\n"
            "override B: f32 = A + 1.0;\n");
        CHECK(p.ok, "acyclic override chain: ok");
        done(&p);
    }

    /* ── Sanity: typical real shader passes ───────────────────── */
    {
        P p; run(&p,
            "const WG: u32 = 256u;\n"
            "fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {\n"
            "  return gid.x + gid.y * nwg.x * WG;\n"
            "}\n"
            "@compute @workgroup_size(256, 1, 1)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n"
            "        @builtin(num_workgroups) nwg: vec3<u32>) {\n"
            "  let tid = flat_id(gid, nwg);\n"
            "  _ = tid;\n"
            "}\n");
        CHECK(p.ok, "real-shader-shape: ok");
        done(&p);
    }

    /* ── Attribute conformance (§12) ──────────────────────────── */

    /* Stage attributes — only on fn decls. */
    {
        P p; run(&p,
            "@vertex\n"
            "var x: i32 = 1;\n");
        CHECK(!p.ok, "@vertex on var: must error");
        CHECK(has_error_with(&p.diag, "is not allowed"),
              "@vertex-on-var diag");
        done(&p);
    }
    /* @workgroup_size — argc check. */
    {
        P p; run(&p,
            "@compute @workgroup_size()\n"
            "fn main() {}\n");
        CHECK(!p.ok, "@workgroup_size(): must error (zero args)");
        CHECK(has_error_with(&p.diag, "expects"),
              "argc diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@compute @workgroup_size(1, 2, 3, 4)\n"
            "fn main() {}\n");
        CHECK(!p.ok, "@workgroup_size(4): must error (too many)");
        done(&p);
    }
    /* @binding / @group on fn → error. */
    {
        P p; run(&p,
            "@binding(0)\n"
            "fn f() {}\n");
        CHECK(!p.ok, "@binding on fn: must error");
        done(&p);
    }
    /* Unknown attribute. */
    {
        P p; run(&p,
            "@foobar\n"
            "fn f() {}\n");
        CHECK(!p.ok, "@foobar: unknown attr error");
        CHECK(has_error_with(&p.diag, "unknown attribute"),
              "unknown-attr diag");
        done(&p);
    }
    /* Valid: @group / @binding on module var. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read> X: array<f32>;\n");
        CHECK(p.ok, "@group + @binding on var: ok");
        done(&p);
    }
    /* Valid: @builtin on fn param. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {}\n");
        CHECK(p.ok, "@compute + @workgroup_size + @builtin: ok");
        done(&p);
    }
    /* @align on struct member is OK. */
    {
        P p; run(&p,
            "struct S {\n"
            "  @align(16) v: vec3<f32>,\n"
            "}\n");
        CHECK(p.ok, "@align on struct member: ok");
        done(&p);
    }

    /* ── Behavior analysis (§9.7) ─────────────────────────────── */

    /* Non-void fn missing return on a path → error. */
    {
        P p; run(&p,
            "fn f(x: i32) -> i32 {\n"
            "  if (x > 0) { return 1; }\n"
            "}\n");
        CHECK(!p.ok, "missing return on else: must error");
        CHECK(has_error_with(&p.diag, "does not return on all paths"),
              "all-paths diag");
        done(&p);
    }
    /* Non-void fn with full coverage via else. */
    {
        P p; run(&p,
            "fn f(x: i32) -> i32 {\n"
            "  if (x > 0) { return 1; }\n"
            "  else       { return 0; }\n"
            "}\n");
        CHECK(p.ok, "if/else both return: ok");
        done(&p);
    }
    /* Non-void fn with terminal return at body end. */
    {
        P p; run(&p,
            "fn f() -> i32 {\n"
            "  let x = 5;\n"
            "  return x;\n"
            "}\n");
        CHECK(p.ok, "terminal return: ok");
        done(&p);
    }
    /* Non-void fn with for-loop (loops always fall through to Next). */
    {
        P p; run(&p,
            "fn f() -> i32 {\n"
            "  for (var i = 0; i < 4; i = i + 1) { return i; }\n"
            "}\n");
        CHECK(!p.ok, "for-loop not all paths: must error");
        done(&p);
    }
    /* void fn without return → ok. */
    {
        P p; run(&p,
            "fn f() { let x = 1; }\n");
        CHECK(p.ok, "void fn no return: ok");
        done(&p);
    }
    /* Switch with default returning everywhere. */
    {
        P p; run(&p,
            "fn f(x: i32) -> i32 {\n"
            "  switch (x) {\n"
            "    case 0: { return 0; }\n"
            "    default: { return 1; }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "switch full coverage: ok");
        done(&p);
    }

    /* ── Entry-point shape (§13) ──────────────────────────────── */
    /* @compute without @workgroup_size → error. */
    {
        P p; run(&p,
            "@compute\n"
            "fn main() {}\n");
        CHECK(!p.ok, "@compute w/o @workgroup_size: must error");
        CHECK(has_error_with(&p.diag, "requires a `@workgroup_size`"),
              "@compute missing-wgs diag");
        done(&p);
    }
    /* Multi-stage on one fn → error. */
    {
        P p; run(&p,
            "@vertex @fragment\n"
            "fn dual() {}\n");
        CHECK(!p.ok, "multi-stage attr: must error");
        CHECK(has_error_with(&p.diag, "more than one shader-stage"),
              "multi-stage diag");
        done(&p);
    }
    /* @workgroup_size on non-compute fn → error. */
    {
        P p; run(&p,
            "@vertex @workgroup_size(64)\n"
            "fn vmain() {}\n");
        CHECK(!p.ok, "@workgroup_size on @vertex: must error");
        done(&p);
    }
    {
        P p; run(&p,
            "@workgroup_size(64)\n"
            "fn helper() {}\n");
        CHECK(!p.ok, "@workgroup_size on plain fn: must error");
        CHECK(has_error_with(&p.diag, "only valid on `@compute`"),
              "wgs-on-non-compute diag");
        done(&p);
    }

    /* ── Address-space conformance (§14.3) ───────────────────── */
    /* var<uniform> without @group/@binding → error. */
    {
        P p; run(&p,
            "var<uniform> X: vec4<f32>;\n");
        CHECK(!p.ok, "var<uniform> w/o group+binding: must error");
        CHECK(has_error_with(&p.diag, "@group") ||
              has_error_with(&p.diag, "@binding"),
              "uniform missing attr diag");
        done(&p);
    }
    /* var<storage> without @group/@binding → error. */
    {
        P p; run(&p,
            "var<storage, read> X: array<f32>;\n");
        CHECK(!p.ok, "var<storage> w/o group+binding: must error");
        done(&p);
    }
    /* var<storage> with @group + @binding → ok. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read_write> Y: array<f32>;\n");
        CHECK(p.ok, "var<storage> w/ group+binding: ok");
        done(&p);
    }
    /* var<workgroup> doesn't need @group/@binding. */
    {
        P p; run(&p,
            "var<workgroup> sh: array<f32, 256>;\n");
        CHECK(p.ok, "var<workgroup>: ok without group+binding");
        done(&p);
    }
    /* var<private> doesn't need them either. */
    {
        P p; run(&p,
            "var<private> g: f32;\n");
        CHECK(p.ok, "var<private>: ok without group+binding");
        done(&p);
    }

    fprintf(stderr, "%s  test_validate  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
