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
#include "internal/ast.h"

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

    /* ── §5 Module-`var` cycles ───────────────────────────────── */
    {
        P p; run(&p,
            "var<private> a: i32 = b;\n"
            "var<private> b: i32 = a;\n");
        CHECK(!p.ok, "mutual module-var cycle: must error");
        CHECK(has_error_with(&p.diag, "module var cycle"),
              "mutual module-var cycle: diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> a: i32 = a;\n");
        CHECK(!p.ok, "self module-var cycle: must error");
        CHECK(has_error_with(&p.diag, "module var cycle"),
              "self module-var cycle: diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> a: i32 = 0;\n"
            "var<private> b: i32 = 1;\n");
        CHECK(p.ok, "acyclic module-var inits: ok");
        done(&p);
    }

    /* ── §5 Cross-kind value cycles ──────────────────────────── */
    {
        /* const A = B; override B: i32 = A; */
        P p; run(&p,
            "const A = B;\n"
            "override B: i32 = A;\n");
        CHECK(!p.ok, "const <-> override: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "const <-> override: cross-kind diag");
        done(&p);
    }
    {
        /* var<private> a = B; const B = a; */
        P p; run(&p,
            "var<private> a: i32 = B;\n"
            "const B: i32 = a;\n");
        CHECK(!p.ok, "module-var <-> const: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "module-var <-> const: cross-kind diag");
        done(&p);
    }
    {
        /* Three-way: const -> override -> var -> const */
        P p; run(&p,
            "const A: i32 = B;\n"
            "override B: i32 = i32(c);\n"
            "var<private> c: i32 = A;\n");
        CHECK(!p.ok, "const -> override -> var: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "three-way mixed cycle: cross-kind diag");
        done(&p);
    }
    {
        /* Single-kind cycle still uses the per-kind message and does
         * NOT fire the cross-kind diag. */
        P p; run(&p,
            "const A: i32 = B + 1;\n"
            "const B: i32 = A * 2;\n");
        CHECK(!p.ok, "const-only cycle: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "single-kind cycle keeps per-kind label");
        CHECK(!has_error_with(&p.diag, "cross-kind value cycle"),
              "single-kind cycle does not fire cross-kind");
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
    {
        P p; run(&p, "fn f(@builtin(position) x: vec4<f32>) {}\n");
        CHECK(!p.ok, "@builtin on ordinary fn param: must error");
        CHECK(has_error_with(&p.diag, "entry-point IO attribute"),
              "ordinary fn param IO attr diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() -> @location(0) f32 { return 0.0; }\n");
        CHECK(!p.ok, "@location on ordinary fn return: must error");
        CHECK(has_error_with(&p.diag, "entry-point IO attribute"),
              "ordinary fn return IO attr diag");
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
    /* §12.11 — @location must be non-negative. */
    {
        P p; run(&p,
            "@vertex\n"
            "fn vs(@location(-1) a: f32) -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@location(-1): must error");
        CHECK(has_error_with(&p.diag, "@location value must be a non-negative integer"),
              "@location negative diag");
        done(&p);
    }
    /* §12.11 — @location must be an integer const-expression. */
    {
        P p; run(&p,
            "@vertex\n"
            "fn vs(@location(1.0) a: f32) -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@location(1.0): must error");
        CHECK(has_error_with(&p.diag,
            "@location value must be a const-expression of type i32 or u32"),
              "@location float diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@vertex\n"
            "fn vs(@location(vec2(1, 1)) a: f32) -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@location(vec2): must error");
        CHECK(has_error_with(&p.diag,
            "@location value must be a const-expression of type i32 or u32"),
              "@location vector diag");
        done(&p);
    }
    /* §12.2 — @binding must be non-negative (regression). */
    {
        P p; run(&p,
            "@group(0) @binding(-1) var<storage, read> X: array<f32>;\n");
        CHECK(!p.ok, "@binding(-1): must error");
        CHECK(has_error_with(&p.diag, "@binding value must be a non-negative integer"),
              "@binding negative diag");
        done(&p);
    }
    /* §13.3.1.2 — @location IO must be numeric scalar / vector. */
    {
        P p; run(&p,
            "@vertex\n"
            "fn vs(@location(0) a: bool) -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@location bool: must error");
        CHECK(has_error_with(&p.diag, "@location IO must have a numeric scalar or vector type"),
              "@location bool diag");
        done(&p);
    }
    /* §12.6 — @diagnostic may repeat with different rules. */
    {
        P p; run(&p,
            "@diagnostic(off, derivative_uniformity)\n"
            "@diagnostic(off, subgroup_uniformity)\n"
            "fn f() {}\n");
        CHECK(p.ok, "two @diagnostic distinct rules: ok");
        done(&p);
    }
    /* §12.6 — same rule repeated must reject. */
    {
        P p; run(&p,
            "@diagnostic(off, derivative_uniformity)\n"
            "@diagnostic(error, derivative_uniformity)\n"
            "fn f() {}\n");
        CHECK(!p.ok, "two @diagnostic same rule: must error");
        CHECK(has_error_with(&p.diag, "must differ on the rule name"),
              "@diagnostic dup-rule diag");
        done(&p);
    }
    /* §12.13 — @size(1.5) must be rejected for type, not value. */
    {
        P p; run(&p,
            "struct S { @size(1.5) x: i32 }\n");
        CHECK(!p.ok, "@size(1.5): must error");
        CHECK(has_error_with(&p.diag,
            "@size value must be a const-expression of type i32 or u32"),
              "@size float diag");
        done(&p);
    }
    /* §12.1 — @align(1.5) must be rejected for type. */
    {
        P p; run(&p,
            "struct S { @align(1.5) x: i32 }\n");
        CHECK(!p.ok, "@align(1.5): must error");
        CHECK(has_error_with(&p.diag,
            "@align value must be a const-expression of type i32 or u32"),
              "@align float diag");
        done(&p);
    }
    /* §12.1 — @align(16) (i32-typed) is OK. */
    {
        P p; run(&p,
            "struct S { @align(16) x: vec3<f32> }\n");
        CHECK(p.ok, "@align(16): ok");
        done(&p);
    }
    /* §12.13 — @size on a runtime-sized array member must reject. */
    {
        P p; run(&p,
            "struct S { @size(64) tail: array<f32> }\n");
        CHECK(!p.ok, "@size on runtime-sized array: must error");
        CHECK(has_error_with(&p.diag,
            "@size requires a member type with a generation-fixed footprint"),
              "@size CFF diag");
        done(&p);
    }
    /* §12.3 — @blend_src(2): out of range. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "@fragment\n"
            "fn fs() -> @location(0) @blend_src(2) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@blend_src(2): must error");
        CHECK(has_error_with(&p.diag, "@blend_src value must be 0 or 1"),
              "@blend_src 0/1 diag");
        done(&p);
    }
    /* §12.3 — @blend_src without @location must reject. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "@fragment\n"
            "fn fs() -> @blend_src(0) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@blend_src no @location: must error");
        CHECK(has_error_with(&p.diag, "@blend_src may only appear on a declaration that also has @location"),
              "@blend_src no-@location diag");
        done(&p);
    }
    /* §12.3 — @blend_src on a non-fragment output. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "@vertex\n"
            "fn vs() -> @location(0) @blend_src(0) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@blend_src on vertex: must error");
        CHECK(has_error_with(&p.diag, "@blend_src is only valid in a fragment entry point"),
              "@blend_src vertex diag");
        done(&p);
    }
    /* §12.3 — @blend_src must be on a structure member, not a direct return. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "@fragment\n"
            "fn fs() -> @location(0) @blend_src(0) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "@blend_src direct return: must error");
        CHECK(has_error_with(&p.diag, "only valid on a structure member"),
              "@blend_src direct-return diag");
        done(&p);
    }
    /* §13.3.1.3 — dual-source struct shape: two @blend_src outputs OK. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "struct Out {\n"
            "  @location(0) @blend_src(0) a: vec4<f32>,\n"
            "  @location(0) @blend_src(1) b: vec4<f32>,\n"
            "}\n"
            "@fragment\n"
            "fn fs() -> Out { return Out(vec4<f32>(0.0), vec4<f32>(0.0)); }\n");
        CHECK(p.ok, "dual-source struct happy path: ok");
        done(&p);
    }
    /* §13.3.1.3 — only one @blend_src side present must reject. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "struct Out {\n"
            "  @location(0) @blend_src(0) a: vec4<f32>,\n"
            "  @location(1) b: vec4<f32>,\n"
            "}\n"
            "@fragment\n"
            "fn fs() -> Out { return Out(vec4<f32>(0.0), vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "dual-source missing index 1: must error");
        CHECK(has_error_with(&p.diag,
            "exactly one @blend_src(0) and one @blend_src(1) member"),
              "dual-source missing-side diag");
        done(&p);
    }
    /* §13.3.1.3 — only two @location members are allowed in a
     * dual-source output struct.  Extra builtin members are OK. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "struct Out {\n"
            "  @location(0) @blend_src(0) a: vec4<f32>,\n"
            "  @location(0) @blend_src(1) b: vec4<f32>,\n"
            "  @location(1) c: vec4<f32>,\n"
            "}\n"
            "@fragment\n"
            "fn fs() -> @location(0) vec4<f32> { return vec4<f32>(); }\n");
        CHECK(!p.ok, "dual-source extra location member: must error");
        CHECK(has_error_with(&p.diag, "exactly two @location members"),
              "dual-source location-count diag");
        done(&p);
    }
    /* §6.4.2 — var<uniform> store type must be host-shareable. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<uniform> X: bool;\n");
        CHECK(!p.ok, "var<uniform> bool: must error");
        CHECK(has_error_with(&p.diag, "store type must be host-shareable"),
              "uniform bool diag");
        done(&p);
    }
    /* §6.4.2 — struct of bool member rejected in storage. */
    {
        P p; run(&p,
            "struct S { f: bool }\n"
            "@group(0) @binding(0) var<storage, read> X: S;\n");
        CHECK(!p.ok, "var<storage> struct-of-bool: must error");
        CHECK(has_error_with(&p.diag, "store type must be host-shareable"),
              "storage struct-of-bool diag");
        done(&p);
    }
    /* §15.6.3 — subgroup op in divergent CF emits `subgroup_uniformity`
     * filterable diag.  (`val_filterable` doesn't flip `had_error`, so
     * we check the message directly rather than `p.ok`.) */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  if (lid.x > 0u) {\n"
            "    let v = subgroupBroadcastFirst(lid.x);\n"
            "  }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "subgroup / quad operation 'subgroupBroadcastFirst' must be called in uniform control flow"),
              "subgroup_uniformity diag");
        done(&p);
    }
    /* §15.6.4 — quad op in divergent CF → subgroup_uniformity. */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  if (f > 0.0) {\n"
            "    let v = quadSwapX(f);\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "subgroup / quad operation 'quadSwapX' must be called in uniform control flow"),
              "quad subgroup_uniformity diag");
        done(&p);
    }
    /* Filter via `@diagnostic(off, subgroup_uniformity)` should silence. */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "@diagnostic(off, subgroup_uniformity)\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  if (lid.x > 0u) {\n"
            "    let v = subgroupBroadcastFirst(lid.x);\n"
            "  }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "subgroup op silenced by @diagnostic(off)");
        done(&p);
    }
    /* §17.12 / §17.13 — subgroup and quad builtins are not valid in
     * vertex-stage code, including through helper functions reachable
     * from a vertex entry point. */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn helper() { _ = subgroupShuffle(0, 0); }\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  helper();\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "subgroup op reachable from vertex: must error");
        CHECK(has_error_with(&p.diag, "not valid in vertex stage code"),
              "subgroup vertex-stage diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn helper() { _ = quadSwapX(1.0); }\n"
            "@fragment\n"
            "fn fs() { helper(); }\n");
        CHECK(p.ok, "quad op reachable from fragment: ok");
        done(&p);
    }
    /* §17.7 / §17.6 derivative sampling builtins are fragment-only,
     * including through helper functions reachable from non-fragment
     * entry points. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn helper() { _ = textureSample(t, s, vec2<f32>(0.0)); }\n"
            "@compute @workgroup_size(1)\n"
            "fn cs() { helper(); }\n");
        CHECK(!p.ok, "textureSample reachable from compute: must error");
        CHECK(has_error_with(&p.diag, "only valid in code reachable from a fragment entry point"),
              "textureSample compute-stage diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn helper(x: f32) -> f32 { return dpdx(x); }\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(helper(1.0));\n"
            "}\n");
        CHECK(!p.ok, "dpdx reachable from vertex: must error");
        CHECK(has_error_with(&p.diag, "only valid in code reachable from a fragment entry point"),
              "dpdx vertex-stage diag");
        done(&p);
    }
    /* §17.11 — barriers are compute-only, including through helpers. */
    {
        P p; run(&p,
            "fn helper() { workgroupBarrier(); }\n"
            "@fragment\n"
            "fn fs() { helper(); }\n");
        CHECK(!p.ok, "workgroupBarrier reachable from fragment: must error");
        CHECK(has_error_with(&p.diag, "only valid in code reachable from a compute entry point"),
              "barrier fragment-stage diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> value = true;\n"
            "const_assert(value);\n");
        CHECK(!p.ok, "const_assert runtime var: must error");
        CHECK(has_error_with(&p.diag, "is not a constant"),
              "const_assert const-expression diag");
        done(&p);
    }
    /* §15.2.5 — function-scope `var` initialized from a uniform expr
     * (literal / const / override) IS uniform.  Reduction patterns like
     *   for (var s = 128u; s > 0u; s >>= 1u) {
     *     if (lid.x < s) { ... }
     *     workgroupBarrier();
     *   }
     * must NOT trigger derivative_uniformity — the loop cond depends
     * only on `s`, a uniform loop counter; the barrier is sibling to
     * (not inside) the `if (lid.x < s)`. */
    {
        P p; run(&p,
            "var<workgroup> sh: array<u32, 256>;\n"
            "@compute @workgroup_size(256)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  for (var s: u32 = 128u; s > 0u; s >>= 1u) {\n"
            "    if (lid.x < s) { sh[lid.x] += sh[lid.x + s]; }\n"
            "    workgroupBarrier();\n"
            "  }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "reduction barrier: no false-positive uniformity diag");
        done(&p);
    }
    /* §15.2.5 — but a barrier inside a genuinely non-uniform `if`
     * (cond uses `lid.x`) still fires the diag. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  if (lid.x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "real divergent barrier: diag still fires");
        done(&p);
    }
    /* §15.2.5 — function-scope `var` initialized from a param IS
     * non-uniform: `var x = lid.x; if (x > 0u) { workgroupBarrier(); }`
     * is still a real spec violation. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = lid.x;\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "var init from param: tracked as non-uniform");
        done(&p);
    }
    /* §15.2.5 — post-init assignment lattice: a var initialized from
     * a uniform value becomes non-uniform after assignment from `lid.x`. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = 0u;\n"
            "  x = lid.x;\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "var assignment from param: tracked as non-uniform");
        done(&p);
    }
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs() {\n"
            "  var x: u32 = 0u;\n"
            "  x = 1u;\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "var assignment from uniform expr: remains uniform");
        done(&p);
    }
    /* §15.2.5 — partial assignments join with the previous value of the
     * originating variable; writing one component does not prove the whole
     * composite uniform. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var pair: vec2<u32> = vec2<u32>(lid.x, lid.y);\n"
            "  pair.x = 0u;\n"
            "  if (pair.y > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "partial assignment preserves non-uniform composite state");
        done(&p);
    }
    /* §15.2.5 — non-Next branches do not poison the state reaching the
     * next statement. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = 0u;\n"
            "  if (lid.x > 0u) { x = lid.x; return; }\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "returning branch does not reach post-if state");
        done(&p);
    }
    /* §15.2.5 / §15.2.10 — loop fixpoint carries assignments out of the
     * loop body into the following statement. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = 0u;\n"
            "  var i: u32 = 0u;\n"
            "  while (i < 1u) {\n"
            "    x = lid.x;\n"
            "    i = i + 1u;\n"
            "  }\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "while assignment reaches post-loop state");
        done(&p);
    }
    /* §15.2.10 — a non-uniform conditional interruption inside a loop
     * makes collective ops in that loop non-uniform, even if the op appears
     * before the interruption syntactically. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  loop {\n"
            "    let x = dpdx(f);\n"
            "    if (f > 0.0) { break; }\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag,
            "collective operation 'dpdx' must be called in uniform control flow"),
              "loop op before conditional break is divergent");
        done(&p);
    }
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  loop {\n"
            "    if (f > 0.0) { break; }\n"
            "  }\n"
            "  let x = dpdx(f);\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!has_error_with(&p.diag,
            "collective operation 'dpdx' must be called in uniform control flow"),
              "conditional break reconverges after loop");
        done(&p);
    }
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  for (; true;) {\n"
            "    if (f > 0.0) { return vec4<f32>(0.0); }\n"
            "    break;\n"
            "  }\n"
            "  let x = dpdx(f);\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag,
            "collective operation 'dpdx' must be called in uniform control flow"),
              "conditional return keeps post-for control divergent");
        done(&p);
    }
    /* §15.2.5 / §15.2.10 — switch cases join assignment state at the
     * statement exit. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = 0u;\n"
            "  switch lid.x {\n"
            "    case 0u: { x = 1u; }\n"
            "    default: {}\n"
            "  }\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "switch non-uniform selector joins divergent case state");
        done(&p);
    }
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs() {\n"
            "  var x: u32 = 0u;\n"
            "  switch 0u {\n"
            "    case 0u: { x = 1u; }\n"
            "    default: {}\n"
            "  }\n"
            "  if (x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "switch uniform selector preserves uniform case state");
        done(&p);
    }
    /* §15.2.8 — short-circuit RHS is evaluated in control flow derived
     * from the left operand. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  if ((f > 0.0) && (dpdx(f) > 0.0)) {}\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "collective operation 'dpdx' must be called in uniform control flow"),
              "short-circuit RHS inherits non-uniform lhs control flow");
        done(&p);
    }
    /* §15.2.8 — spec-listed uniform builtin inputs are uniform leaves. */
    {
        P p; run(&p,
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(workgroup_id) wid: vec3<u32>) {\n"
            "  if (wid.x > 0u) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
              "workgroup_id builtin input is uniform");
        done(&p);
    }
    {
        P p; run(&p,
            "var<workgroup> wg: u32;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            "  if (gid.x > 0u) { _ = workgroupUniformLoad(&wg); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag,
                  "collective operation 'workgroupUniformLoad' must be called in uniform control flow"),
              "workgroupUniformLoad divergent-CF diag");
        done(&p);
    }
    {
        P p; run(&p,
            "requires readonly_and_readwrite_storage_textures;\n"
            "@group(0) @binding(0) var tex : texture_storage_2d<rgba8unorm, read_write>;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs() {\n"
            "  if (textureLoad(tex, vec2<i32>(0)).x == 0.0) { workgroupBarrier(); }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
              "read_write storage texture load result is non-uniform");
        done(&p);
    }
    /* §15.2.8 / §17.12 — subgroup shuffle delta/mask parameters must be
     * uniform even when the call site itself is in uniform control flow. */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  let v = subgroupShuffleUp(lid.x, lid.x);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "argument 'delta' to 'subgroupShuffleUp' must be uniform"),
              "subgroupShuffleUp delta uniformity diag");
        done(&p);
    }
    /* §7.2.2 / §12.14 — `@workgroup_size` arg must be const- or
     * override-expression; passing a `var<private>` is rejected. */
    {
        P p; run(&p,
            "var<private> sz: u32 = 64u;\n"
            "@compute @workgroup_size(sz)\n"
            "fn cs() {}\n");
        CHECK(!p.ok, "@workgroup_size with var arg: must error");
        CHECK(has_error_with(&p.diag, "@workgroup_size argument must be a const- or override-expression"),
              "@workgroup_size override-expr diag");
        done(&p);
    }
    /* §7.2.2 — @workgroup_size with override symbol: ok. */
    {
        P p; run(&p,
            "override SZ: u32 = 64u;\n"
            "@compute @workgroup_size(SZ)\n"
            "fn cs() {}\n");
        CHECK(p.ok, "@workgroup_size with override: ok");
        done(&p);
    }
    /* §11.3 / §12.5 — `@const` is a notational marker for builtin
     * fn signatures; applying it to a user-defined fn is invalid.
     * Regression: an earlier audit (gaps.md) flagged this case as
     * causing an infinite-loop hang in the validator pipeline.  The
     * placement reject (`check_attr` in src/validate.c) closes both
     * the spec violation and the hang; this test pins the behaviour. */
    {
        P p; run(&p, "@const fn f(x: i32) -> i32 { return x + 1; }\n");
        CHECK(!p.ok, "@const fn: must error");
        CHECK(has_error_with(&p.diag,
            "attribute '@const' cannot be applied to user-defined functions"),
              "@const fn diag");
        done(&p);
    }
    /* §11.3 — recursive `@const` user fn must not hang either. */
    {
        P p; run(&p,
            "@const fn f(x: i32) -> i32 { return f(x); }\n");
        /* Recursion check fires + @const placement reject; both diags
         * present.  No hang. */
        CHECK(!p.ok, "@const recursive fn: must error");
        CHECK(has_error_with(&p.diag,
            "attribute '@const' cannot be applied to user-defined functions"),
              "@const recursive: placement diag");
        done(&p);
    }
    /* §15.2.6 — `for` body inherits divergence from non-uniform header. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  for (var i = 0u; i < u32(f); i = i + 1u) {\n"
            "    let d = dpdx(f);\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "collective operation 'dpdx' must be called in uniform control flow"),
              "for-divergence dpdx diag");
        done(&p);
    }
    /* §15.2.6 — `while` body inherits divergence. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  var i = 0u;\n"
            "  while (i < u32(f)) {\n"
            "    let d = dpdx(f);\n"
            "    i = i + 1u;\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "collective operation 'dpdx' must be called in uniform control flow"),
              "while-divergence dpdx diag");
        done(&p);
    }
    /* §15.2.6 — `switch` cases inherit divergence from non-uniform sel. */
    {
        P p; run(&p,
            "enable subgroups;\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  switch lid.x {\n"
            "    case 0u: { let v = subgroupBroadcastFirst(lid.x); }\n"
            "    default: {}\n"
            "  }\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "subgroup / quad operation 'subgroupBroadcastFirst' must be called in uniform control flow"),
              "switch-divergence subgroup diag");
        done(&p);
    }
    /* §15.2.3 / §15.2.7 — call to a user-fn that transitively uses
     * dpdx must itself be in uniform CF. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn helper(f: f32) -> f32 { return dpdx(f); }\n"
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  if (f > 0.0) {\n"
            "    let v = helper(f);\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "call to 'helper' must be in uniform control flow"),
              "user-fn propagation diag");
        done(&p);
    }
    /* §15.2 — uniform-AS reads are uniform across invocations, so a
     * call-site driven by `var<uniform>` must NOT trip divergence. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<uniform> threshold: f32;\n"
            "@fragment\n"
            "fn fs(@location(0) f: f32) -> @location(0) vec4<f32> {\n"
            "  if (threshold > 0.0) {\n"
            "    let d = dpdx(f);\n"
            "  }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "uniform control flow"),
              "uniform var → no divergence diag");
        done(&p);
    }
    /* §15.2.4 — pointer params desugar to call-site actuals for
     * uniformity.  A helper branching on `*p` is uniform when every call
     * passes a uniform root. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<uniform> threshold: u32;\n"
            "fn helper(p: ptr<uniform, u32>) {\n"
            "  if (*p > 0u) { workgroupBarrier(); }\n"
            "}\n"
            "@compute @workgroup_size(64)\n"
            "fn cs() {\n"
            "  helper(&threshold);\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "uniform control flow"),
              "ptr param from uniform root: no divergence diag");
        done(&p);
    }
    /* §15.2.4 — forwarding a uniform pointer through another helper
     * preserves the uniformity bit. */
    {
        P p; run(&p,
            "fn helper(p: ptr<function, u32>) {\n"
            "  if (*p > 0u) { workgroupBarrier(); }\n"
            "}\n"
            "fn wrapper(p: ptr<function, u32>) { helper(p); }\n"
            "@compute @workgroup_size(64)\n"
            "fn cs() {\n"
            "  var x: u32 = 1u;\n"
            "  wrapper(&x);\n"
            "}\n");
        CHECK(!has_error_with(&p.diag, "uniform control flow"),
              "forwarded ptr param from uniform local: no divergence diag");
        done(&p);
    }
    /* §15.2.4 — a non-uniform actual marks the callee pointer param
     * non-uniform and the branch on `*p` still diagnoses. */
    {
        P p; run(&p,
            "fn helper(p: ptr<function, u32>) {\n"
            "  if (*p > 0u) { workgroupBarrier(); }\n"
            "}\n"
            "@compute @workgroup_size(64)\n"
            "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
            "  var x: u32 = lid.x;\n"
            "  helper(&x);\n"
            "}\n");
        CHECK(has_error_with(&p.diag, "uniform control flow"),
              "ptr param from non-uniform local: divergence diag");
        done(&p);
    }
    /* §9.4.3 / §9.7 — `loop { }` is obviously infinite (no break / return). */
    {
        P p; run(&p, "fn f() { loop { } }\n");
        CHECK(!p.ok, "loop { }: must error");
        CHECK(has_error_with(&p.diag, "loop has no exit edge"),
              "infinite-loop diag");
        done(&p);
    }
    /* §9.4.3 — `loop { discard; }` is also infinite (discard doesn't exit). */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs() -> @location(0) vec4<f32> {\n"
            "  loop { discard; }\n"
            "}\n");
        CHECK(!p.ok, "loop { discard; }: must error");
        CHECK(has_error_with(&p.diag, "loop has no exit edge"),
              "infinite-loop discard diag");
        done(&p);
    }
    /* §9.4.3 — `loop { break; }` is OK. */
    {
        P p; run(&p, "fn f() { loop { break; } }\n");
        CHECK(p.ok, "loop { break }: ok");
        done(&p);
    }
    /* §9.4.3 — `loop { continuing { break if true; } }` is OK (break_if exits). */
    {
        P p; run(&p, "fn f() { loop { continuing { break if true; } } }\n");
        CHECK(p.ok, "loop { continuing { break if true } }: ok");
        done(&p);
    }
    /* §9.4.3 — `loop { return; }` is OK (return is also an exit). */
    {
        P p; run(&p, "fn f() { loop { return; } }\n");
        CHECK(p.ok, "loop { return }: ok");
        done(&p);
    }
    /* §9.7 — `for (;;)` without a break/return has no exit edge. */
    {
        P p; run(&p, "fn f() { for (;;) { } }\n");
        CHECK(!p.ok, "for (;;) { }: must error");
        CHECK(has_error_with(&p.diag, "for loop has no exit edge"),
              "infinite-for diag");
        done(&p);
    }
    /* Unreachable invalid statements still have to be behavior-valid. */
    {
        P p; run(&p, "fn f() { return; loop { } }\n");
        CHECK(!p.ok, "return; loop { }: must error");
        CHECK(has_error_with(&p.diag, "loop has no exit edge"),
              "unreachable invalid loop diag");
        done(&p);
    }
    /* §9.4.4 — `while (true) { }` does NOT trigger the infinite-loop
     * gate in v1 (we don't const-eval the condition); while always
     * carries an implicit break edge. */
    {
        P p; run(&p, "fn f() { while (true) { } }\n");
        CHECK(p.ok, "while (true) { }: ok (v1 doesn't const-eval cond)");
        done(&p);
    }
    /* §9.4.11 — discard reachable from non-fragment entry must reject. */
    {
        P p; run(&p,
            "@compute @workgroup_size(1)\n"
            "fn cs() { discard; }\n");
        CHECK(!p.ok, "discard in compute: must error");
        CHECK(has_error_with(&p.diag, "'discard' is only valid in code reachable from a fragment entry point"),
              "discard compute diag");
        done(&p);
    }
    /* §9.4.11 — discard via call chain into vertex entry must reject. */
    {
        P p; run(&p,
            "fn helper() { discard; }\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> { helper(); return vec4<f32>(0.0); }\n");
        CHECK(!p.ok, "discard via helper from vertex: must error");
        CHECK(has_error_with(&p.diag, "'discard' is only valid in code reachable from a fragment entry point"),
              "discard vertex-via-helper diag");
        done(&p);
    }
    /* §9.4.11 — discard in fragment entry: ok. */
    {
        P p; run(&p,
            "@fragment\n"
            "fn fs() -> @location(0) vec4<f32> { discard; return vec4<f32>(0.0); }\n");
        CHECK(p.ok, "discard in fragment: ok");
        done(&p);
    }
    /* §14.3 / §14.5 — read_write storage WRITTEN from vertex stage rejects. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  buf[0] = 1u; return vec4<f32>(0.0);\n"
            "}\n");
        CHECK(!p.ok, "read_write storage write from vertex: must error");
        CHECK(has_error_with(&p.diag, "vertex stage cannot access read_write storage"),
              "vertex storage write diag");
        done(&p);
    }
    /* §14.3 — read_write storage is not accessible from vertex stage,
     * even when the expression is only a read. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  let x = buf[0];\n"
            "  return vec4<f32>(f32(x));\n"
            "}\n");
        CHECK(!p.ok, "read_write storage read from vertex: must error");
        CHECK(has_error_with(&p.diag, "vertex stage cannot access read_write storage"),
              "vertex storage read_write access diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var tex: texture_storage_2d<r32uint, write>;\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  _ = tex;\n"
            "  return vec4<f32>();\n"
            "}\n");
        CHECK(!p.ok, "write storage texture from vertex: must error");
        CHECK(has_error_with(&p.diag, "storage textures"),
              "vertex storage texture access diag");
        done(&p);
    }
    /* §17.8 — atomic builtins are not valid in vertex-stage code, even
     * when the operation is a read-only atomicLoad. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read_write> a: atomic<u32>;\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  let x = atomicLoad(&a);\n"
            "  return vec4<f32>(f32(x));\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad storage from vertex: must error");
        CHECK(has_error_with(&p.diag, "not valid in vertex stage code"),
              "atomicLoad-from-vertex diag");
        done(&p);
    }
    /* §17.8 — atomicAdd from vertex is likewise rejected by stage. */
    {
        P p; run(&p,
            "@group(0) @binding(0) var<storage, read_write> a: atomic<u32>;\n"
            "@vertex\n"
            "fn vs() -> @builtin(position) vec4<f32> {\n"
            "  let x = atomicAdd(&a, 1u);\n"
            "  return vec4<f32>(f32(x));\n"
            "}\n");
        CHECK(!p.ok, "atomicAdd storage from vertex: must error");
        CHECK(has_error_with(&p.diag, "not valid in vertex stage code"),
              "atomicAdd-from-vertex diag");
        done(&p);
    }

    /* §11.4.1 — same-root pointer args reject when a callee may write. */
    {
        P p; run(&p,
            "fn pair(p1: ptr<function, i32>, p2: ptr<function, i32>) {\n"
            "  *p1 = *p2;\n"
            "}\n"
            "fn wrap(p1: ptr<function, i32>, p2: ptr<function, i32>) {\n"
            "  pair(p1, p2);\n"
            "}\n"
            "fn f() {\n"
            "  var a: i32 = 0;\n"
            "  wrap(&a, &a);\n"
            "}\n");
        CHECK(!p.ok, "alias: transitive same-root write/read must error");
        CHECK(has_error_with(&p.diag, "alias analysis"),
              "alias same-root diag");
        done(&p);
    }
    /* §11.4.1 — same-root args are OK when both pointer params are read-only. */
    {
        P p; run(&p,
            "fn read_pair(p1: ptr<function, i32>, p2: ptr<function, i32>) -> i32 {\n"
            "  return *p1 + *p2;\n"
            "}\n"
            "fn f() {\n"
            "  var a: i32 = 1;\n"
            "  let b = read_pair(&a, &a);\n"
            "  _ = b;\n"
            "}\n");
        CHECK(p.ok, "alias: read-only same-root args are ok");
        done(&p);
    }
    /* §11.4.1.1 — let-initialized pointer values preserve root identifier. */
    {
        P p; run(&p,
            "fn pair(p1: ptr<function, i32>, p2: ptr<function, i32>) {\n"
            "  *p1 = *p2;\n"
            "}\n"
            "fn f() {\n"
            "  var a: i32 = 0;\n"
            "  let pa = &a;\n"
            "  pair(pa, &a);\n"
            "}\n");
        CHECK(!p.ok, "alias: let pointer root must be tracked");
        CHECK(has_error_with(&p.diag, "alias analysis"),
              "alias let-root diag");
        done(&p);
    }
    /* §11.4.1.2 — module var aliased by ptr arg rejects on read/write overlap. */
    {
        P p; run(&p,
            "var<private> x: i32 = 0;\n"
            "fn touches_x(p: ptr<private, i32>) {\n"
            "  x = *p;\n"
            "}\n"
            "fn f() {\n"
            "  touches_x(&x);\n"
            "}\n");
        CHECK(!p.ok, "alias: module var write/read overlap must error");
        CHECK(has_error_with(&p.diag, "alias analysis"),
              "alias module conflict diag");
        done(&p);
    }
    /* §11.4.1.2 — module read plus pointer read is not an aliasing error. */
    {
        P p; run(&p,
            "var<private> x: i32 = 0;\n"
            "fn reads_x(p: ptr<private, i32>) -> i32 {\n"
            "  return x + *p;\n"
            "}\n"
            "fn f() {\n"
            "  let y = reads_x(&x);\n"
            "  _ = y;\n"
            "}\n");
        CHECK(p.ok, "alias: module read + pointer read is ok");
        done(&p);
    }

    /* §12.1 / §14.4 — @align value must be a multiple of RequiredAlignOf. */
    {
        /* vec3<f32> has AlignOf 16; @align(8) is < 16 and not a multiple → reject. */
        P p; run(&p,
            "struct S { @align(8) v: vec3<f32> }\n");
        CHECK(!p.ok, "@align(8) on vec3<f32>: must error");
        CHECK(has_error_with(&p.diag, "must be a multiple of RequiredAlignOf"),
              "@align multiplicity diag");
        done(&p);
    }
    /* @align(32) on vec3<f32> (align 16): 32 is a multiple → OK. */
    {
        P p; run(&p,
            "struct S { @align(32) v: vec3<f32> }\n");
        CHECK(p.ok, "@align(32) on vec3<f32>: ok");
        done(&p);
    }
    /* §14.4.5 — uniform AS struct member offset rule.  Inner struct
     * with explicit @align(8) breaks the roundUp(16,…) rule. */
    {
        P p; run(&p,
            "struct Inner { @align(8) a: f32 }\n"
            "struct Outer { x: f32, y: Inner }\n"
            "@group(0) @binding(0) var<uniform> u: Outer;\n");
        /* layout_struct for Outer in uniform: offset(x)=0, then y must
         * be at multiple of roundUp(16, AlignOf(Inner=8)) = 16; if
         * placed at 8 it would break. The validator's offset walk
         * already aligns to 16 here, so no error.  This case verifies
         * the happy path doesn't false-positive. */
        CHECK(p.ok, "uniform Outer { f32, Inner }: ok (auto-padded)");
        done(&p);
    }
    /* §6.4.2 — happy path: i32 / vec3<f32> / struct of those. */
    {
        P p; run(&p,
            "struct S { a: vec3<f32>, b: i32 }\n"
            "@group(0) @binding(0) var<uniform> X: S;\n");
        CHECK(p.ok, "var<uniform> host-shareable struct: ok");
        done(&p);
    }
    /* §12.11 / §13.3.1.2 — @location forbidden on compute entry-point IO. */
    {
        P p; run(&p,
            "@compute @workgroup_size(1)\n"
            "fn cs(@location(0) x: f32) {}\n");
        CHECK(!p.ok, "@location on compute param: must error");
        CHECK(has_error_with(&p.diag,
            "@compute entry point must not have user-defined IO"),
              "@location compute diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct In { @location(0) x: f32 }\n"
            "@compute @workgroup_size(1)\n"
            "fn cs(in: In) {}\n");
        CHECK(!p.ok, "@location on compute struct param: must error");
        CHECK(has_error_with(&p.diag,
            "@compute entry point must not have user-defined IO"),
              "@location compute struct diag");
        done(&p);
    }
    /* §13.3.1.3 — type mismatch between sides must reject. */
    {
        P p; run(&p,
            "enable dual_source_blending;\n"
            "struct Out {\n"
            "  @location(0) @blend_src(0) a: vec4<f32>,\n"
            "  @location(0) @blend_src(1) b: vec2<f32>,\n"
            "}\n"
            "@fragment\n"
            "fn fs() -> Out { return Out(vec4<f32>(0.0), vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "dual-source type mismatch: must error");
        CHECK(has_error_with(&p.diag,
            "dual-source blending members must have the same type"),
              "dual-source type-mismatch diag");
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

    /* §15.2.9 — per-node uniformity tags.  Walk the AST after validation
     * and confirm:
     *   - a fn param ident in `if (param)` cond gets WGSL_FLAG_NONUNIFORM
     *   - a builtin call inside the divergent body gets WGSL_FLAG_DIVERGENT_CF
     *
     * We don't enforce annotation on every expr (the validator only walks
     * the ones it cares about); this just pins that the bits ARE set
     * where the validator visits them. */
    {
        P p; run(&p,
            "@fragment fn fs(@location(0) v: f32) -> @location(0) vec4<f32> {\n"
            "  if (v > 0.0) { let d = dpdx(v); }\n"
            "  return vec4<f32>(0.0);\n"
            "}\n");
        /* The shader has a divergent-CF derivative call, so validation
         * emits a `derivative_uniformity` diag.  That's fine for this
         * test — we're checking that the annotation bits were planted
         * during the walk, not the validity verdict. */
        int saw_nonuniform_node = 0;
        int saw_divergent_node  = 0;
        /* DFS over AST. */
        WGSLNode *stack[64];
        int sp = 0;
        stack[sp++] = p.ast.root;
        while (sp > 0) {
            WGSLNode *n = stack[--sp];
            if (!n) continue;
            if (n->flags & WGSL_FLAG_NONUNIFORM)   saw_nonuniform_node = 1;
            if (n->flags & WGSL_FLAG_DIVERGENT_CF) saw_divergent_node  = 1;
            for (uint32_t i = 0; i < n->child_count && sp < 64; i++) {
                stack[sp++] = n->children[i];
            }
        }
        CHECK(saw_nonuniform_node, "annotation: some node got NONUNIFORM bit");
        CHECK(saw_divergent_node,  "annotation: some node got DIVERGENT_CF bit");
        done(&p);
    }

    fprintf(stderr, "%s  test_validate  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
