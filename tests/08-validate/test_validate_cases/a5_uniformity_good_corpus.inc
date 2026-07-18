/* §A5 — known-good uniformity corpus.
 *
 * Every ACCEPT case below is a valid shader that Tint/Naga accept and that
 * must NOT raise `derivative_uniformity` / `subgroup_uniformity`.  The anchor
 * (deep uniform call chain) is the exact false-reject that the pre-A5
 * `user_function_result_is_uniform` `init_depth>=16 → return 0` cap produced
 * (chain length ≥ ~17 wrongly rejected `workgroupBarrier`); the call-graph
 * `returns_uniform` fixpoint fixes it with no depth limit.
 *
 * The REJECT controls at the end must STILL fire — so the corpus cannot be
 * satisfied by weakening the diagnostic. */

/* ── ACCEPT 1 (anchor): 20-deep uniform constant-return chain guards a
 * barrier.  Every fn returns a constant → the whole chain is uniform → the
 * guarding `if` is uniform → the barrier is in uniform control flow. */
{
    P p; run(&p,
        "fn f0() -> u32 { return 0u; }\n"
        "fn f1() -> u32 { return f0(); }\n"
        "fn f2() -> u32 { return f1(); }\n"
        "fn f3() -> u32 { return f2(); }\n"
        "fn f4() -> u32 { return f3(); }\n"
        "fn f5() -> u32 { return f4(); }\n"
        "fn f6() -> u32 { return f5(); }\n"
        "fn f7() -> u32 { return f6(); }\n"
        "fn f8() -> u32 { return f7(); }\n"
        "fn f9() -> u32 { return f8(); }\n"
        "fn f10() -> u32 { return f9(); }\n"
        "fn f11() -> u32 { return f10(); }\n"
        "fn f12() -> u32 { return f11(); }\n"
        "fn f13() -> u32 { return f12(); }\n"
        "fn f14() -> u32 { return f13(); }\n"
        "fn f15() -> u32 { return f14(); }\n"
        "fn f16() -> u32 { return f15(); }\n"
        "fn f17() -> u32 { return f16(); }\n"
        "fn f18() -> u32 { return f17(); }\n"
        "fn f19() -> u32 { return f18(); }\n"
        "fn f20() -> u32 { return f19(); }\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  if (f20() == 0u) { workgroupBarrier(); }\n"
        "}\n");
    CHECK(p.ok, "a5 deep-const-chain: shader is fully valid");
    CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 deep-const-chain(20): no false-reject of barrier");
    done(&p);
}

/* ── ACCEPT 2: deep chain returning a `var<uniform>` module value.  A
 * read-only uniform resource is uniform; a 12-deep helper chain preserves
 * that → barrier in uniform CF. */
{
    P p; run(&p,
        "@group(0) @binding(0) var<uniform> ubuf: u32;\n"
        "fn h0() -> u32 { return ubuf; }\n"
        "fn h1() -> u32 { return h0(); }\n"
        "fn h2() -> u32 { return h1(); }\n"
        "fn h3() -> u32 { return h2(); }\n"
        "fn h4() -> u32 { return h3(); }\n"
        "fn h5() -> u32 { return h4(); }\n"
        "fn h6() -> u32 { return h5(); }\n"
        "fn h7() -> u32 { return h6(); }\n"
        "fn h8() -> u32 { return h7(); }\n"
        "fn h9() -> u32 { return h8(); }\n"
        "fn h10() -> u32 { return h9(); }\n"
        "fn h11() -> u32 { return h10(); }\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  if (h11() == 0u) { workgroupBarrier(); }\n"
        "}\n");
    CHECK(p.ok, "a5 deep-uniform-var-chain: shader is fully valid");
    CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 deep-uniform-var-chain(12): no false-reject of barrier");
    done(&p);
}

/* ── ACCEPT 3: `workgroupUniformLoad` result is known-uniform → using it as
 * a branch condition keeps the guarded barrier in uniform CF. */
{
    P p; run(&p,
        "var<workgroup> wsh: u32;\n"
        "@compute @workgroup_size(64)\n"
        "fn main() {\n"
        "  let v = workgroupUniformLoad(&wsh);\n"
        "  if (v == 0u) { workgroupBarrier(); }\n"
        "}\n");
    CHECK(p.ok, "a5 workgroupUniformLoad-guard: shader is fully valid");
    CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 workgroupUniformLoad-guard: no false-reject of barrier");
    done(&p);
}

/* ── ACCEPT 4: uniform loop counter driving a helper call each iteration;
 * the barrier is a sibling of the (possibly divergent) inner `if`, guarded
 * only by the uniform counter `s`. */
{
    P p; run(&p,
        "fn dec(x: u32) -> u32 { return x >> 1u; }\n"
        "var<workgroup> sh: array<u32, 256>;\n"
        "@compute @workgroup_size(256)\n"
        "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
        "  for (var s: u32 = 128u; s > 0u; s = dec(s)) {\n"
        "    if (lid.x < s) { sh[lid.x] += sh[lid.x + s]; }\n"
        "    workgroupBarrier();\n"
        "  }\n"
        "}\n");
    CHECK(p.ok, "a5 uniform-loop-counter-helper: shader is fully valid");
    CHECK(!has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 uniform-loop-counter-helper: no false-reject of barrier");
    done(&p);
}

/* ── REJECT control 1: a genuinely divergent barrier (cond uses `lid.x`)
 * must STILL raise the diagnostic. */
{
    P p; run(&p,
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
        "  if (lid.x > 0u) { workgroupBarrier(); }\n"
        "}\n");
    CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 divergent-barrier control: diag still fires");
    done(&p);
}

/* ── REJECT control 2: a deep chain that FORWARDS a non-uniform parameter
 * (the entry passes `lid.x`) is non-uniform → guarded barrier must STILL
 * raise the diagnostic.  Proves the fixpoint propagates non-uniformity. */
{
    P p; run(&p,
        "fn g0(x: u32) -> u32 { return x; }\n"
        "fn g1(x: u32) -> u32 { return g0(x); }\n"
        "fn g2(x: u32) -> u32 { return g1(x); }\n"
        "fn g3(x: u32) -> u32 { return g2(x); }\n"
        "fn g4(x: u32) -> u32 { return g3(x); }\n"
        "fn g5(x: u32) -> u32 { return g4(x); }\n"
        "fn g6(x: u32) -> u32 { return g5(x); }\n"
        "fn g7(x: u32) -> u32 { return g6(x); }\n"
        "fn g8(x: u32) -> u32 { return g7(x); }\n"
        "fn g9(x: u32) -> u32 { return g8(x); }\n"
        "fn g10(x: u32) -> u32 { return g9(x); }\n"
        "fn g11(x: u32) -> u32 { return g10(x); }\n"
        "fn g12(x: u32) -> u32 { return g11(x); }\n"
        "fn g13(x: u32) -> u32 { return g12(x); }\n"
        "fn g14(x: u32) -> u32 { return g13(x); }\n"
        "fn g15(x: u32) -> u32 { return g14(x); }\n"
        "fn g16(x: u32) -> u32 { return g15(x); }\n"
        "fn g17(x: u32) -> u32 { return g16(x); }\n"
        "fn g18(x: u32) -> u32 { return g17(x); }\n"
        "fn g19(x: u32) -> u32 { return g18(x); }\n"
        "@compute @workgroup_size(64)\n"
        "fn cs(@builtin(local_invocation_id) lid: vec3<u32>) {\n"
        "  if (g19(lid.x) == 0u) { workgroupBarrier(); }\n"
        "}\n");
    CHECK(has_error_with(&p.diag, "must be called in uniform control flow"),
          "a5 non-uniform-param-chain control: diag still fires");
    done(&p);
}
