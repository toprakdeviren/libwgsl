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
