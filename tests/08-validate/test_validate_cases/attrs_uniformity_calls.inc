
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
