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

