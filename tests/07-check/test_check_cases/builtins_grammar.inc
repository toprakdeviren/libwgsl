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

