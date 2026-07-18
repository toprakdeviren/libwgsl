    /* ── §9 — statement-level conformance ─────────────────────── */
    {
        /* §9.2.1 — RHS type vs LHS store type. */
        P p; run(&p, "fn f() { var x: i32 = 0; x = 1.5; }\n");
        CHECK(!p.ok, "x = 1.5 with i32 store type: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "assign type-mismatch diag");
        done(&p);
    }
    {
        /* §9.2.2 — phony RHS must be constructible / ptr / texture / sampler. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: array<f32>;\n"
            "fn f() { _ = a; }\n");
        CHECK(!p.ok, "_ = runtime-array: must error");
        CHECK(has_error_with(&p.diag, "constructible"),
              "phony-rhs diag");
        done(&p);
    }
    {
        /* Phony positives — scalar / pointer arg / texture all legal. */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "fn f() { _ = 1; _ = t; }\n");
        CHECK(p.ok, "_ = scalar / texture: tc_ok");
        done(&p);
    }
    {
        /* §9.3 — `++` requires concrete integer scalar. */
        P p; run(&p, "fn f() { var a: f32 = 0.0; a++; }\n");
        CHECK(!p.ok, "a++ on f32: must error");
        CHECK(has_error_with(&p.diag, "integer"),
              "inc-on-float diag");
        done(&p);
    }
    {
        /* §9.4.2 — switch selector must be concrete integer scalar. */
        P p; run(&p,
            "fn f() { var x: f32 = 0.0; switch x { default: {} } }\n");
        CHECK(!p.ok, "switch f32 selector: must error");
        CHECK(has_error_with(&p.diag, "selector must be a concrete integer"),
              "switch f32 diag");
        done(&p);
    }
    {
        /* §9.4.2 — duplicate default reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { default: {} default: {} } }\n");
        CHECK(!p.ok, "switch two defaults: must error");
        CHECK(has_error_with(&p.diag, "more than one default"),
              "two-defaults diag");
        done(&p);
    }
    {
        /* §9.4.2 — missing default reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0; switch x { case 0: {} } }\n");
        CHECK(!p.ok, "switch missing default: must error");
        CHECK(has_error_with(&p.diag, "exactly one default"),
              "missing-default diag");
        done(&p);
    }
    {
        /* §9.4.2 — duplicate case selectors reject. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0: {} case 0: {} default: {} } }\n");
        CHECK(!p.ok, "switch duplicate case: must error");
        CHECK(has_error_with(&p.diag, "duplicate case selector"),
              "duplicate-case diag");
        done(&p);
    }
    {
        /* §9.4.2 — concrete case selectors must have a common type. */
        P p; run(&p,
            "fn f() { switch 1 { case i32(1): {} case 2u: {} default: {} } }\n");
        CHECK(!p.ok, "switch mixed concrete case types: must error");
        CHECK(has_error_with(&p.diag, "earlier case selector type"),
              "mixed concrete case type diag");
        done(&p);
    }
    {
        /* Happy switch path: distinct cases + default. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0: {} case 1, 2: {} default: {} } }\n");
        CHECK(p.ok, "well-formed switch: tc_ok");
        done(&p);
    }
    {
        /* §9.4.2 — case selector must be a const-expression.  A `let`
         * is not a const-expression in WGSL. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  let k: i32 = 1;\n"
            "  switch x { case k: {} default: {} } }\n");
        CHECK(!p.ok, "non-const case selector: must error");
        CHECK(has_error_with(&p.diag, "case selector must be a const-expression"),
              "non-const case diag");
        done(&p);
    }
    {
        /* §9.4.2 — `const` selector is a const-expression: ok. */
        P p; run(&p,
            "const K: i32 = 1;\n"
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case K: {} default: {} } }\n");
        CHECK(p.ok, "const case selector: ok");
        done(&p);
    }
    {
        /* §9.4.9 — `return` inside continuing is rejected. */
        P p; run(&p, "fn f() { loop { continuing { return; } } }\n");
        CHECK(!p.ok, "return in continuing: must error");
        CHECK(has_error_with(&p.diag, "'return' is not allowed inside"),
              "return-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.9 — `continue` inside continuing is rejected. */
        P p; run(&p, "fn f() { loop { continuing { continue; } } }\n");
        CHECK(!p.ok, "continue in continuing: must error");
        CHECK(has_error_with(&p.diag, "'continue' is not allowed inside"),
              "continue-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.6 — `break` inside continuing (no inner loop/switch)
         * would exit the loop's continuing block. */
        P p; run(&p,
            "fn f() {\n"
            "  loop { continuing { if (true) { break; } } }\n"
            "}\n");
        CHECK(!p.ok, "break in continuing: must error");
        CHECK(has_error_with(&p.diag, "'break' is not allowed to exit"),
              "break-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.6 — `break` inside a switch inside continuing is OK
         * (it targets the switch, not the outer loop). */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      switch 1 { case 0: { break; } default: { break; } }\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "break in switch in continuing: ok");
        done(&p);
    }
    {
        /* §9.4.6 — `break_if` must be the final statement of a
         * continuing block.  Anything after it makes the parser emit
         * "expected '}'" rather than silently accepting. */
        P p; run(&p,
            "fn f() { loop { continuing { break if true; let x = 1; } } }\n");
        CHECK(!p.ok, "break_if not last in continuing: must error");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { loop { continuing { break if 1i; } } }\n");
        CHECK(!p.ok, "break_if non-bool condition: must error");
        CHECK(has_error_with(&p.diag, "'break if' condition must evaluate to bool"),
              "break_if condition type diag");
        done(&p);
    }
    {
        /* §7.3 — var<private> store type must be storable; texture
         * types are excluded. */
        P p; run(&p, "var<private> t: texture_2d<f32>;\n");
        CHECK(!p.ok, "var<private> texture_2d: must error");
        CHECK(has_error_with(&p.diag, "var<private> store type must be storable"),
              "var<private> texture diag");
        done(&p);
    }
    {
        /* §9.4.8 / §9.7 — `continue` inside an inner loop inside a
         * continuing block targets the inner loop, not the continuing.
         * Must NOT be flagged.  The inner loop has its own exit edge
         * via `if (true) { break; }`. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      loop {\n"
            "        if (true) { break; }\n"
            "        continue;\n"
            "      }\n"
            "      break if true;\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continue in inner loop inside continuing: ok");
        CHECK(!has_error_with(&p.diag, "'continue' is not allowed"),
              "no continue-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.2 — two `default` keywords in the same case-selector
         * list (`case 0, default, default, 2:`) is invalid. */
        P p; run(&p,
            "fn f() { var x: i32 = 0;\n"
            "  switch x { case 0, default, default, 2: { } }\n"
            "}\n");
        CHECK(!p.ok, "two defaults in one list: must error");
        CHECK(has_error_with(&p.diag, "'default' must not appear more than once in the same case"),
              "double-default-in-list diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    continuing {\n"
            "      loop {\n"
            "        if (true) { break; }\n"
            "        return;\n"
            "      }\n"
            "      break if true;\n"
            "    }\n"
            "  }\n"
            "}\n");
        CHECK(!p.ok, "return in inner loop inside continuing: must error");
        CHECK(has_error_with(&p.diag, "'return' is not allowed inside a continuing block"),
              "return-in-inner-loop-in-continuing diag");
        done(&p);
    }
    {
        /* §9.4.8 — continue before a decl used in continuing. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    if (true) { continue; }\n"
            "    let x = 1;\n"
            "    continuing { let y = x; break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(!p.ok, "continue before referenced decl: must error");
        CHECK(has_error_with(&p.diag, "would re-enter the loop's continuing block"),
              "continue-past-decl diag");
        done(&p);
    }
    {
        /* §9.4.8 — the continuing-uses set must grow past its old fixed cap
         * of 8.  Here 8 referenced decls precede the `continue` and only the
         * 9th (a8) is declared after it, so the error is surfaced only if a8
         * is tracked — which a fixed 8-entry set would have dropped. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    let a0=0; let a1=0; let a2=0; let a3=0;\n"
            "    let a4=0; let a5=0; let a6=0; let a7=0;\n"
            "    if (true) { continue; }\n"
            "    let a8=0;\n"
            "    continuing { let y = a0+a1+a2+a3+a4+a5+a6+a7+a8;"
            " break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(!p.ok, "continue before 9th referenced decl: must error");
        CHECK(has_error_with(&p.diag, "would re-enter the loop's continuing block"),
              "continue-past-decl (>8 uses) diag");
        done(&p);
    }
    {
        /* §9.4.8 — continue AFTER the decl is fine. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    let x = 1;\n"
            "    if (true) { continue; }\n"
            "    continuing { let y = x; break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continue after decl: ok");
        done(&p);
    }
    {
        /* §9.4.8 — continuing that doesn't reference any loop-local
         * decl: continues are unrestricted. */
        P p; run(&p,
            "fn f() {\n"
            "  loop {\n"
            "    if (true) { continue; }\n"
            "    let x = 1;\n"
            "    continuing { break if true; }\n"
            "  }\n"
            "}\n");
        CHECK(p.ok, "continuing references no loop-local decl: ok");
        done(&p);
    }
    {
        /* §9.4.10 — void fn with `return expr;` rejected. */
        P p; run(&p, "fn f() { return 1; }\n");
        CHECK(!p.ok, "void return expr: must error");
        CHECK(has_error_with(&p.diag, "void function"),
              "void-return-expr diag");
        done(&p);
    }
    {
        /* §9.4.10 — non-void fn with bare `return;` rejected. */
        P p; run(&p, "fn f() -> i32 { return; }\n");
        CHECK(!p.ok, "bare return in non-void fn: must error");
        CHECK(has_error_with(&p.diag, "requires an expression"),
              "bare-return diag");
        done(&p);
    }
    {
        /* §9.4.10 — return value type mismatch. */
        P p; run(&p, "fn f() -> i32 { return 1.5; }\n");
        CHECK(!p.ok, "return type mismatch: must error");
        CHECK(has_error_with(&p.diag, "does not match function"),
              "return-type-mismatch diag");
        done(&p);
    }
    {
        /* §9.4.10 — happy path: matching return types. */
        P p; run(&p, "fn f() -> i32 { return 0; }\n");
        CHECK(p.ok, "return with matching type: tc_ok");
        done(&p);
    }
    {
        /* §9.5 — call statement discards `@must_use` result. */
        P p; run(&p,
            "@must_use fn g() -> i32 { return 0; }\n"
            "fn f() { g(); }\n");
        CHECK(!p.ok, "@must_use call discarded: must error");
        CHECK(has_error_with(&p.diag, "@must_use"),
              "must_use diag");
        done(&p);
    }
    {
        /* §9.5 — builtin `@must_use` comes from generated metadata. */
        P p; run(&p, "fn f() { sin(1.0); }\n");
        CHECK(!p.ok, "builtin @must_use call discarded: must error");
        CHECK(has_error_with(&p.diag, "@must_use"),
              "builtin must_use diag");
        done(&p);
    }
    {
        /* §9.5 — consuming the result is legal. */
        P p; run(&p,
            "@must_use fn g() -> i32 { return 0; }\n"
            "fn f() { let _x = g(); }\n");
        CHECK(p.ok, "@must_use call consumed: tc_ok");
        done(&p);
    }

    /* ── §11 — Function restrictions ──────────────────────────── */
    {
        /* §11.4 — entry points must never be call targets. */
        P p; run(&p,
            "@compute @workgroup_size(1) fn main_cs() {}\n"
            "fn other() { main_cs(); }\n");
        CHECK(!p.ok, "calling @compute entry: must error");
        CHECK(has_error_with(&p.diag,
                  "cannot call entry point 'main_cs'"),
              "entry-point-call diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@vertex fn vs() -> @builtin(position) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n"
            "fn other() { let x = vs(); }\n");
        CHECK(!p.ok, "calling @vertex entry: must error");
        CHECK(has_error_with(&p.diag, "vertex"),
              "entry-point-call vertex diag");
        done(&p);
    }
    {
        /* §11.4 — vertex shader must return @builtin(position).
         * Direct return-attribute case. */
        P p; run(&p,
            "@vertex fn vs() -> @location(0) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n");
        CHECK(!p.ok, "vertex w/o @position: must error");
        CHECK(has_error_with(&p.diag, "@builtin(position)"),
              "vertex-no-position diag");
        done(&p);
    }
    {
        /* §11.4 — happy path: @vertex with @builtin(position) on the
         * return type. */
        P p; run(&p,
            "@vertex fn vs() -> @builtin(position) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n");
        CHECK(p.ok, "vertex w/ @position: tc_ok");
        done(&p);
    }
    {
        /* §11.4 — struct return: walks members for @builtin(position). */
        P p; run(&p,
            "struct VsOut {\n"
            "  @builtin(position) pos: vec4<f32>,\n"
            "  @location(0) color: vec3<f32>,\n"
            "}\n"
            "@vertex fn vs() -> VsOut {\n"
            "  return VsOut(vec4<f32>(0.0), vec3<f32>(1.0));\n"
            "}\n");
        CHECK(p.ok, "vertex returning struct with @position: tc_ok");
        done(&p);
    }
    {
        /* §11.4 — struct return without @builtin(position) member. */
        P p; run(&p,
            "struct VsOut {\n"
            "  @location(0) color: vec3<f32>,\n"
            "}\n"
            "@vertex fn vs() -> VsOut {\n"
            "  return VsOut(vec3<f32>(1.0));\n"
            "}\n");
        CHECK(!p.ok, "vertex returning struct w/o @position: must error");
        CHECK(has_error_with(&p.diag, "@builtin(position)"),
              "vertex-struct-no-position diag");
        done(&p);
    }
    {
        /* §11.4 — @fragment and @compute are NOT subject to the
         * vertex-position rule. */
        P p; run(&p,
            "@fragment fn fs() -> @location(0) vec4<f32> "
            "{ return vec4<f32>(0.0); }\n"
            "@compute @workgroup_size(1) fn cs() {}\n");
        CHECK(p.ok, "fragment + compute w/o @position: tc_ok");
        done(&p);
    }

    /* ── §12 — Attributes ─────────────────────────────────────── */
    /* The conformance round-up below mostly drives `validate.c`'s
     * attribute table — but it uses the same `run` harness so a
     * regression in any of the new rules surfaces here. */
    {
        /* §12 general — duplicate attribute reject. */
        WGSLResult *r = wgsl_check(
            "@vertex fn vs(@location(0) @location(1) v: f32) -> "
            "@builtin(position) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location(0) @location(1): must error");
        const char *needle = "must not be specified more than once";
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, needle)) { hit = 1; break; }
        }
        CHECK(hit, "duplicate-attr diag");
        wgsl_free(r);
    }
    {
        /* §12.1 @align positive. */
        WGSLResult *r = wgsl_check("struct S { @align(-4) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@align(-4): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@align value must be positive")) { hit = 1; break; }
        }
        CHECK(hit, "@align-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.1 @align power of 2. */
        WGSLResult *r = wgsl_check("struct S { @align(7) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@align(7): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "power of 2")) { hit = 1; break; }
        }
        CHECK(hit, "@align-power2 diag");
        wgsl_free(r);
    }
    {
        /* §12.7 @group non-negative. */
        WGSLResult *r = wgsl_check(
            "@group(-1) @binding(0) var<storage> b: array<f32>;\n");
        CHECK(!wgsl_ok(r), "@group(-1): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "non-negative")) { hit = 1; break; }
        }
        CHECK(hit, "@group-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.9 @interpolate without @location. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@interpolate(flat) v: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@interpolate w/o @location: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@interpolate")) { hit = 1; break; }
        }
        CHECK(hit, "@interpolate-no-location diag");
        wgsl_free(r);
    }
    {
        /* §12.10 @invariant only on @builtin(position). */
        WGSLResult *r = wgsl_check(
            "struct S {\n"
            "  @builtin(position) p: vec4<f32>,\n"
            "  @location(0) @invariant v: f32,\n"
            "}\n"
            "@vertex fn vs() -> S { return S(vec4<f32>(0), 0.0); }\n");
        CHECK(!wgsl_ok(r), "@invariant on non-position: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@invariant")) { hit = 1; break; }
        }
        CHECK(hit, "@invariant-pos diag");
        wgsl_free(r);
    }
    {
        /* §12.12 @must_use on void fn. */
        WGSLResult *r = wgsl_check("@must_use fn g() {}\n");
        CHECK(!wgsl_ok(r), "@must_use on void: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@must_use")) { hit = 1; break; }
        }
        CHECK(hit, "@must_use-void diag");
        wgsl_free(r);
    }
    {
        /* §12.13 @size positive. */
        WGSLResult *r = wgsl_check("struct S { @size(-4) x: f32 }\n");
        CHECK(!wgsl_ok(r), "@size(-4): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@size value must be positive")) { hit = 1; break; }
        }
        CHECK(hit, "@size-negative diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(0). */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(0) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(0): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@workgroup_size argument must be a positive")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-zero diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(1.0) — float forbidden. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1.0) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(1.0): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "floating-point")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-float diag");
        wgsl_free(r);
    }
    {
        /* §12.14 @workgroup_size(1i, 1u) — same-type rule. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1i, 1u) fn cs() {}\n");
        CHECK(!wgsl_ok(r), "@workgroup_size(1i, 1u): must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "same type")) { hit = 1; break; }
        }
        CHECK(hit, "@workgroup_size-mixed diag");
        wgsl_free(r);
    }
    {
        /* §12.14 happy path — all-AbstractInt accepted. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(8, 8, 1) fn cs() {}\n");
        CHECK(wgsl_ok(r), "@workgroup_size(8, 8, 1): tc_ok");
        wgsl_free(r);
    }

    /* ── §13 — Entry-point conformance ────────────────────────── */
    {
        /* §13.2 — @compute must not have a return type. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs() -> i32 { return 0; }\n");
        CHECK(!wgsl_ok(r), "@compute return: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "must not have a return")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "compute-return diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — @builtin matrix: stage / direction / type. */
        WGSLResult *r = wgsl_check(
            "@vertex fn vs(@builtin(vertex_index) i: i32) -> "
            "@builtin(position) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "vertex_index as i32: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "requires type u32")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-type diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — @position on a compute param rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs(@builtin(position) p: vec4<f32>) {}\n");
        CHECK(!wgsl_ok(r), "@position on compute: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "not valid on this shader stage")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-stage diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.1 — duplicate @builtin on one entry point rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs("
            "@builtin(global_invocation_id) a: vec3<u32>, "
            "@builtin(global_invocation_id) b: vec3<u32>) {}\n");
        CHECK(!wgsl_ok(r), "dup @builtin: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "more than once")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "builtin-dup diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — @location with bool type rejected. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) b: bool) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location bool: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "numeric scalar or vector")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "location-bool diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — bare entry-point param with no @location /
         * @builtin rejected. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(v: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "bare entry param: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "@location or @builtin")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "bare-entry-param diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.2 — @location on compute entry rejected. */
        WGSLResult *r = wgsl_check(
            "@compute @workgroup_size(1) fn cs(@location(0) v: vec3<f32>) {}\n");
        CHECK(!wgsl_ok(r), "@location on compute: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "compute")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "compute-user-IO diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — duplicate @location across params. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) a: f32, @location(0) b: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "dup @location params: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "duplicate @location")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "dup-location diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — input vs output share @location is allowed. */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) a: f32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(wgsl_ok(r), "input/output @location(0) overlap: tc_ok");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — @location on a struct-typed param rejected. */
        WGSLResult *r = wgsl_check(
            "struct D { x: f32 }\n"
            "@fragment fn fs(@location(0) in: D) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "@location on struct param: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "struct-typed entry parameter")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "location-on-struct-param diag");
        wgsl_free(r);
    }
    {
        /* §13.3.1.3 — nested IO struct rejected. */
        WGSLResult *r = wgsl_check(
            "struct Inner { @location(0) v: f32 }\n"
            "struct Outer { i: Inner }\n"
            "@fragment fn fs(in: Outer) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "nested IO struct: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "cannot nest")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "nested-IO diag");
        wgsl_free(r);
    }
    {
        /* CTS parity: integer-typed inter-stage user-defined IO requires
         * explicit @interpolate(flat). */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) v: u32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(!wgsl_ok(r), "integer IO w/o flat: must error");
        wgsl_free(r);
    }
    {
        /* §13.3.1.4 — happy path: integer IO with @interpolate(flat). */
        WGSLResult *r = wgsl_check(
            "@fragment fn fs(@location(0) @interpolate(flat) v: u32) -> "
            "@location(0) vec4<f32> { return vec4<f32>(0); }\n");
        CHECK(wgsl_ok(r), "integer IO w/ flat: tc_ok");
        wgsl_free(r);
    }
    {
        /* §13.3.2 — duplicate (group, binding) pair. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<storage> a: array<f32>;\n"
            "@group(0) @binding(0) var<storage> b: array<f32>;\n"
            "@compute @workgroup_size(1) fn main() {\n"
            "  _ = arrayLength(&a); _ = arrayLength(&b);\n"
            "}\n");
        CHECK(!wgsl_ok(r), "dup binding: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "duplicate resource binding")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "dup-binding diag");
        wgsl_free(r);
    }

    /* ── §14 — Memory layout (uniform AS constraints) ─────────── */
    {
        /* §14.4.5 — uniform AS forbids runtime-sized array. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<f32>;\n");
        CHECK(!wgsl_ok(r), "uniform runtime-array: must error");
        int hit = 0;
        for (int i = 0; i < wgsl_diagnostic_count(r); i++) {
            if (strstr(wgsl_diagnostic(r, i)->message, "runtime-sized")) {
                hit = 1; break;
            }
        }
        CHECK(hit, "uniform-rta diag");
        wgsl_free(r);
    }
    {
        /* With uniform_buffer_standard_layout support enabled, uniform AS
         * arrays use standard layout and no 16-byte stride bump is needed. */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<f32, 8>;\n");
        CHECK(wgsl_ok(r), "uniform array<f32, N>: ok under standard layout");
        wgsl_free(r);
    }
    {
        /* §14.4.5 happy path: uniform array of vec4<f32> (stride 16). */
        WGSLResult *r = wgsl_check(
            "@group(0) @binding(0) var<uniform> a: array<vec4<f32>, 4>;\n");
        CHECK(wgsl_ok(r), "uniform array<vec4<f32>, N>: tc_ok");
        wgsl_free(r);
    }

