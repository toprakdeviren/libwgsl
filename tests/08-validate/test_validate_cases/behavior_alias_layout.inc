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

