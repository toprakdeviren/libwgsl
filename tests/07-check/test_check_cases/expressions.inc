    /* ── §8 — expression-level gates ─────────────────────────── */
    {
        /* §8.5.1 — mixed swizzle letterings rejected. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v.xrgb; }\n");
        CHECK(!p.ok, "v.xrgb: must error");
        CHECK(has_error_with(&p.diag, "mixed swizzle"),
              "mixed-swizzle diag");
        done(&p);
    }
    {
        /* §8.5.1 — vec const-index out of range. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[5]; }\n");
        CHECK(!p.ok, "v[5] on vec3: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "vec out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.1 — vector index must be an integer scalar. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[1.0]; }\n");
        CHECK(!p.ok, "v[1.0] on vec3: must error");
        CHECK(has_error_with(&p.diag, "expected integer scalar"),
              "vec float-index diag");
        done(&p);
    }
    {
        /* §8.5.1 — bool is not a valid vector index type. */
        P p; run(&p, "fn f() { let v = vec3<f32>(0.0); let x = v[true]; }\n");
        CHECK(!p.ok, "v[true] on vec3: must error");
        CHECK(has_error_with(&p.diag, "expected integer scalar"),
              "vec bool-index diag");
        done(&p);
    }
    {
        /* §8.5.2 — mat const-index out of range. */
        P p; run(&p, "fn f() { var m: mat3x2<f32>; let x = m[5]; }\n");
        CHECK(!p.ok, "m[5] on mat3x2: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "mat out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.3 — array const-index out of range. */
        P p; run(&p, "fn f() { var a: array<i32,3>; let x = a[5]; }\n");
        CHECK(!p.ok, "a[5] on array<i32,3>: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "array out-of-range diag");
        done(&p);
    }
    {
        /* §8.5.3 — negative const-index on array. */
        P p; run(&p, "fn f() { var a: array<i32,3>; let x = a[-1]; }\n");
        CHECK(!p.ok, "a[-1]: must error");
        CHECK(has_error_with(&p.diag, "negative"),
              "negative-index diag");
        done(&p);
    }
    {
        /* Runtime-sized arrays still reject negative const indices. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: array<i32>;\n"
            "fn f() { let x = a[-1]; }\n");
        CHECK(!p.ok, "runtime array a[-1]: must error");
        CHECK(has_error_with(&p.diag, "negative"),
              "runtime array negative-index diag");
        done(&p);
    }
    {
        /* Runtime index must NOT trip the static range check. */
        P p; run(&p,
            "fn f(i: u32) { var a: array<i32,3>; let x = a[i]; }\n");
        CHECK(p.ok, "runtime-index a[i]: tc_ok");
        CHECK(!has_error_with(&p.diag, "out of range"),
              "runtime index: no static range diag");
        done(&p);
    }
    {
        /* §8.6 — vec<bool> component-wise `&` / `|` logical ops allowed. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let c = a & b;\n"
            "  let d = a | b;\n"
            "}\n");
        CHECK(p.ok, "vec<bool> &|: tc_ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let e = a ^ b;\n"
            "}\n");
        CHECK(!p.ok, "vec<bool> ^ : must error");
        done(&p);
    }
    {
        /* §8.6 — integer bitwise ops do not scalar-broadcast to vectors. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<i32>(0, 0);\n"
            "  let b = 0;\n"
            "  let c = a & b;\n"
            "}\n");
        CHECK(!p.ok, "vec2<i32> & scalar: must error");
        CHECK(has_error_with(&p.diag, "matching shapes"),
              "bitwise scalar-vector diag");
        done(&p);
    }
    {
        /* §8.6 — shifts still integer-only. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = vec2<bool>(true, false);\n"
            "  let b = vec2<bool>(false, true);\n"
            "  let c = a << b;\n"
            "}\n");
        CHECK(!p.ok, "vec<bool> << : must error");
        CHECK(has_error_with(&p.diag, "integer"),
              "vec<bool> shift diag");
        done(&p);
    }
    {
        /* §8.7 — div by zero at const-eval. */
        P p; run(&p, "fn f() { let x: i32 = 5 / 0; }\n");
        CHECK(!p.ok, "5/0 const-eval: must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "div-by-zero diag");
        done(&p);
    }
    {
        /* §8.7 — mod by zero at const-eval. */
        P p; run(&p, "fn f() { let x: i32 = 5 % 0; }\n");
        CHECK(!p.ok, "5%0 const-eval: must error");
        CHECK(has_error_with(&p.diag, "modulo by zero"),
              "mod-by-zero diag");
        done(&p);
    }
    {
        /* Runtime div-by-zero must NOT fire at type-check time
         * (the consteval noise filter rolls back "not a constant"
         * for runtime initializers). */
        P p; run(&p, "fn f(x: i32, y: i32) { let z = x / y; }\n");
        CHECK(p.ok, "runtime x/y: tc_ok");
        CHECK(!has_error_with(&p.diag, "division by zero"),
              "runtime div: no spurious diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 1; x /= 0; }\n");
        CHECK(!p.ok, "compound division by const zero: must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "compound div-by-zero diag");
        done(&p);
    }
    {
        /* §17.5.52 — dot(vecN<AbstractInt>) evaluates as AbstractInt
         * and must reject intermediate products outside the i64 domain. */
        P p; run(&p,
            "const x = dot(vec2(9223372036854775807, 9223372036854775807), "
            "vec2(2, 2));\n");
        CHECK(!p.ok, "abstract-int dot overflow: must error");
        CHECK(has_error_with(&p.diag, "overflow"),
              "abstract-int dot overflow diag");
        done(&p);
    }
    {
        /* §8.9 — shift amount out of range at const-eval. */
        P p; run(&p, "fn f() { let x = 1u << 33u; }\n");
        CHECK(!p.ok, "1u << 33u: must error");
        CHECK(has_error_with(&p.diag, "shift amount"),
              "shift-range diag");
        done(&p);
    }
    {
        /* §8.9 — a constant shift amount is checked even when the
         * shifted value is runtime. */
        P p; run(&p, "fn f() { var v: i32 = 0; let x = v << 32u; }\n");
        CHECK(!p.ok, "runtime lhs << 32u: must error");
        CHECK(has_error_with(&p.diag, "shift amount"),
              "runtime shift const amount diag");
        done(&p);
    }
    {
        /* §8.9 — AbstractInt right shift by >= 64 is defined and
         * saturates in const-eval; left shift still range-checks. */
        P p; run(&p, "const x = 1 >> 64;\n");
        CHECK(p.ok, "abstract-int 1 >> 64: ok");
        done(&p);
    }
    {
        /* §8.9 — shifts are only defined for scalar/vector operands,
         * not arrays even when their element type is integer. */
        P p; run(&p,
            "var<private> a: array<u32, 4>;\n"
            "fn f() { let x = a << a; }\n");
        CHECK(!p.ok, "array << array: must error");
        CHECK(has_error_with(&p.diag, "shift operands"),
              "array shift diag");
        done(&p);
    }
    {
        /* §15.7.3 finite-math — AbstractFloat overflow producing inf
         * is rejected (1e200 * 1e200 ≈ 1e400 > 1.8e308). */
        P p; run(&p, "fn f() { let x: f32 = 1e200 * 1e200; }\n");
        CHECK(!p.ok, "1e200 * 1e200: must error");
        CHECK(has_error_with(&p.diag, "infinity"),
              "fp inf diag");
        done(&p);
    }
    {
        /* §17.5.53 — pow(0, y) is only defined for positive y in
         * const-eval validation. */
        P p; run(&p, "const x = pow(0.0, 0.0);\n");
        CHECK(!p.ok, "pow(0, 0): must error");
        CHECK(has_error_with(&p.diag, "zero base"),
              "pow zero-base diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { _ = atanh(1f); }\n");
        CHECK(!p.ok, "atanh(1f) const builtin domain: must error");
        CHECK(has_error_with(&p.diag, "builtin 'atanh' domain error"),
              "atanh runtime-literal domain diag");
        done(&p);
    }
    {
        /* §17.5.16 — clamp() rejects constant low/high bounds with
         * low > high even when the clamped value is runtime. */
        P p; run(&p,
            "fn f() { var v: f32; let x = clamp(v, 1.0, 0.0); }\n");
        CHECK(!p.ok, "clamp runtime value with low > high: must error");
        CHECK(has_error_with(&p.diag, "low must be <= high"),
              "clamp low/high diag");
        done(&p);
    }
    {
        /* §17.5.62 — smoothstep() rejects equal constant edges even
         * when x is runtime. */
        P p; run(&p,
            "fn f() { var x: f32; let y = smoothstep(0.0, 0.0, x); }\n");
        CHECK(!p.ok, "smoothstep equal edges: must error");
        CHECK(has_error_with(&p.diag, "edge0 and edge1 must differ"),
              "smoothstep edge diag");
        done(&p);
    }
    {
        /* §17.5.56 — normalize() squares components during const-eval;
         * those intermediate operations must stay inside the result type. */
        P p; run(&p,
            "const x = normalize(vec2<f32>("
            "3.4028234663852886e+38f, 3.4028234663852886e+38f));\n");
        CHECK(!p.ok, "normalize f32 intermediate overflow: must error");
        CHECK(has_error_with(&p.diag, "infinity") ||
              has_error_with(&p.diag, "non-finite"),
              "normalize f32 overflow diag");
        done(&p);
    }
    {
        /* §8.9 — shift amount must be u32 (or vec<u32>); `1u << 2i`
         * was previously rejected as "wrong reason" (binary numeric). */
        P p; run(&p, "fn f() { let x = 1u << 2i; }\n");
        CHECK(!p.ok, "1u << 2i: must error");
        CHECK(has_error_with(&p.diag, "shift amount must be u32"),
              "shift amount type diag");
        done(&p);
    }
    {
        /* §8.9 — abstract-int shift amount converts to u32: ok. */
        P p; run(&p, "fn f() { let x = 1u << 2; }\n");
        CHECK(p.ok, "1u << 2: ok");
        done(&p);
    }
    {
        /* §8.9 — vector LHS with scalar RHS is rejected. */
        P p; run(&p,
            "fn f() { let v = vec2<u32>(1u, 2u); let x = v << 1u; }\n");
        CHECK(!p.ok, "vec << scalar: must error");
        CHECK(has_error_with(&p.diag, "shift operand shapes must match"),
              "shift shape diag");
        done(&p);
    }
    {
        /* §8.1 — short-circuit at const-eval: RHS skipped when LHS
         * already decides the result.  The RHS would otherwise trip
         * div-by-zero. */
        P p; run(&p,
            "const X: bool = false && (5 / 0 == 0);\n");
        CHECK(p.ok, "false && (5/0==0): ok via short-circuit");
        CHECK(!has_error_with(&p.diag, "division by zero"),
              "no spurious div-by-zero");
        done(&p);
    }
    {
        /* §8.1 — || mirror: true || (5/0==0) skips RHS. */
        P p; run(&p,
            "const X: bool = true || (5 / 0 == 0);\n");
        CHECK(p.ok, "true || (5/0==0): ok via short-circuit");
        done(&p);
    }
    {
        /* §8.1 — non-short-circuit path still walks RHS. */
        P p; run(&p,
            "const X: bool = true && (5 / 0 == 0);\n");
        CHECK(!p.ok, "true && (5/0==0): RHS evaluated, must error");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "div-by-zero from RHS");
        done(&p);
    }
    {
        /* §8.10 — argument count mismatch on user-fn call. */
        P p; run(&p, "fn g(a: i32, b: i32) {} fn f() { g(1); }\n");
        CHECK(!p.ok, "g(1) of g(i32,i32): must error");
        CHECK(has_error_with(&p.diag, "argument count mismatch"),
              "arg-count diag");
        done(&p);
    }
    {
        /* §8.10 — argument type mismatch (concrete f32 → concrete i32
         * has no automatic conversion). */
        P p; run(&p, "fn g(a: i32) {} fn f() { g(1.5f); }\n");
        CHECK(!p.ok, "g(1.5f): must error");
        CHECK(has_error_with(&p.diag, "argument 1"),
              "arg-type diag");
        done(&p);
    }
    {
        /* §8.10 — happy path: argument that converts via abstract→i32
         * stays legal. */
        P p; run(&p, "fn g(a: i32) {} fn f() { g(1); }\n");
        CHECK(p.ok, "g(1) of g(i32): tc_ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let abs = 4;\n"
            "  _ = abs(1);\n"
            "}\n");
        CHECK(!p.ok, "shadowed builtin callee is not callable");
        CHECK(has_error_with(&p.diag, "not callable"),
              "shadowed builtin callee diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let vec2 = 4;\n"
            "  _ = vec2<f32>(1.0, 2.0);\n"
            "}\n");
        CHECK(!p.ok, "shadowed type generator callee is not callable");
        CHECK(has_error_with(&p.diag, "not callable") ||
              has_error_with(&p.diag, "not a type"),
              "shadowed typegen callee diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> sampler: f32;\n"
            "@group(0) @binding(0) var s: sampler;\n");
        CHECK(!p.ok, "shadowed handle type is not a type");
        CHECK(has_error_with(&p.diag, "not a type"),
              "shadowed handle type diag");
        done(&p);
    }
    {
        /* §8.13 — `&v.x` (vector component) rejected. */
        P p; run(&p, "fn f() { var v: vec3<f32>; let p = &v.x; }\n");
        CHECK(!p.ok, "&v.x: must error");
        CHECK(has_error_with(&p.diag, "vector component"),
              "addr-of-vec-component diag");
        done(&p);
    }
    {
        /* §8.13 — `&t` on a handle-AS texture rejected. */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "fn f() { let p = &t; }\n");
        CHECK(!p.ok, "&t (texture): must error");
        CHECK(has_error_with(&p.diag, "handle"),
              "addr-of-handle diag");
        done(&p);
    }

