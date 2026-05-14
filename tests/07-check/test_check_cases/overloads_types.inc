    /* ── §6.1.2 / §6.1.3 — Overload resolution ────────────────── *
     *
     * Pins the §6.2.1 spec example shape: `log2(32)` ranks the
     * AbstractFloat overload (rank 5) above the f32/f16 overloads
     * (ranks 6/7), so the call returns AbstractFloat — which the
     * surrounding let then concretizes to f32.
     */
    {
        P p; run(&p, "fn f() { let r = log2(32); }\n");
        CHECK(p.ok, "log2(32): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "log2(32) → AbstractFloat → f32 via let");
        done(&p);
    }
    {
        /* `log2(32f)` — only the f32 overload is feasible. */
        P p; run(&p, "fn f() { let r = log2(32f); }\n");
        CHECK(p.ok, "log2(32f): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32, "log2(32f) → f32");
        done(&p);
    }
    {
        /* `log2(1u)` — no float overload accepts u32 (i32→f32 is also
         * infeasible per §6.1.2).  Must error. */
        P p; run(&p, "fn f() { let r = log2(1u); }\n");
        CHECK(!p.ok, "log2(1u): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "log2(1u) no-overload diag");
        done(&p);
    }
    {
        /* §6.2.1 example 2: `1 + 2.5` style — applied to a builtin.
         * `min(1, 1.5)` ranks AbstractFloat (5, 0) above f32 (6, 1)
         * and f16 (7, 2).  AbstractFloat wins → let concretizes to f32. */
        P p; run(&p, "fn f() { let r = min(1, 1.5); }\n");
        CHECK(p.ok, "min(1, 1.5): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "min(1, 1.5) → AbstractFloat → f32");
        done(&p);
    }
    {
        /* `max(1, 2)` — both AbstractInt, lowest rank is AbstractInt
         * (0, 0).  Let then concretizes AbstractInt → i32. */
        P p; run(&p, "fn f() { let r = max(1, 2); }\n");
        CHECK(p.ok, "max(1, 2): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32,
              "max(1, 2) → AbstractInt → i32");
        done(&p);
    }
    {
        /* `min(1u, 1)` — only the u32 overload is feasible
         * (AbstractInt→u32 is rank 4, u32→u32 is rank 0; the
         * AbstractInt overload requires u32→AbstractInt which is
         * infeasible). */
        P p; run(&p, "fn f() { let r = min(1u, 1); }\n");
        CHECK(p.ok, "min(1u, 1): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_u32, "min(1u, 1) → u32");
        done(&p);
    }
    {
        /* Float-only family rejects integer concrete arg. */
        P p; run(&p, "fn f() { let r = cos(1u); }\n");
        CHECK(!p.ok, "cos(1u): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "cos(1u) no-overload diag");
        done(&p);
    }
    {
        /* Integer-only family rejects float arg. */
        P p; run(&p, "fn f() { let r = reverseBits(1.0); }\n");
        CHECK(!p.ok, "reverseBits(1.0): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "reverseBits(1.0) no-overload diag");
        done(&p);
    }
    {
        /* Numeric family on a vec carries the shape through. */
        P p; run(&p, "fn f() { let r = abs(vec3<f32>(1.0)); }\n");
        CHECK(p.ok, "abs(vec3<f32>): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 3 && r->type->ref == p.types.t_f32,
              "abs(vec3<f32>) → vec3<f32>");
        done(&p);
    }
    {
        /* Reduction builtins: BR_ARG0_ELEM combined with overload
         * resolution.  `length(vec3<f32>(1))` picks T=f32, then the
         * BR_ARG0_ELEM rule yields f32 (the element scalar). */
        P p; run(&p, "fn f() { let r = length(vec3<f32>(1.0)); }\n");
        CHECK(p.ok, "length(vec3<f32>): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "length(vec3<f32>) → f32 (scalar element)");
        done(&p);
    }
    {
        /* §17 exact rows: sign excludes unsigned integer T. */
        P p; run(&p, "fn f() { let r = sign(1u); }\n");
        CHECK(!p.ok, "sign(u32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "sign(u32) no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: normalize is vector-only, not scalar float. */
        P p; run(&p, "fn f() { let r = normalize(1.0); }\n");
        CHECK(!p.ok, "normalize(f32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "normalize scalar no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: determinant only accepts square matrices. */
        P p; run(&p, "fn f() { let r = determinant(mat2x3<f32>()); }\n");
        CHECK(!p.ok, "determinant(mat2x3): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "determinant rectangular no-overload diag");
        done(&p);
    }
    {
        /* §17 exact rows: transpose flips C/R in the return shape. */
        P p; run(&p, "fn f() { let r = transpose(mat2x3<f32>()); }\n");
        CHECK(p.ok, "transpose(mat2x3): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_MAT &&
              r->type->width == 3 && r->type->rows == 2 &&
              r->type->ref == p.types.t_f32,
              "transpose(mat2x3<f32>) → mat3x2<f32>");
        done(&p);
    }
    {
        /* §17 exact rows: dot4I8Packed returns i32, not u32. */
        P p; run(&p,
            "fn f() { let r: i32 = dot4I8Packed(1u, 2u); }\n");
        CHECK(p.ok, "dot4I8Packed return i32: tc_ok");
        done(&p);
    }
    {
        /* §17.6 exact rows: derivative builtins accept f32, not f16. */
        P p; run(&p,
            "enable f16;\n"
            "fn f() { let r = dpdx(1.0h); }\n");
        CHECK(!p.ok, "dpdx(f16): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "dpdx(f16) no-overload diag");
        done(&p);
    }
    {
        /* Step-2 smoke: a non-const concrete arg forces concretization.
         * `min(1, v)` with `var v: f32 = 0.0;` resolves to T=f32 (the
         * AbstractFloat candidate is eliminated by step 1, since
         * f32→AbstractFloat is infeasible). */
        P p; run(&p,
            "fn f() {\n"
            "  var v: f32 = 0.0;\n"
            "  let r = min(1, v);\n"
            "}\n");
        CHECK(p.ok, "min(1, v) non-const: tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_f32,
              "min(1, var f32) → f32");
        done(&p);
    }
    {
        /* Const-expression ident participates as const-expr — picks the
         * AbstractInt overload just like a literal would. */
        P p; run(&p,
            "const C: i32 = 1;\n"
            "fn f() { let r = max(C, 2); }\n");
        CHECK(p.ok, "max(const C, 2): tc_ok");
        /* C is a CONST symbol with i32 type, so arg0 has rank table
         * (i32→i32)=0, (AbstractInt→i32)=3; the AbstractInt overload
         * is eliminated by step 1 since i32→AbstractInt is infeasible.
         * Result: i32. */
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32, "max(const i32, 2) → i32");
        done(&p);
    }
    {
        /* `clamp(0, lo, hi)` over abstract literals — ternary picks
         * AbstractInt with rank vector (0, 0, 0). */
        P p; run(&p, "fn f() { let r = clamp(0, 1, 2); }\n");
        CHECK(p.ok, "clamp(0, 1, 2): tc_ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type == p.types.t_i32,
              "clamp(0,1,2) → AbstractInt → i32");
        done(&p);
    }
    {
        /* Mixing concrete float and concrete int with no automatic
         * conversion between them — must error. */
        P p; run(&p, "fn f() { let r = min(1.0f, 1u); }\n");
        CHECK(!p.ok, "min(f32, u32): must error");
        CHECK(has_error_with(&p.diag, "no matching overload"),
              "min(f32, u32) no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = mix(vec2<f32>(0.0), vec2<f32>(1.0), 0.5); }\n");
        CHECK(p.ok, "mix(vec2, vec2, scalar): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 2 && r->type->ref == p.types.t_f32,
              "mix scalar factor preserves vector result");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = mix(vec2<f32>(0.0), vec2<f32>(1.0), vec3<f32>(0.5)); }\n");
        CHECK(!p.ok, "mix(vec2, vec2, vec3 factor): must error");
        CHECK(has_error_with(&p.diag, "mix factor"),
              "mix factor shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = ldexp(vec2<f32>(1.0), vec2<i32>(1)); }\n");
        CHECK(p.ok, "ldexp(vec2<f32>, vec2<i32>): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { let r = ldexp(vec2<f32>(1.0), 1); }\n");
        CHECK(!p.ok, "ldexp(vec2<f32>, scalar exponent): must error");
        CHECK(has_error_with(&p.diag, "exponent argument"),
              "ldexp exponent shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() { var v: f32; let r = ldexp(v, i32(129)); }\n");
        CHECK(!p.ok, "ldexp(f32, 129): must error");
        CHECK(has_error_with(&p.diag, "exponent too large"),
              "ldexp f32 exponent domain diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable f16;\n"
            "fn f() { var v: vec2<f16>; let r = ldexp(v, vec2<i32>(17)); }\n");
        CHECK(!p.ok, "ldexp(vec2<f16>, 17): must error");
        CHECK(has_error_with(&p.diag, "exponent too large"),
              "ldexp f16 exponent domain diag");
        done(&p);
    }
    {
        P p; run(&p, "var<private> X = min(1, 2);\n");
        CHECK(p.ok, "@const builtin call is accepted in private var init");
        done(&p);
    }

    /* ── §6.2.1 — Abstract → concrete overflow at narrowing ──── *
     *
     * The §6.1.2 conversion table says "AbstractInt → i32 is identity
     * if the value is in i32, error otherwise."  Same for u32.  These
     * tests exercise that range check for `let` / `var` / `override`
     * (the `const` path already runs through the consteval module pass
     * and was tested separately in test_consteval_module).
     */
    {
        P p; run(&p, "fn f() { let x: i32 = 2147483648; }\n");
        CHECK(!p.ok, "let x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x: i32 = 2147483648; }\n");
        CHECK(!p.ok, "var x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "var i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "var<private> x: i32 = 2147483648;\n");
        CHECK(!p.ok, "var<private> x: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "var<private> i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "override X: i32 = 2147483648;\n");
        CHECK(!p.ok, "override X: i32 = 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "override i32 overflow diag");
        done(&p);
    }
    {
        /* Negative AbstractInt → u32 must error. */
        P p; run(&p, "fn f() { let x: u32 = -1; }\n");
        CHECK(!p.ok, "let x: u32 = -1: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let u32 negative diag");
        done(&p);
    }
    {
        /* AbstractInt > UINT32_MAX → u32 must error. */
        P p; run(&p, "fn f() { let x: u32 = 4294967296; }\n");
        CHECK(!p.ok, "let x: u32 = 2^32: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let u32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 2147483648i; }\n");
        CHECK(!p.ok, "suffixed i32 literal 2^31: must error");
        CHECK(has_error_with(&p.diag, "out of range for i32"),
              "suffixed i32 overflow diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { var x = 4294967296u; }\n");
        CHECK(!p.ok, "suffixed u32 literal 2^32: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "suffixed u32 overflow diag");
        done(&p);
    }
    {
        /* Computed const-expression that overflows materializes
         * via the binary op then the i32 conversion — must error. */
        P p; run(&p, "fn f() { let x: i32 = 1073741824 * 2; }\n");
        CHECK(!p.ok, "let x: i32 = 2^30 * 2: must error");
        CHECK(has_error_with(&p.diag, "out of range"),
              "let i32 computed overflow diag");
        done(&p);
    }
    {
        /* Boundary values that DO fit must remain accepted. */
        P p; run(&p,
            "fn f() {\n"
            "  let a: i32 =  2147483647;\n"
            "  let b: i32 = -2147483648;\n"
            "  let c: u32 =  4294967295;\n"
            "}\n");
        CHECK(p.ok, "i32/u32 boundary values: tc_ok");
        done(&p);
    }
    {
        /* Runtime initializers (non-const) MUST NOT trigger spurious
         * range diagnostics: the helper rolls back consteval's
         * "not a constant" complaints. */
        P p; run(&p, "fn f(p: i32) { let x: i32 = p + 1; }\n");
        CHECK(p.ok, "runtime fn-param init: tc_ok");
        CHECK(!has_error_with(&p.diag, "out of range"),
              "runtime init: no spurious range diag");
        CHECK(!has_error_with(&p.diag, "not a constant"),
              "runtime init: no leaked 'not a constant' diag");
        done(&p);
    }

    /* ── §6.2.4 — Per-suffix float literal range gates ──────── *
     *
     * `f` / `h` suffixes pin the literal to a concrete float type;
     * a value that doesn't fit the binding's IEEE range is a shader-
     * generation error.  strtod's ERANGE branch only catches double-
     * precision overflow (~1e308); the per-suffix check closes the
     * f32 (FLT_MAX, ~3.4e38) and f16 (65504) gates.
     */
    {
        P p; run(&p, "enable f16;\nfn f() { let x = 65504.0h; }\n");
        CHECK(p.ok, "1.0h at f16 max: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "enable f16;\nfn f() { let x = 65505.0h; }\n");
        CHECK(!p.ok, "65505.0h: must error");
        CHECK(has_error_with(&p.diag, "out of range for f16"),
              "h-suffix overflow diag");
        done(&p);
    }
    {
        /* `const` path must NOT double-emit (module pass + type
         * checker both visit the literal; CONST_EVALED flag dedup). */
        P p; run(&p, "enable f16;\nconst x = 1e10h;\n");
        CHECK(!p.ok, "const 1e10h: must error");
        int cnt = 0;
        for (size_t k = 0; k < p.diag.count; k++) {
            if (strstr(p.diag.items[k].message, "out of range for f16")) cnt++;
        }
        CHECK(cnt == 1, "h-suffix overflow: single emit (no double-fire)");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 3.4e38f; }\n");
        CHECK(p.ok, "3.4e38f: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1e40f; }\n");
        CHECK(!p.ok, "1e40f: must error");
        CHECK(has_error_with(&p.diag, "out of range for f32"),
              "f-suffix overflow diag");
        done(&p);
    }

    /* ── §6.2.8 — Atomic-typed expression is illegal ─────────── *
     *
     * "An expression must not evaluate to an atomic type."  Reading
     * an atomic var as a value (`let x = a;`) is the canonical
     * non-builtin path; atomicLoad(&a) is the only legal way.
     */
    {
        P p; run(&p,
            "var<workgroup> a: atomic<i32>;\n"
            "fn f() { let x = a; }\n");
        CHECK(!p.ok, "let x = atomic_var: must error");
        CHECK(has_error_with(&p.diag, "atomic type"),
              "atomic-in-expr diag");
        done(&p);
    }
    {
        /* The atomicLoad path remains legal — pointer arg via `&`. */
        P p; run(&p,
            "var<workgroup> a: atomic<i32>;\n"
            "fn f() { let x = atomicLoad(&a); }\n");
        CHECK(p.ok, "atomicLoad(&a): tc_ok");
        done(&p);
    }

    /* ── §6.2.9 — Override-sized array element count ────────── *
     *
     * `array<E, N>` where N is an override declaration is legal
     * (per §6.2.9 + §14.4.5 AS placement).  The length isn't
     * const-evaluable, so the type checker treats the array as
     * runtime-sized for v1 layout purposes; the spec's identifier-
     * equality refinement for two distinct overrides is a known v1.x
     * gap (see missing.md).
     */
    {
        P p; run(&p,
            "override N: u32;\n"
            "var<workgroup> wg: array<f32, N>;\n");
        CHECK(p.ok, "var<workgroup> array<f32, override N>: tc_ok");
        CHECK(!has_error_with(&p.diag, "not a constant"),
              "override-sized array: no 'not a constant' diag");
        CHECK(!has_error_with(&p.diag, "greater than zero"),
              "override-sized array: no spurious 'must be > 0' diag");
        done(&p);
    }
    {
        /* A let/var with a non-const, non-override length must still
         * error — only override symbols get the size-source pass. */
        P p; run(&p, "var<private> v: u32 = 4u;\n"
                     "var<private> a: array<f32, v>;\n");
        CHECK(!p.ok, "array sized by `var`: must error");
        CHECK(has_error_with(&p.diag, "greater than zero") ||
              has_error_with(&p.diag, "not a constant"),
              "var-sized array rejection diag");
        done(&p);
    }

    /* ── §6.2.11 / §6.2.12 / §6.2.13 — Composite predicates ──── *
     *
     * The new predicates feed the let / param / return gates so a
     * non-constructible type in those positions produces a clear,
     * single-source diagnostic instead of cascading "cannot convert"
     * complaints.
     */
    {
        /* Function return type must be constructible. */
        P p; run(&p, "fn f() -> array<f32> { return; }\n");
        CHECK(!p.ok, "fn f() -> array<f32>: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "fn return non-constructible diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() -> texture_2d<f32> { return; }\n");
        CHECK(!p.ok, "fn f() -> texture_2d<f32>: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "fn return texture diag");
        done(&p);
    }
    {
        /* Function param type must be constructible / ptr / tex / sampler. */
        P p; run(&p, "fn f(a: array<f32>) {}\n");
        CHECK(!p.ok, "fn(a: array<f32>): must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "fn param runtime-array diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(a: atomic<i32>) {}\n");
        CHECK(!p.ok, "fn(a: atomic<i32>): must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "fn param atomic diag");
        done(&p);
    }
    {
        /* Param positives — ptr / texture / sampler still legal. */
        P p; run(&p,
            "fn f(p: ptr<function, i32>, t: texture_2d<f32>, s: sampler) {}\n");
        CHECK(p.ok, "ptr / texture / sampler params: tc_ok");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<handle, u32>) {}\n");
        CHECK(!p.ok, "ptr<handle, u32> param: must error");
        CHECK(has_error_with(&p.diag, "ptr address space"),
              "ptr handle AS diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<private, u32, read>) {}\n");
        CHECK(!p.ok, "ptr<private, u32, read> param: must error");
        CHECK(has_error_with(&p.diag, "ptr access mode"),
              "ptr private access-mode diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<storage, u32, write>;\n");
        CHECK(!p.ok, "ptr<storage, u32, write>: must error");
        CHECK(has_error_with(&p.diag, "read' or 'read_write"),
              "ptr storage write access-mode diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<private, 1>;\n");
        CHECK(!p.ok, "ptr literal store type: must error");
        CHECK(has_error_with(&p.diag, "ptr store type must be a type"),
              "ptr literal store type diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<private, atomic<u32>>;\n");
        CHECK(!p.ok, "ptr<private, atomic<u32>>: must error");
        CHECK(has_error_with(&p.diag, "atomic requires the workgroup or storage"),
              "ptr atomic AS pairing diag");
        done(&p);
    }
    {
        P p; run(&p, "alias P = ptr<workgroup, array<i32>>;\n");
        CHECK(!p.ok, "ptr<workgroup, runtime array>: must error");
        CHECK(has_error_with(&p.diag, "runtime-sized array requires"),
              "ptr runtime array AS pairing diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<private, clamp>) {}\n");
        CHECK(!p.ok, "ptr<private, clamp> param: must error");
        CHECK(has_error_with(&p.diag, "not a type"),
              "ptr builtin-as-store-type diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f(p: ptr<function, texture_external>) {}\n");
        CHECK(!p.ok, "ptr<function, texture_external> param: must error");
        CHECK(has_error_with(&p.diag, "ptr store type must be storable"),
              "ptr texture store-type diag");
        done(&p);
    }
    {
        /* Fixed-size array param: constructible, legal. */
        P p; run(&p, "fn f(a: array<f32, 4>) {}\n");
        CHECK(p.ok, "array<f32, 4> param: tc_ok");
        done(&p);
    }
    {
        /* Struct deep-walk: a struct containing a runtime-sized array
         * is not constructible, so it cannot be a return type. */
        P p; run(&p,
            "struct S { a: array<f32> }\n"
            "fn f() -> S { return; }\n");
        CHECK(!p.ok, "struct with runtime array as return: must error");
        CHECK(has_error_with(&p.diag, "return type must be constructible"),
              "struct-runtime-array return diag");
        done(&p);
    }
    {
        /* Struct deep-walk: a struct containing an atomic is not
         * constructible, so it cannot be a function param. */
        P p; run(&p,
            "struct S { a: atomic<i32> }\n"
            "fn f(s: S) {}\n");
        CHECK(!p.ok, "struct with atomic as param: must error");
        CHECK(has_error_with(&p.diag, "parameter type must be constructible"),
              "struct-with-atomic param diag");
        done(&p);
    }
    {
        /* Nested struct of constructibles — legal. */
        P p; run(&p,
            "struct Inner { x: f32 }\n"
            "struct Outer { a: Inner, b: vec3<f32> }\n"
            "fn f(o: Outer) -> Outer { return o; }\n");
        CHECK(p.ok, "nested constructible struct param/return: tc_ok");
        done(&p);
    }
    {
        /* `let x: array<f32>` (runtime-sized): non-constructible. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> b: array<f32>;\n"
            "fn f() { let x: array<f32> = b; }\n");
        CHECK(!p.ok, "let x: array<f32>: must error");
        CHECK(has_error_with(&p.diag,
                  "'let' type must be a concrete constructible"),
              "let runtime-array diag");
        done(&p);
    }

    /* ── §6.3 — Enumerants are template parameters only ─────── *
     *
     * Address-space, access-mode, and texel-format enumerants live in
     * the predeclared scope so name resolution finds them — but using
     * one as a value (let init / arithmetic / etc.) is forbidden per
     * spec §6.3.  Previously this surfaced as "has no type yet"; now
     * a labelled diagnostic names the enumeration.
     */
    {
        P p; run(&p, "fn f() { let x = read_write; }\n");
        CHECK(!p.ok, "let x = read_write: must error");
        CHECK(has_error_with(&p.diag,
                  "access mode enumerant 'read_write' can only be used "
                  "as a template parameter"),
              "access-mode enumerant value diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = storage; }\n");
        CHECK(!p.ok, "let x = storage: must error");
        CHECK(has_error_with(&p.diag,
                  "address space enumerant 'storage'"),
              "address-space enumerant value diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = rgba8unorm; }\n");
        CHECK(!p.ok, "let x = rgba8unorm: must error");
        CHECK(has_error_with(&p.diag,
                  "texel format enumerant 'rgba8unorm'"),
              "texel-format enumerant value diag");
        done(&p);
    }
    {
        /* Arithmetic on enumerant — still flagged on the leaf. */
        P p; run(&p, "fn f() { let x = read + 1; }\n");
        CHECK(!p.ok, "read + 1: must error");
        CHECK(has_error_with(&p.diag, "access mode enumerant 'read'"),
              "enumerant-in-expr diag");
        done(&p);
    }
    {
        /* Template-arg usage stays legal. */
        P p; run(&p,
            "@group(0) @binding(0) "
            "var t: texture_storage_2d<rgba8unorm, write>;\n");
        CHECK(p.ok, "texture_storage_2d<rgba8unorm, write>: tc_ok");
        CHECK(!has_error_with(&p.diag, "enumerant"),
              "template-arg enumerant: no enumerant-value diag");
        done(&p);
    }

    /* ── §7 — declaration / decl-attr conformance round-up ────── */
