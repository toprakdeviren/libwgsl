    {
        /* §7.2.2 @id range — now wired through consteval so the
         * literal's span is read correctly (the previous version
         * pulled offset/length out of LITERAL_INT's payload[0],
         * which actually carries suffix bits). */
        P p; run(&p, "@id(70000) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(70000): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id range diag");
        done(&p);
    }
    {
        /* §7.2.2 @id boundary — 65535 must be accepted. */
        P p; run(&p, "@id(65535) override X: i32 = 1;\n");
        CHECK(p.ok, "@id(65535): tc_ok");
        done(&p);
    }
    {
        /* §7.2.2 @id uniqueness across overrides. */
        P p; run(&p,
            "@id(1) override A: i32 = 1;\n"
            "@id(1) override B: i32 = 2;\n");
        CHECK(!p.ok, "duplicate @id(1): must error");
        CHECK(has_error_with(&p.diag, "must be unique across overrides"),
              "@id uniqueness diag");
        done(&p);
    }
    {
        /* §12.8 @id must be non-negative; previously a bare `@id(-1)`
         * silently slipped past the typecker. */
        P p; run(&p, "@id(-1) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(-1): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id negative diag");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing another override is OK. */
        P p; run(&p,
            "override A: i32 = 1;\n"
            "override B: i32 = A * 2 + 5;\n");
        CHECK(p.ok, "override init referencing override: ok");
        done(&p);
    }
    {
        P p; run(&p, "override X;\n");
        CHECK(!p.ok, "override without type or initializer: must error");
        CHECK(has_error_with(&p.diag, "must specify a type or an initializer"),
              "override missing type/init diag");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing const is OK. */
        P p; run(&p,
            "const K: i32 = 7;\n"
            "override B: i32 = K + 1;\n");
        CHECK(p.ok, "override init referencing const: ok");
        done(&p);
    }
    {
        /* §17.9 — pack4x8snorm requires vec4<f32>. */
        P p; run(&p,
            "fn f() { let x = pack4x8snorm(vec4<i32>(0)); }\n");
        CHECK(!p.ok, "pack4x8snorm with vec4<i32>: must error");
        CHECK(has_error_with(&p.diag, "'pack4x8snorm' requires a vec4<f32> argument"),
              "pack4x8snorm shape diag");
        done(&p);
    }
    {
        /* §17.9 — pack4xI8 requires vec4<i32>. */
        P p; run(&p,
            "fn f() { let x = pack4xI8(vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "pack4xI8 with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "'pack4xI8' requires a vec4<i32> argument"),
              "pack4xI8 shape diag");
        done(&p);
    }
    {
        /* §17.9 — pack2x16float requires vec2<f32>. */
        P p; run(&p,
            "fn f() { let x = pack2x16float(vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "pack2x16float with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "'pack2x16float' requires a vec2<f32> argument"),
              "pack2x16float shape diag");
        done(&p);
    }
    {
        /* §17.10 — unpack4xU8 requires u32. */
        P p; run(&p,
            "fn f() { let x = unpack4xU8(1.5); }\n");
        CHECK(!p.ok, "unpack4xU8 with f32: must error");
        CHECK(has_error_with(&p.diag, "'unpack4xU8' requires a u32 argument"),
              "unpack4xU8 shape diag");
        done(&p);
    }
    {
        /* §17.9/10 happy paths. */
        P p; run(&p,
            "fn f() {\n"
            "  let a = pack4x8snorm(vec4<f32>(0.5));\n"
            "  let b = pack4xI8(vec4<i32>(1));\n"
            "  let c = pack4xU8(vec4<u32>(2u));\n"
            "  let d = pack2x16float(vec2<f32>(0.25));\n"
            "  let e = unpack4x8snorm(1u);\n"
            "  let g = unpack4xI8(2u);\n"
            "  let h = unpack2x16float(3u);\n"
            "}\n");
        CHECK(p.ok, "pack/unpack happy paths: ok");
        done(&p);
    }
    {
        /* §17.10 — return types are correct (regression: unpack4xI8
         * was BR_VEC4F before, now BR_VEC4I). */
        P p; run(&p,
            "fn f() {\n"
            "  let v: vec4<i32> = unpack4xI8(1u);\n"
            "  let u: vec4<u32> = unpack4xU8(2u);\n"
            "  let w: vec2<f32> = unpack2x16snorm(3u);\n"
            "}\n");
        CHECK(p.ok, "unpack return-type fixes: ok");
        done(&p);
    }
    {
        /* §17.8 — atomic ptr arg validation: passing a non-atomic var
         * is rejected. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: i32;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let r = atomicAdd(&a, 1);\n"
            "}\n");
        CHECK(!p.ok, "atomicAdd on non-atomic: must error");
        CHECK(has_error_with(&p.diag, "requires a pointer to atomic<T>"),
              "atomicAdd ptr-arg diag");
        done(&p);
    }
    {
        /* §17.8 — value-arg type must match atomic's element T. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  atomicStore(&a, 1.5);\n"
            "}\n");
        CHECK(!p.ok, "atomicStore u32 with f32: must error");
        CHECK(has_error_with(&p.diag, "value argument has type"),
              "atomicStore type diag");
        done(&p);
    }
    {
        /* §17.8 — atomicLoad arg-count: 1 arg expected. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a, 1u);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad with extra arg: must error");
        CHECK(has_error_with(&p.diag, "atomicLoad' expects 1 argument"),
              "atomicLoad arity diag");
        done(&p);
    }
    {
        /* §17.8 — atomic builtins require read_write atomic pointers. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read> a: atomic<u32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad on storage/read atomic: must error");
        CHECK(has_error_with(&p.diag, "ptr<storage, atomic<T>, read_write>"),
              "atomic storage-read access diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic pointer store types are only
         * instantiable in workgroup or storage address spaces. */
        P p; run(&p,
            "fn f(p: ptr<private, atomic<i32>>) {\n"
            "  let x = atomicLoad(p);\n"
            "}\n");
        CHECK(!p.ok, "atomicLoad on ptr<private, atomic<i32>>: must error");
        CHECK(has_error_with(&p.diag, "atomic requires the workgroup or storage"),
              "atomic private pointer diag");
        done(&p);
    }
    {
        /* §6.2.9 — fixed-size arrays of different lengths are
         * structurally distinct (regression — was previously trusted
         * via the kind-equal fallthrough in `init_type_mismatch`). */
        P p; run(&p,
            "var<private> x: array<f32, 4>;\n"
            "var<private> y: array<f32, 8>;\n"
            "fn f() { x = y; }\n");
        CHECK(!p.ok, "array<f32,4> = array<f32,8>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted to the assignment target"),
              "fixed-size array mismatch diag");
        done(&p);
    }
    {
        /* §6.2.13 — override-sized arrays are only valid as workgroup
         * memory views, not private/function/storage store types. */
        P p; run(&p,
            "override N: u32 = 4;\n"
            "var<private> x: array<f32, N>;\n");
        CHECK(!p.ok, "var<private> array<f32,N>: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array placement diag");
        done(&p);
    }
    {
        /* The same placement rule applies through aliases. */
        P p; run(&p,
            "override N: u32 = 4;\n"
            "alias T = array<f32, N>;\n"
            "fn f() { var x: T; }\n");
        CHECK(!p.ok, "function var alias to override array: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array alias placement diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override N: u32 = 4;\n"
            "fn f(p: ptr<private, array<u32, N>>) { }\n");
        CHECK(!p.ok, "ptr<private, override array>: must error");
        CHECK(has_error_with(&p.diag, "requires the workgroup address space"),
              "override array ptr placement diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override N: u32 = 4;\n"
            "fn f() { let x = array<u32, N>(); }\n");
        CHECK(!p.ok, "construct override-sized array: must error");
        CHECK(has_error_with(&p.diag, "cannot be constructed"),
              "override array constructor diag");
        done(&p);
    }
    {
        /* §6.2.9 — override-sized arrays are shader-valid even when
         * the default value is not a valid fixed array count; pipeline
         * specialization can supply the generation-time length. */
        P p; run(&p,
            "override N: i32 = 0;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "array<f32, override=0>: shader-valid");
        done(&p);
    }
    {
        /* §6.2.9 — override default > 0: ok. */
        P p; run(&p,
            "override N: i32 = 4;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "override default > 0: ok");
        done(&p);
    }
    {
        /* §6.2.9 — no default → check skipped (pipeline-time value). */
        P p; run(&p,
            "override N: u32;\n"
            "var<workgroup> a: array<f32, N>;\n");
        CHECK(p.ok, "override no default: ok (skip static check)");
        done(&p);
    }
    {
        /* §6.2.10 — duplicate struct member name. */
        P p; run(&p,
            "struct S { x: i32, x: f32 }\n");
        CHECK(!p.ok, "dup struct member name: must error");
        CHECK(has_error_with(&p.diag,
            "struct member name 'x' must be unique within the struct"),
              "dup-member-name diag");
        done(&p);
    }
    {
        /* §17.8 — happy paths covering Load / Store / RMW / CX. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage,read_write> a: atomic<i32>;\n"
            "@compute @workgroup_size(1) fn cs() {\n"
            "  let x = atomicLoad(&a);\n"
            "  atomicStore(&a, 1);\n"
            "  let y = atomicAdd(&a, 1);\n"
            "  let z = atomicExchange(&a, 5);\n"
            "  let w = atomicCompareExchangeWeak(&a, 0, 1);\n"
            "}\n");
        CHECK(p.ok, "atomic happy paths: ok");
        done(&p);
    }
    {
        /* §7.2.2 — override init referencing a `let`/`var` is rejected. */
        P p; run(&p,
            "var<private> v: i32 = 3;\n"
            "override B: i32 = v + 1;\n");
        CHECK(!p.ok, "override init referencing var: must error");
        CHECK(has_error_with(&p.diag, "override initializer must be an override-expression"),
              "override-expr diag");
        done(&p);
    }
    {
        /* §7.2.1 — `const` type must be constructible.  Use a struct
         * containing a runtime-sized array (deep non-constructible) to
         * avoid the pre-existing atomic-init crash path. */
        P p; run(&p,
            "struct S { a: array<f32> }\n"
            "const c: S;\n");
        CHECK(!p.ok, "const non-constructible: must error");
        CHECK(has_error_with(&p.diag, "'const' type must be constructible"),
              "const constructible diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic in non-shared AS is rejected. */
        P p; run(&p, "var<private> a: atomic<i32>;\n");
        CHECK(!p.ok, "var<private> atomic: must error");
        CHECK(has_error_with(&p.diag,
                  "atomic store type requires the workgroup or storage"),
              "atomic-AS pairing diag");
        done(&p);
    }
    {
        /* §6.2.8 / §14.3 — atomic nested in an array has the same AS
         * restriction. */
        P p; run(&p, "var<private> a: array<atomic<i32>, 4>;\n");
        CHECK(!p.ok, "var<private> array<atomic<i32>>: must error");
        CHECK(has_error_with(&p.diag,
                  "atomic store type requires the workgroup or storage"),
              "array-atomic-AS pairing diag");
        done(&p);
    }
    {
        /* Atomic in workgroup AS is legal. */
        P p; run(&p, "var<workgroup> a: atomic<i32>;\n");
        CHECK(p.ok, "var<workgroup> atomic: tc_ok");
        done(&p);
    }
    {
        /* Storage atomics require read_write access. */
        P p; run(&p,
            "@group(0) @binding(0) var<storage> a: atomic<u32>;\n");
        CHECK(!p.ok, "var<storage> atomic default access: must error");
        CHECK(has_error_with(&p.diag, "requires read_write access mode"),
              "storage atomic access-mode diag");
        done(&p);
    }
    {
        /* Runtime-sized arrays are only valid in storage store types. */
        P p; run(&p, "var<workgroup> a: array<i32>;\n");
        CHECK(!p.ok, "var<workgroup> runtime array: must error");
        CHECK(has_error_with(&p.diag,
                  "runtime-sized array store type requires the storage"),
              "runtime-array-AS diag");
        done(&p);
    }
    {
        /* The runtime-sized-array restriction is deep through structs. */
        P p; run(&p,
            "struct S { a: array<i32> }\n"
            "var<private> s: S;\n");
        CHECK(!p.ok, "var<private> struct with runtime array: must error");
        CHECK(has_error_with(&p.diag,
                  "runtime-sized array store type requires the storage"),
              "struct-runtime-array-AS diag");
        done(&p);
    }
    {
        /* §12.2 / §12.7 — @group / @binding on a private var is
         * rejected (these attributes are for resource variables only). */
        P p; run(&p, "@group(0) var<private> x: i32;\n");
        CHECK(!p.ok, "@group on var<private>: must error");
        CHECK(has_error_with(&p.diag,
                  "@group is only allowed on resource variables"),
              "@group-on-non-resource diag");
        done(&p);
    }
    {
        /* @group / @binding on a workgroup var likewise rejected. */
        P p; run(&p, "@binding(0) var<workgroup> wg: i32;\n");
        CHECK(!p.ok, "@binding on var<workgroup>: must error");
        CHECK(has_error_with(&p.diag,
                  "@binding is only allowed on resource variables"),
              "@binding-on-non-resource diag");
        done(&p);
    }
    {
        /* @group / @binding on a uniform / storage / texture var stays
         * legal (the long-standing happy path). */
        P p; run(&p,
            "@group(0) @binding(0) var t: texture_2d<f32>;\n"
            "struct S { x: f32 }\n"
            "@group(0) @binding(1) var<uniform> u: S;\n"
            "@group(0) @binding(2) var<storage> s: array<f32>;\n");
        CHECK(p.ok, "@group/@binding on resources: tc_ok");
        done(&p);
    }

