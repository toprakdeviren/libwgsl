    {
        /* §7 (general): var must have type or initializer */
        P p; run(&p, "fn f() { var x; }\n");
        CHECK(!p.ok, "var without type or init: must error");
        CHECK(has_error_with(&p.diag, "must specify a type or an initializer"),
              "var no type/init diag");
        done(&p);
    }
    {
        /* §7 (general): init type must be convertible to declared type */
        P p; run(&p, "fn f() { var x: i32 = 1.5; }\n");
        CHECK(!p.ok, "i32 = 1.5: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "i32 = float mismatch diag");
        done(&p);
    }
    {
        /* §7 (general): composite init compatibility is structural:
         * same `vec` kind alone is not enough. */
        P p; run(&p, "fn f() { let x: vec2<bool> = vec2<u32>(1u, 2u); }\n");
        CHECK(!p.ok, "vec2<bool> = vec2<u32>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "vec element mismatch diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x: vec2<i32> = vec3<i32>(1, 2, 3); }\n");
        CHECK(!p.ok, "vec2<i32> = vec3<i32>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "vec width mismatch diag");
        done(&p);
    }
    {
        /* §7.2.2: override must be concrete scalar */
        P p; run(&p, "override X: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);\n");
        CHECK(!p.ok, "override vec3: must error");
        CHECK(has_error_with(&p.diag, "override type must be a scalar"),
              "override vec3 diag");
        done(&p);
    }
    {
        /* §7.2.2: @id(N) must be in [0, 65535] */
        P p; run(&p, "@id(70000) override X: i32 = 1;\n");
        CHECK(!p.ok, "@id(70000): must error");
        CHECK(has_error_with(&p.diag, "@id value must be in [0, 65535]"),
              "@id range diag");
        done(&p);
    }
    {
        /* §7.3: var<storage, write> forbidden */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<storage, write> s: S;\n");
        CHECK(!p.ok, "storage write: must error");
        CHECK(has_error_with(&p.diag, "do not support 'write' access mode"),
              "storage write diag");
        done(&p);
    }
    {
        /* §7.3: var template list accepts address space and access only. */
        P p; run(&p, "@group(0) @binding(0) var<storage, read, read> x: i32;\n");
        CHECK(!p.ok, "var with extra template arg: must error");
        CHECK(has_error_with(&p.diag, "at most address space and access mode"),
              "var template arity diag");
        done(&p);
    }
    {
        /* §7.3: access mode only for storage */
        P p; run(&p, "var<private, read> x: i32;\n");
        CHECK(!p.ok, "private read: must error");
        CHECK(has_error_with(&p.diag, "access mode can only be specified for 'storage'"),
              "private access mode diag");
        done(&p);
    }
    {
        /* §7.3: initializer disallowed for var<uniform> */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<uniform> u: S = S(1);\n");
        CHECK(!p.ok, "uniform with init: must error");
        CHECK(has_error_with(&p.diag, "cannot have an initializer"),
              "uniform init diag");
        done(&p);
    }
    {
        /* §7.3: handle address-space variables cannot have initializers. */
        P p; run(&p,
            "@group(0) @binding(0) var s0: sampler;\n"
            "@group(0) @binding(1) var s1: sampler = s0;\n");
        CHECK(!p.ok, "sampler with init: must error");
        CHECK(has_error_with(&p.diag, "cannot have an initializer"),
              "handle init diag");
        done(&p);
    }
    {
        /* §7.3: var<function> at module scope forbidden */
        P p; run(&p, "var<function> x: i32 = 0;\n");
        CHECK(!p.ok, "var<function> at module: must error");
        CHECK(has_error_with(&p.diag, "can only appear inside a function body"),
              "var<function> module scope diag");
        done(&p);
    }
    {
        /* §7.3: var<private> inside function forbidden */
        P p; run(&p, "fn f() { var<private> x: i32 = 0; }\n");
        CHECK(!p.ok, "var<private> in function: must error");
        CHECK(has_error_with(&p.diag, "must be at module scope"),
              "var<private> fn scope diag");
        done(&p);
    }
    {
        /* §7.3: texture var inside function forbidden */
        P p; run(&p, "fn f() { var t: texture_2d<f32>; }\n");
        CHECK(!p.ok, "texture in fn: must error");
        CHECK(has_error_with(&p.diag, "texture and sampler variables must be at module scope"),
              "texture fn scope diag");
        done(&p);
    }
    {
        /* Depth texture: no template args. */
        P p; run(&p,
            "@group(0) @binding(0) var d : texture_depth_2d;\n");
        CHECK(p.ok, "var d : texture_depth_2d: tc_ok");
        WGSLSymbol *s = sym_for(&p, "d");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE &&
              s->type->width == WGSL_TEX_DIM_DEPTH_2D,
              "d: depth_2d");
        done(&p);
    }
    {
        /* texture_external: no template args, predeclared shape. */
        P p; run(&p,
            "@group(0) @binding(0) var e : texture_external;\n");
        CHECK(p.ok, "var e : texture_external: tc_ok");
        WGSLSymbol *s = sym_for(&p, "e");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE &&
              s->type->width == WGSL_TEX_DIM_EXTERNAL,
              "e: external");
        done(&p);
    }
    {
        /* Storage texture: <texel_format, access>. */
        P p; run(&p,
            "@group(0) @binding(0) "
            "var st : texture_storage_2d<rgba8unorm, write>;\n");
        CHECK(p.ok, "var st : texture_storage_2d<rgba8unorm, write>: tc_ok");
        CHECK(!has_error_with(&p.diag, "undeclared identifier 'rgba8unorm'"),
              "rgba8unorm must be recognised");
        WGSLSymbol *s = sym_for(&p, "st");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE,
              "st: typed as texture");
        CHECK(s && s->type && s->type->width == WGSL_TEX_DIM_STORAGE_2D,
              "st: dim = storage_2d");
        CHECK(s && s->type && s->type->array_len == WGSL_TEX_FORMAT_RGBA8UNORM,
              "st: format = rgba8unorm");
        CHECK(s && s->type && s->type->rows == WGSL_ACCESS_WRITE,
              "st: access = write");
        done(&p);
    }
    {
        /* End-to-end: a textured fragment shader argument list must
         * not trip "has no type yet" diagnostics. */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "@fragment fn fs(@location(0) uv : vec2<f32>) "
            "-> @location(0) vec4<f32> {\n"
            "  return textureSampleLevel(t, s, uv, 0.0);\n"
            "}\n");
        CHECK(p.ok, "textured fragment: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "textured-fragment must not trip 'has no type yet'");
        CHECK(!has_error_with(&p.diag, "is not a type"),
              "textured-fragment must not trip 'is not a type'");
        done(&p);
    }
    {
        /* Texture / sampler types are interned: same shape → same ptr. */
        P p; run(&p,
            "@group(0) @binding(0) var a : texture_2d<f32>;\n"
            "@group(0) @binding(1) var b : texture_2d<f32>;\n");
        CHECK(p.ok, "two texture_2d<f32>: tc_ok");
        WGSLSymbol *A = sym_for(&p, "a");
        WGSLSymbol *B = sym_for(&p, "b");
        CHECK(A && B && A->type && A->type == B->type,
              "texture_2d<f32> interned");
        done(&p);
    }
    {
        /* Generated texture metadata now validates coordinate shapes. */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_1d<f32>;\n"
            "fn f() { _ = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureSample(texture_1d, vec2 coords): must error");
        CHECK(has_error_with(&p.diag, "coordinates"),
              "textureSample coordinate shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureSample(texture_2d, vec2 coords): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_f32,
              "textureSample(texture_2d<f32>) -> vec4<f32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<u32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureSample(texture_2d<u32>): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureSample'"),
              "integer textureSample overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<i32>;\n"
            "fn f() { let r = textureSampleLevel(t, s, vec2<f32>(0.0), 0.0); }\n");
        CHECK(!p.ok, "textureSampleLevel(texture_2d<i32>): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureSampleLevel'"),
              "integer textureSampleLevel overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r: f32 = textureSampleCompare(d, sc, vec2<f32>(0.0), 0.5); }\n");
        CHECK(p.ok, "textureSampleCompare(depth_2d): returns f32");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var st : texture_storage_2d<rgba8uint, write>;\n"
            "fn f() { textureStore(st, vec2<i32>(0), vec4<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureStore rgba8uint with vec4<f32>: must error");
        CHECK(has_error_with(&p.diag, "value argument"),
              "textureStore value shape diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(component, texture_2d): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_f32,
              "textureGather(texture_2d<f32>) -> vec4<f32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<i32>;\n"
            "fn f() { let r: vec4<i32> = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(texture_2d<i32>) -> vec4<i32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<u32>;\n"
            "fn f() { let r: vec4<u32> = textureGather(1, t, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(texture_2d<u32>) -> vec4<u32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureGather(4, t, s, vec2<f32>(0.0)); }\n");
        CHECK(!p.ok, "textureGather component 4: must error");
        CHECK(has_error_with(&p.diag, "component value"),
              "textureGather component range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r = textureGather(d, s, vec2<f32>(0.0)); }\n");
        CHECK(p.ok, "textureGather(depth texture): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n"
            "@group(0) @binding(1) var d : texture_depth_2d;\n"
            "fn f() { let r = textureGatherCompare(d, sc, vec2<f32>(0.0), 0.5); }\n");
        CHECK(p.ok, "textureGatherCompare(depth texture): ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0), vec2<i32>(7, -8)); }\n");
        CHECK(p.ok, "textureSample const offset in range: ok");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureSample(t, s, vec2<f32>(0.0), vec2<i32>(8, 0)); }\n");
        CHECK(!p.ok, "textureSample offset out of range: must error");
        CHECK(has_error_with(&p.diag, "offset component"),
              "texture offset range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n"
            "@group(0) @binding(1) var t : texture_2d<f32>;\n"
            "fn f(o : vec2<i32>) { let r = textureSample(t, s, vec2<f32>(0.0), o); }\n");
        CHECK(!p.ok, "textureSample non-const offset: must error");
        CHECK(has_error_with(&p.diag, "offset must be a const-expression"),
              "texture offset const diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_multisampled_2d<i32>;\n"
            "fn f() { let r = textureLoad(t, vec2<i32>(0), 0); }\n");
        CHECK(p.ok, "textureLoad(multisampled_2d<i32>): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 4 && r->type->ref == p.types.t_i32,
              "textureLoad(texture_multisampled_2d<i32>) -> vec4<i32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureLoad(t, vec2<i32>(0), 0.0); }\n");
        CHECK(!p.ok, "textureLoad float mip level: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad exact-row no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_cube<f32>;\n"
            "fn f() { let r = textureLoad(t, vec3<i32>(0), 0); }\n");
        CHECK(!p.ok, "textureLoad(texture_cube): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad cube no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_3d<f32>;\n"
            "fn f() { let r = textureDimensions(t, 0); }\n");
        CHECK(p.ok, "textureDimensions(texture_3d, level): ok");
        WGSLSymbol *r = sym_for(&p, "r");
        CHECK(r && r->type && r->type->kind == WGSL_TYPE_VEC &&
              r->type->width == 3 && r->type->ref == p.types.t_u32,
              "textureDimensions(texture_3d) -> vec3<u32>");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r: vec2<bool> = textureDimensions(t, 0); }\n");
        CHECK(!p.ok, "textureDimensions vec2<u32> -> vec2<bool>: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "textureDimensions declared vec mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n"
            "fn f() { let r = textureDimensions(t, 0.0); }\n");
        CHECK(!p.ok, "textureDimensions float level: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureDimensions'"),
              "textureDimensions exact-row no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var st : texture_storage_2d<rgba8uint, write>;\n"
            "fn f() { let r = textureLoad(st, vec2<i32>(0)); }\n");
        CHECK(!p.ok, "textureLoad(write-only storage texture): must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'textureLoad'"),
              "textureLoad storage access exact-row diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { let r: f32 = subgroupShuffle(0, 0); }\n");
        CHECK(!p.ok, "subgroupShuffle abstract int result -> f32: must error");
        CHECK(has_error_with(&p.diag, "cannot be converted"),
              "subgroupShuffle concrete result mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { let r: i32 = subgroupShuffle(0, 0); }\n");
        CHECK(p.ok, "subgroupShuffle abstract int result materializes to i32");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "fn f() { _ = subgroupShuffleXor(0, i32(0)); }\n");
        CHECK(!p.ok, "subgroupShuffleXor i32 mask: must error");
        CHECK(has_error_with(&p.diag, "no matching overload for 'subgroupShuffleXor'"),
              "subgroupShuffleXor i32 mask no-overload diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const bad: u32 = 128u;\n"
            "fn f() { _ = subgroupShuffleUp(0, bad); }\n");
        CHECK(!p.ok, "subgroupShuffleUp const delta 128: must error");
        CHECK(has_error_with(&p.diag, "second argument must be in [0, 128)"),
              "subgroupShuffleUp range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "var<private> lane: u32;\n"
            "fn f() { _ = subgroupBroadcast(0, lane); }\n");
        CHECK(!p.ok, "subgroupBroadcast runtime id: must error");
        CHECK(has_error_with(&p.diag, "second argument must be a const-expression"),
              "subgroupBroadcast const id diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const lane = 4;\n"
            "fn f() { _ = quadBroadcast(0, lane); }\n");
        CHECK(!p.ok, "quadBroadcast lane 4: must error");
        CHECK(has_error_with(&p.diag, "second argument must be in [0, 4)"),
              "quadBroadcast range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "enable subgroups;\n"
            "const lane = 127u;\n"
            "fn f() { _ = subgroupBroadcast(0, lane); }\n");
        CHECK(p.ok, "subgroupBroadcast const lane 127: ok");
        done(&p);
    }

