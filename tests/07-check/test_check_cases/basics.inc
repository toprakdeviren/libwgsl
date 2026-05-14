    /* ── Typed module const ───────────────────────────────────── */
    {
        P p; run(&p, "const N: i32 = 5;\n");
        CHECK(p.ok, "const N: i32 = 5: tc_ok");
        WGSLSymbol *N = sym_for(&p, "N");
        CHECK(N && N->type == p.types.t_i32, "N: i32");
        done(&p);
    }
    {
        P p; run(&p, "const N = 5;\n");
        CHECK(p.ok, "const N = 5 (untyped): tc_ok");
        WGSLSymbol *N = sym_for(&p, "N");
        CHECK(N && N->type == p.types.t_abstract_int, "N: AbstractInt");
        done(&p);
    }
    {
        P p; run(&p, "const PI = 3.14;\n");
        CHECK(p.ok, "const PI (untyped float): tc_ok");
        WGSLSymbol *S = sym_for(&p, "PI");
        CHECK(S && S->type == p.types.t_abstract_float, "PI: AbstractFloat");
        done(&p);
    }
    {
        P p; run(&p, "const X: u32 = 256;\n");
        CHECK(p.ok, "const X: u32: tc_ok");
        WGSLSymbol *X = sym_for(&p, "X");
        CHECK(X && X->type == p.types.t_u32, "X: u32");
        done(&p);
    }
    {
        P p; run(&p,
            "alias U = u32;\n"
            "const X: U = -1;\n");
        CHECK(!p.ok, "const X: alias u32 = -1: must error");
        CHECK(has_error_with(&p.diag, "out of range for u32"),
              "alias-typed const range diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const C = 5;\n"
            "fn f() { let x = C; }\n");
        CHECK(p.ok, "let from untyped const: tc_ok");
        WGSLSymbol *x = sym_for(&p, "x");
        CHECK(x && x->type == p.types.t_i32,
              "let x = const AbstractInt materializes to i32");
        done(&p);
    }

    /* ── Forward-ref module const / var ───────────────────────── */
    {
        P p; run(&p,
            "const A = B;\n"
            "const B: i32 = 5;\n");
        CHECK(p.ok, "const A = B (forward ref): tc_ok");
        WGSLSymbol *A = sym_for(&p, "A");
        CHECK(A && A->type == p.types.t_i32, "A: i32 (from forward ref B)");
        done(&p);
    }
    {
        P p; run(&p,
            "const A = B;\n"
            "const B = A;\n");
        CHECK(!p.ok, "cyclic const: must error");
        CHECK(has_error_with(&p.diag, "cyclic dependency"),
              "cyclic dependency diagnostic");
        done(&p);
    }
    {
        /* §7.3 — `var<private>` initializer must be a const- or
         * override-expression; referencing another `var` is rejected
         * (this test used to assert acceptance pre-§7.3 init gate). */
        P p; run(&p,
            "var<private> X = Y;\n"
            "var<private> Y = 10u;\n");
        CHECK(!p.ok, "var<private> X = (var) Y: must error");
        CHECK(has_error_with(&p.diag, "var<private> initializer must be a const- or override-expression"),
              "var<private> non-const init diag");
        done(&p);
    }
    {
        /* §7.3 — `var<private>` with a const-expr init is OK; with an
         * override-ident is also OK (override-expression). */
        P p; run(&p,
            "const K: u32 = 42;\n"
            "override O: u32 = 1;\n"
            "var<private> X: u32 = K;\n"
            "var<private> Y: u32 = O + K;\n"
            "var<private> Z: u32 = 7u;\n");
        CHECK(p.ok, "var<private> with const/override init: ok");
        done(&p);
    }

    /* ── Vec / mat / array type-specs ─────────────────────────── */
    {
        P p; run(&p, "alias UV = vec2<f32>;\n");
        CHECK(p.ok, "alias UV = vec2<f32>: tc_ok");
        WGSLSymbol *UV = sym_for(&p, "UV");
        CHECK(UV && UV->type, "UV: typed");
        CHECK(UV->type->kind  == WGSL_TYPE_VEC,           "UV: vec");
        CHECK(UV->type->width == 2,                       "UV: width 2");
        CHECK(UV->type->ref   == p.types.t_f32,           "UV: elem f32");
        done(&p);
    }
    {
        P p; run(&p, "alias M = mat4x4<f32>;\n");
        CHECK(p.ok, "alias M = mat4x4<f32>: tc_ok");
        WGSLSymbol *M = sym_for(&p, "M");
        CHECK(M && M->type, "M: typed");
        CHECK(M->type->kind == WGSL_TYPE_MAT, "M: mat");
        CHECK(M->type->width == 4 && M->type->rows == 4, "M: 4x4");
        CHECK(M->type->ref == p.types.t_f32,             "M: elem f32");
        done(&p);
    }
    {
        P p; run(&p, "alias Buf = array<f32, 8>;\n");
        CHECK(p.ok, "alias Buf = array<f32, 8>: tc_ok");
        WGSLSymbol *Buf = sym_for(&p, "Buf");
        CHECK(Buf && Buf->type && Buf->type->kind == WGSL_TYPE_ARRAY,
              "Buf: array");
        CHECK(Buf->type->array_len == 8,            "Buf: length 8");
        CHECK(Buf->type->ref      == p.types.t_f32, "Buf: elem f32");
        done(&p);
    }
    {
        P p; run(&p,
            "const N: u32 = 4u;\n"
            "alias Buf = array<f32, N>;\n");
        CHECK(p.ok, "alias array<f32, N>: tc_ok");
        WGSLSymbol *Buf = sym_for(&p, "Buf");
        CHECK(Buf && Buf->type && Buf->type->array_len == 4,
              "Buf: length 4 (from const N)");
        done(&p);
    }
    {
        P p; run(&p, "alias Run = array<u32>;\n");
        CHECK(p.ok, "alias array<u32> (runtime): tc_ok");
        WGSLSymbol *R = sym_for(&p, "Run");
        CHECK(R && R->type && R->type->array_len == 0,
              "Run: runtime-sized");
        done(&p);
    }

    /* ── Predeclared alias ───────────────────────────────────── */
    {
        P p; run(&p, "alias V = vec3f;\n");
        CHECK(p.ok, "alias V = vec3f: tc_ok");
        WGSLSymbol *V = sym_for(&p, "V");
        CHECK(V && V->type && V->type->kind == WGSL_TYPE_VEC &&
              V->type->width == 3 && V->type->ref == p.types.t_f32,
              "V: vec3<f32>");
        done(&p);
    }

    /* ── Struct ──────────────────────────────────────────────── */
    {
        P p; run(&p,
            "struct Vertex {\n"
            "  pos: vec3<f32>,\n"
            "  uv:  vec2<f32>,\n"
            "}\n");
        CHECK(p.ok, "struct: tc_ok");
        WGSLSymbol *V = sym_for(&p, "Vertex");
        CHECK(V && V->type && V->type->kind == WGSL_TYPE_STRUCT,
              "Vertex: STRUCT type");
        done(&p);
    }

    /* ── Function param + return ─────────────────────────────── */
    {
        P p; run(&p,
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n");
        CHECK(p.ok, "add: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        CHECK(a && a->type == p.types.t_i32, "a: i32 param");
        CHECK(b && b->type == p.types.t_i32, "b: i32 param");
        /* Find `a + b` and verify it types to i32. */
        int skip0 = 0; const WGSLNode *id_a = find_ident(p.ast.root, &p.src, "a", &skip0);
        CHECK(id_a, "found ident 'a'");
        if (id_a) {
            CHECK(wgsl_typecheck_type_of(&p.tc, id_a) == p.types.t_i32,
                  "ident 'a' typed i32");
        }
        done(&p);
    }

    /* ── var produces ref ────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: i32 = 1;\n"
            "  let r = v + 1;\n"
            "}\n");
        CHECK(p.ok, "var v: tc_ok");
        WGSLSymbol *v = sym_for(&p, "v");
        CHECK(v && v->type == p.types.t_i32, "v: i32");
        int skip0 = 0;
        const WGSLNode *use_v = find_ident(p.ast.root, &p.src, "v", &skip0);
        CHECK(use_v, "found ident 'v'");
        if (use_v) {
            CHECK(wgsl_typecheck_is_ref(&p.tc, use_v),
                  "var ident 'v' is_ref=1");
            WGSLTypeInfo *vt = wgsl_typecheck_type_of(&p.tc, use_v);
            CHECK(vt && vt->kind == WGSL_TYPE_REF &&
                  vt->ref == p.types.t_i32 &&
                  (vt->flags & 0xFFu) == WGSL_AS_FUNCTION &&
                  ((vt->flags >> 8) & 0xFFu) == WGSL_ACCESS_READ_WRITE,
                  "var ident 'v' type is ref<function, i32, read_write>");
        }
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  var v: vec3<f32>;\n"
            "  v.x = 1.0;\n"
            "}\n");
        CHECK(p.ok, "single swizzle ref structural type: tc_ok");
        const WGSLNode *m = find_kind(p.ast.root, WGSL_NODE_EXPR_MEMBER);
        CHECK(m, "found member expression");
        if (m) {
            WGSLTypeInfo *mt = wgsl_typecheck_type_of(&p.tc, m);
            CHECK(mt && mt->kind == WGSL_TYPE_REF &&
                  mt->ref == p.types.t_f32 &&
                  (mt->flags & 0xFFu) == WGSL_AS_FUNCTION &&
                  ((mt->flags >> 8) & 0xFFu) == WGSL_ACCESS_READ_WRITE,
                  "single swizzle type is ref<function, f32, read_write>");
        }
        done(&p);
    }

    /* ── Implicit numeric promotion ─────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 + 2.5;\n"
            "}\n");
        CHECK(p.ok, "1 + 2.5: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        CHECK(a && a->type == p.types.t_f32,
              "a: AbstractInt+AbstractFloat → f32 (concretised)");
        done(&p);
    }
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 + 2i;\n"
            "}\n");
        CHECK(p.ok, "1 + 2i: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        CHECK(a && a->type == p.types.t_i32, "a: i32 (AbstractInt → i32)");
        done(&p);
    }

    /* ── Comparison / logical / bitwise types ───────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = 1 < 2;\n"
            "  let b = a && true;\n"
            "  let c = 0xFFu | 0x0Fu;\n"
            "}\n");
        CHECK(p.ok, "compare/logical/bitwise: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        WGSLSymbol *c = sym_for(&p, "c");
        CHECK(a && a->type == p.types.t_bool, "a: bool");
        CHECK(b && b->type == p.types.t_bool, "b: bool");
        CHECK(c && c->type == p.types.t_u32,  "c: u32");
        done(&p);
    }

    /* ── Unary ───────────────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() {\n"
            "  let a = -1;\n"
            "  let b = !true;\n"
            "  let c = ~0u;\n"
            "}\n");
        CHECK(p.ok, "unary: tc_ok");
        WGSLSymbol *a = sym_for(&p, "a");
        WGSLSymbol *b = sym_for(&p, "b");
        WGSLSymbol *c = sym_for(&p, "c");
        CHECK(a && a->type == p.types.t_i32,  "a: -1 → i32 (concretised)");
        CHECK(b && b->type == p.types.t_bool, "b: bool");
        CHECK(c && c->type == p.types.t_u32,  "c: u32");
        done(&p);
    }

    /* ── Errors ───────────────────────────────────────────────── */
    {
        P p; run(&p, "fn f() { let x = -1u; }\n");
        CHECK(!p.ok, "unary minus on u32: must error");
        CHECK(has_error_with(&p.diag, "unary minus"),
              "u32 unary minus diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1i + 1u; }\n");
        CHECK(!p.ok, "i32 + u32: must error");
        CHECK(has_error_with(&p.diag, "no implicit conversion"),
              "concrete-mismatch diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1 && 2; }\n");
        CHECK(!p.ok, "logical with non-bool: must error");
        CHECK(has_error_with(&p.diag, "logical operator requires bool"),
              "logical-non-bool diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = !1; }\n");
        CHECK(!p.ok, "!1 (non-bool): must error");
        CHECK(has_error_with(&p.diag, "logical not requires"),
              "!non-bool diag");
        done(&p);
    }
    {
        P p; run(&p, "fn f() { let x = 1.5 & 2; }\n");
        CHECK(!p.ok, "1.5 & 2: must error");
        CHECK(has_error_with(&p.diag, "bitwise"),
              "bitwise-non-int diag");
        done(&p);
    }

    /* ── Texture / sampler type-specs (§6.5) ─────────────────── */
    {
        /* Bare `sampler` resolves to a real type, not "is not a type". */
        P p; run(&p,
            "@group(0) @binding(0) var s : sampler;\n");
        CHECK(p.ok, "var s : sampler: tc_ok");
        CHECK(!has_error_with(&p.diag, "is not a type"),
              "sampler must not trip 'is not a type'");
        WGSLSymbol *s = sym_for(&p, "s");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_SAMPLER,
              "s: typed as sampler");
        CHECK(s && s->type && s->type->width == 0,
              "s: non-comparison sampler");
        done(&p);
    }
    {
        P p; run(&p,
            "@group(0) @binding(0) var sc : sampler_comparison;\n");
        CHECK(p.ok, "var sc : sampler_comparison: tc_ok");
        WGSLSymbol *s = sym_for(&p, "sc");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_SAMPLER &&
              s->type->width == 1,
              "sc: comparison sampler");
        done(&p);
    }
    {
        /* Sampled texture: texture_2d<f32>. */
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<f32>;\n");
        CHECK(p.ok, "var t : texture_2d<f32>: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "texture_2d<f32> must not trip 'has no type yet'");
        WGSLSymbol *s = sym_for(&p, "t");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_TEXTURE,
              "t: typed as texture");
        CHECK(s && s->type && s->type->width == WGSL_TEX_DIM_2D,
              "t: dim = 2d");
        CHECK(s && s->type && s->type->ref == p.types.t_f32,
              "t: sampled elem = f32");
        done(&p);
    }
    {
        /* Invalid sampled texture: texture_2d<bool>. */
        P p; run(&p,
            "@group(0) @binding(0) var t : texture_2d<bool>;\n");
        CHECK(!p.ok, "texture_2d<bool>: must error");
        CHECK(has_error_with(&p.diag, "sampled texture type must be f32, i32, or u32"),
              "texture_2d<bool> diagnostic");
        done(&p);
    }
    {
        /* ptr type specifier */
        P p; run(&p, "fn f(x: ptr<function, i32>) {}\n");
        CHECK(p.ok, "ptr<function, i32> parameter: tc_ok");
        CHECK(!has_error_with(&p.diag, "has no type yet"),
              "ptr param must not trip 'has no type yet'");
        WGSLSymbol *s = sym_for(&p, "x");
        CHECK(s && s->type && s->type->kind == WGSL_TYPE_PTR,
              "x: typed as ptr");
        CHECK(s && s->type && (s->type->flags & 0xFF) == WGSL_AS_FUNCTION,
              "x: address space function");
        done(&p);
    }
    {
        /* Address-of and indirection */
        P p; run(&p, "fn f() { var i: i32; let p = &i; let v = *p; }\n");
        CHECK(p.ok, "address-of and indirection: tc_ok");
        WGSLSymbol *sp = sym_for(&p, "p");
        CHECK(sp && sp->type && sp->type->kind == WGSL_TYPE_PTR, "p is ptr");
        WGSLSymbol *sv = sym_for(&p, "v");
        CHECK(sv && sv->type && sv->type->kind == WGSL_TYPE_I32, "v is i32");
        done(&p);
    }
    {
        /* §11.2 — pointer args must match address space, store type,
         * and access mode, not just the outer `ptr` kind. */
        P p; run(&p,
            "var<private> g: i32;\n"
            "fn take_fn(p: ptr<function, i32>) {}\n"
            "fn f() { take_fn(&g); }\n");
        CHECK(!p.ok, "ptr arg AS mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_fn'"),
              "ptr AS mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn take_i32(p: ptr<function, i32>) {}\n"
            "fn f() { var u: u32; take_i32(&u); }\n");
        CHECK(!p.ok, "ptr arg store-type mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_i32'"),
              "ptr store-type mismatch diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<storage, read> s: i32;\n"
            "fn take_rw(p: ptr<storage, i32, read_write>) {}\n"
            "fn f() { take_rw(&s); }\n");
        CHECK(!p.ok, "ptr arg access-mode mismatch: must error");
        CHECK(has_error_with(&p.diag, "argument 1 of 'take_rw'"),
              "ptr access-mode mismatch diag");
        done(&p);
    }
    {
        /* Assignment to non-reference */
        P p; run(&p, "fn f() { let x = 1; x = 2; }\n");
        CHECK(!p.ok, "assign to let: must error");
        CHECK(has_error_with(&p.diag, "left-hand side must be a reference"),
              "assign to value diag");
        done(&p);
    }
    {
        /* Assignment to read-only reference */
        P p; run(&p, "struct S { a: i32 }\n@group(0) @binding(0) var<uniform> u: S;\nfn f() { u.a = 1; }\n");
        CHECK(!p.ok, "assign to uniform: must error");
        CHECK(has_error_with(&p.diag, "cannot assign to read-only reference"),
              "assign to read-only diag");
        done(&p);
    }
    {
        /* Pointer in struct */
        P p; run(&p, "struct S { p: ptr<function, i32> }\n");
        CHECK(!p.ok, "ptr in struct: must error");
        CHECK(has_error_with(&p.diag, "struct member type cannot be a ptr"),
              "ptr in struct diag");
        done(&p);
    }
    {
        /* Override-sized arrays cannot be nested in structs. */
        P p; run(&p, "override N: u32;\nstruct S { a: array<u32, N> }\n");
        CHECK(!p.ok, "override-sized array in struct: must error");
        CHECK(has_error_with(&p.diag, "must not contain an override-sized array"),
              "override array struct-member diag");
        done(&p);
    }
    /* ── §7 Variable and Value Declarations ──────────────────────── */
