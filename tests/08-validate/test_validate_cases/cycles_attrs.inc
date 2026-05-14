    /* ── Call-graph cycles ─────────────────────────────────────── */
    {
        P p; run(&p,
            "fn f() { f(); }\n");
        CHECK(!p.ok, "self-recursion: must error");
        CHECK(has_error_with(&p.diag, "recursion"),
              "self-recursion: 'recursion' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn a() { b(); }\n"
            "fn b() { a(); }\n");
        CHECK(!p.ok, "mutual recursion: must error");
        CHECK(has_error_with(&p.diag, "recursion"),
              "mutual recursion: 'recursion' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "fn leaf() -> i32 { return 1; }\n"
            "fn middle() -> i32 { return leaf(); }\n"
            "fn top() -> i32 { return middle() + leaf(); }\n");
        CHECK(p.ok, "acyclic call graph: ok");
        done(&p);
    }

    /* ── Struct cycles ─────────────────────────────────────────── */
    {
        P p; run(&p,
            "struct S { x: S }\n");
        CHECK(!p.ok, "self-recursive struct: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "self-recursive struct: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct A { b: B }\n"
            "struct B { a: A }\n");
        CHECK(!p.ok, "mutually-recursive structs: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "mutual structs: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct S { arr: array<S, 4> }\n");
        CHECK(!p.ok, "array<S,N> field cycle: must error");
        CHECK(has_error_with(&p.diag, "struct cycle"),
              "array-of-struct cycle: 'struct cycle' diag");
        done(&p);
    }
    {
        P p; run(&p,
            "struct Inner { v: vec3<f32> }\n"
            "struct Outer { i: Inner, w: f32 }\n");
        CHECK(p.ok, "acyclic struct nesting: ok");
        done(&p);
    }

    /* ── Alias cycles ──────────────────────────────────────────── */
    {
        P p; run(&p,
            "alias A = B;\n"
            "alias B = A;\n");
        CHECK(!p.ok, "alias cycle: must error");
        CHECK(has_error_with(&p.diag, "alias cycle"),
              "alias cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "alias A = u32;\n"
            "alias B = A;\n"
            "alias C = B;\n");
        CHECK(p.ok, "alias chain: ok");
        done(&p);
    }

    /* ── Const cycles ──────────────────────────────────────────── */
    {
        P p; run(&p,
            "const A: i32 = A + 1;\n");
        CHECK(!p.ok, "self-cycle const: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "const cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const A: i32 = B + 1;\n"
            "const B: i32 = A * 2;\n");
        CHECK(!p.ok, "mutual const cycle: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "mutual const cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "const A: i32 = 2;\n"
            "const B: i32 = A + 1;\n"
            "const C: i32 = A * B;\n");
        CHECK(p.ok, "acyclic const chain: ok");
        done(&p);
    }

    /* ── Override cycles ───────────────────────────────────────── */
    {
        P p; run(&p,
            "override A: f32 = B + 1.0;\n"
            "override B: f32 = A * 2.0;\n");
        CHECK(!p.ok, "mutual override cycle: must error");
        CHECK(has_error_with(&p.diag, "override cycle"),
              "override cycle diag");
        done(&p);
    }
    {
        P p; run(&p,
            "override A: f32 = 1.0;\n"
            "override B: f32 = A + 1.0;\n");
        CHECK(p.ok, "acyclic override chain: ok");
        done(&p);
    }

    /* ── §5 Module-`var` cycles ───────────────────────────────── */
    {
        P p; run(&p,
            "var<private> a: i32 = b;\n"
            "var<private> b: i32 = a;\n");
        CHECK(!p.ok, "mutual module-var cycle: must error");
        CHECK(has_error_with(&p.diag, "module var cycle"),
              "mutual module-var cycle: diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> a: i32 = a;\n");
        CHECK(!p.ok, "self module-var cycle: must error");
        CHECK(has_error_with(&p.diag, "module var cycle"),
              "self module-var cycle: diag");
        done(&p);
    }
    {
        P p; run(&p,
            "var<private> a: i32 = 0;\n"
            "var<private> b: i32 = 1;\n");
        CHECK(p.ok, "acyclic module-var inits: ok");
        done(&p);
    }

    /* ── §5 Cross-kind value cycles ──────────────────────────── */
    {
        /* const A = B; override B: i32 = A; */
        P p; run(&p,
            "const A = B;\n"
            "override B: i32 = A;\n");
        CHECK(!p.ok, "const <-> override: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "const <-> override: cross-kind diag");
        done(&p);
    }
    {
        /* var<private> a = B; const B = a; */
        P p; run(&p,
            "var<private> a: i32 = B;\n"
            "const B: i32 = a;\n");
        CHECK(!p.ok, "module-var <-> const: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "module-var <-> const: cross-kind diag");
        done(&p);
    }
    {
        /* Three-way: const -> override -> var -> const */
        P p; run(&p,
            "const A: i32 = B;\n"
            "override B: i32 = i32(c);\n"
            "var<private> c: i32 = A;\n");
        CHECK(!p.ok, "const -> override -> var: must error");
        CHECK(has_error_with(&p.diag, "cross-kind value cycle"),
              "three-way mixed cycle: cross-kind diag");
        done(&p);
    }
    {
        /* Single-kind cycle still uses the per-kind message and does
         * NOT fire the cross-kind diag. */
        P p; run(&p,
            "const A: i32 = B + 1;\n"
            "const B: i32 = A * 2;\n");
        CHECK(!p.ok, "const-only cycle: must error");
        CHECK(has_error_with(&p.diag, "const cycle"),
              "single-kind cycle keeps per-kind label");
        CHECK(!has_error_with(&p.diag, "cross-kind value cycle"),
              "single-kind cycle does not fire cross-kind");
        done(&p);
    }

    /* ── Sanity: typical real shader passes ───────────────────── */
    {
        P p; run(&p,
            "const WG: u32 = 256u;\n"
            "fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {\n"
            "  return gid.x + gid.y * nwg.x * WG;\n"
            "}\n"
            "@compute @workgroup_size(256, 1, 1)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n"
            "        @builtin(num_workgroups) nwg: vec3<u32>) {\n"
            "  let tid = flat_id(gid, nwg);\n"
            "  _ = tid;\n"
            "}\n");
        CHECK(p.ok, "real-shader-shape: ok");
        done(&p);
    }

    /* ── Attribute conformance (§12) ──────────────────────────── */
