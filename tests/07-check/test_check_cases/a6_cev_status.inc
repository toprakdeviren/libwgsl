    /* ── §A6 — typed const-eval status (not message-text control flow) ── */

    {
        /* A: noise swallowed — runtime param expr is not const; no fake diag. */
        P p; run(&p,
            "fn f(x: i32) {\n"
            "  let y = x + 1;\n"
            "  var z = x * 2;\n"
            "}\n");
        CHECK(p.ok, "A6-A: runtime param init typechecks");
        CHECK(!has_error_with(&p.diag, "not a constant"),
              "A6-A: no 'not a constant' leak");
        CHECK(!has_error_with(&p.diag, "not a const-expression"),
              "A6-A: no const-expression leak");
        done(&p);
    }
    {
        /* B: real value error surfaces — module const div0. */
        P p; run(&p, "const bad = 1 / 0;\n");
        CHECK(!p.ok, "A6-B: const 1/0 must fail");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "A6-B: div0 diagnostic kept");
        done(&p);
    }
    {
        /* B′: real value error on let with const-expr init also surfaces. */
        P p; run(&p, "fn f() { let y = 1 / 0; }\n");
        CHECK(!p.ok, "A6-B: let 1/0 must fail");
        CHECK(has_error_with(&p.diag, "division by zero"),
              "A6-B: let div0 diagnostic kept");
        done(&p);
    }
    {
        /* C: rewording invariant — classification is typed status, not
         * message substring.  Evaluate a runtime ident and a value error
         * through wgsl_consteval_expr; assert last_status enum values.
         * Changing the human-readable string at cev_error_kind sites must
         * not flip NOT_CONST ↔ VALUE_ERROR. */
        P p;
        memset(&p, 0, sizeof p);
        wgsl_arena_init(&p.arena);
        wgsl_diag_init(&p.diag);
        static const char *src =
            "fn f(x: i32) { let a = x; let b = 1 / 0; }\n";
        wgsl_source_init(&p.src, src, strlen(src));
        CHECK(wgsl_tokenize(&p.src, &p.arena, &p.diag, &p.lex), "A6-C lex");
        CHECK(wgsl_parse(&p.lex, &p.src, &p.arena, &p.diag, &p.ast), "A6-C parse");
        wgsl_types_init(&p.types, &p.arena);
        CHECK(wgsl_resolve(&p.ast, &p.src, &p.arena, &p.diag, &p.types, &p.res),
              "A6-C resolve");
        wgsl_consteval_init(&p.cev, &p.arena, &p.diag, &p.types, &p.src, &p.res);

        /* Find the two let inits: `x` and `1 / 0`. */
        const WGSLNode *fn = p.ast.root && p.ast.root->child_count
            ? p.ast.root->children[0] : NULL;
        const WGSLNode *body = fn && fn->child_count
            ? fn->children[fn->child_count - 1] : NULL;
        CHECK(body && body->child_count >= 2, "A6-C body shape");
        if (body && body->child_count >= 2) {
            const WGSLNode *let_a = body->children[0];
            const WGSLNode *let_b = body->children[1];
            int ht_a = let_a ? wgsl_letlike_has_type(let_a) : 0;
            int ht_b = let_b ? wgsl_letlike_has_type(let_b) : 0;
            WGSLNode *init_a = let_a ? let_a->children[ht_a ? 1 : 0] : NULL;
            WGSLNode *init_b = let_b ? let_b->children[ht_b ? 1 : 0] : NULL;

            WGSLValue v = {0};
            size_t n0 = p.diag.count;
            int ok_a = wgsl_consteval_expr(&p.cev, init_a, &v);
            CHECK(!ok_a, "A6-C: param ref fails const-eval");
            CHECK(p.cev.store.last_status == WGSL_CEV_NOT_CONST,
                  "A6-C: runtime → WGSL_CEV_NOT_CONST (status, not text)");
            CHECK(wgsl_cev_status_is_noise(p.cev.store.last_status),
                  "A6-C: NOT_CONST is noise");
            /* Roll back noise diags like soft-eval does. */
            p.diag.count = n0;
            p.diag.error_count = 0;
            p.cev.store.had_error = 0;

            memset(&v, 0, sizeof v);
            int ok_b = wgsl_consteval_expr(&p.cev, init_b, &v);
            CHECK(!ok_b, "A6-C: 1/0 fails const-eval");
            CHECK(p.cev.store.last_status == WGSL_CEV_VALUE_ERROR,
                  "A6-C: div0 → WGSL_CEV_VALUE_ERROR (status, not text)");
            CHECK(!wgsl_cev_status_is_noise(p.cev.store.last_status),
                  "A6-C: VALUE_ERROR is not noise");
        }
        wgsl_consteval_destroy(&p.cev);
        wgsl_resolver_destroy(&p.res);
        wgsl_types_destroy(&p.types);
        wgsl_diag_destroy(&p.diag);
        wgsl_source_destroy(&p.src);
        wgsl_arena_destroy(&p.arena);
    }
