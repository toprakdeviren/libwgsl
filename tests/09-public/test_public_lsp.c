/**
 * Phase 9 Iter B — semantic tokens overlay + hover + go-to-def.
 *
 * Covers:
 *   - wgsl_semantic_tokens upgrades raw IDENT tokens via resolver
 *     bindings: predeclared types → TYPE, builtin fns → BUILTIN_FUNC,
 *     user fn name → FUNCTION_NAME, `let` decl name → LET, etc.
 *   - wgsl_hover_at returns the resolved symbol's type formatted via
 *     wgsl_type_format (e.g. "i32", "vec3<f32>")
 *   - wgsl_definition_at returns the declaration name span; queries
 *     against predeclared identifiers return absent (they have no
 *     in-source decl)
 *   - hover/goto-def in whitespace gaps return absent
 */
#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

/* Find the first byte offset of `needle` in `src`.  -1 if absent. */
static int find_offset(const char *src, const char *needle) {
    const char *p = strstr(src, needle);
    return p ? (int)(p - src) : -1;
}

int main(void) {
    /* ── Semantic tokens overlay ──────────────────────────────── */
    {
        const char *src =
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n";
        WGSLLexToken *toks = NULL;
        int n = 0;
        int rc = wgsl_semantic_tokens(src, &toks, &n);
        CHECK(rc == 1, "sem_tokens: rc = 1");
        CHECK(n > 0, "sem_tokens: n > 0");

        int saw_fn_name = 0, saw_param = 0, saw_type = 0, saw_a_ident = 0;
        for (int i = 0; i < n; i++) {
            const WGSLLexToken *t = &toks[i];
            uint32_t off = t->offset;
            uint32_t len = t->length;
            if (t->kind == WGSL_LEX_FUNCTION_NAME &&
                len == 3 && memcmp(src + off, "add", 3) == 0)
            {
                saw_fn_name = 1;
            }
            if (t->kind == WGSL_LEX_PARAMETER &&
                len == 1 && src[off] == 'a')
            {
                saw_param = 1;
            }
            if (t->kind == WGSL_LEX_TYPE &&
                len == 3 && memcmp(src + off, "i32", 3) == 0)
            {
                saw_type = 1;
            }
            /* The use of `a` inside `a + b` is also an ident. */
            if (t->kind == WGSL_LEX_PARAMETER && len == 1 && src[off] == 'a' &&
                off > 30)
            {
                saw_a_ident = 1;
            }
        }
        CHECK(saw_fn_name, "sem_tokens: `add` upgraded to FUNCTION_NAME");
        CHECK(saw_param,   "sem_tokens: `a` (param decl) upgraded to PARAMETER");
        CHECK(saw_type,    "sem_tokens: `i32` upgraded to TYPE");
        CHECK(saw_a_ident, "sem_tokens: `a` use also tagged PARAMETER");
        wgsl_lex_free(toks);
    }

    /* Builtin function name. */
    {
        const char *src =
            "fn f() {\n"
            "  let x = sin(1.0);\n"
            "}\n";
        WGSLLexToken *toks = NULL;
        int n = 0;
        wgsl_semantic_tokens(src, &toks, &n);
        int saw_builtin = 0;
        for (int i = 0; i < n; i++) {
            if (toks[i].kind == WGSL_LEX_BUILTIN_FUNC &&
                toks[i].length == 3 &&
                memcmp(src + toks[i].offset, "sin", 3) == 0)
            {
                saw_builtin = 1; break;
            }
        }
        CHECK(saw_builtin, "sem_tokens: `sin` → BUILTIN_FUNC");
        wgsl_lex_free(toks);
    }

    /* Struct decl + use. */
    {
        const char *src =
            "struct Vertex { pos: vec3<f32>, uv: vec2<f32> }\n"
            "fn f(v: Vertex) -> vec3<f32> { return v.pos; }\n";
        WGSLLexToken *toks = NULL;
        int n = 0;
        wgsl_semantic_tokens(src, &toks, &n);
        int saw_struct_decl = 0, saw_struct_use = 0, saw_field = 0;
        for (int i = 0; i < n; i++) {
            const WGSLLexToken *t = &toks[i];
            if (t->kind == WGSL_LEX_STRUCT_NAME &&
                t->length == 6 && memcmp(src + t->offset, "Vertex", 6) == 0)
            {
                if (saw_struct_decl) saw_struct_use = 1;
                else                 saw_struct_decl = 1;
            }
            if (t->kind == WGSL_LEX_FIELD &&
                t->length == 3 && memcmp(src + t->offset, "pos", 3) == 0)
            {
                saw_field = 1;
            }
        }
        CHECK(saw_struct_decl, "sem_tokens: Vertex decl → STRUCT_NAME");
        CHECK(saw_struct_use,  "sem_tokens: Vertex use → STRUCT_NAME");
        CHECK(saw_field,       "sem_tokens: `pos` member → FIELD");
        wgsl_lex_free(toks);
    }

    /* ── Hover ────────────────────────────────────────────────── */
    {
        const char *src = "fn f() { let x: i32 = 5; let y = x + 1; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "hover-src: ok");

        /* Hover on `x` at the first definition site. */
        int off_def = find_offset(src, "x: i32");
        WGSLHover h_def = wgsl_hover_at(r, (uint32_t)off_def);
        CHECK(h_def.present, "hover at def-x: present");
        CHECK(strcmp(h_def.type_repr, "i32") == 0, "hover at def-x: i32");

        /* Hover on `x` at its use site. */
        int off_use = find_offset(src, "x + 1");
        WGSLHover h_use = wgsl_hover_at(r, (uint32_t)off_use);
        CHECK(h_use.present, "hover at use-x: present");
        CHECK(strcmp(h_use.type_repr, "i32") == 0, "hover at use-x: i32");
        wgsl_free(r);
    }
    {
        const char *src = "fn f() { let v = vec3<f32>(0.0, 1.0, 2.0); _ = v; }\n";
        WGSLResult *r = wgsl_check(src);
        int off = find_offset(src, "v = vec3");
        WGSLHover h = wgsl_hover_at(r, (uint32_t)off);
        CHECK(h.present, "hover at v: present");
        CHECK(strcmp(h.type_repr, "vec3<f32>") == 0,
              "hover at v: vec3<f32>");
        wgsl_free(r);
    }
    /* Hover on whitespace → absent. */
    {
        WGSLResult *r = wgsl_check("fn f() {  }\n");
        WGSLHover h = wgsl_hover_at(r, 8);   /* inside the body whitespace */
        CHECK(!h.present, "hover whitespace: absent");
        wgsl_free(r);
    }

    /* ── Go-to-definition ─────────────────────────────────────── */
    {
        const char *src =
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n"
            "fn caller() { _ = add(1, 2); }\n";
        WGSLResult *r = wgsl_check(src);

        /* `add` use site → definition span on the user fn. */
        int use_off = find_offset(src, "add(1, 2)");
        WGSLDefinition d = wgsl_definition_at(r, (uint32_t)use_off);
        CHECK(d.present, "goto-def: add use → present");
        if (d.present) {
            int decl_off = find_offset(src, "add");   /* first occurrence */
            CHECK((int)d.offset == decl_off, "goto-def: points at decl name");
            CHECK(d.length == 3, "goto-def: length = 3");
            CHECK(memcmp(src + d.offset, "add", 3) == 0,
                  "goto-def: bytes match");
        }
        wgsl_free(r);
    }
    /* Goto-def on a predeclared identifier → absent (no in-source decl). */
    {
        const char *src = "fn f() { let v: i32 = 1; }\n";
        WGSLResult *r = wgsl_check(src);
        int off = find_offset(src, "i32 = 1");
        WGSLDefinition d = wgsl_definition_at(r, (uint32_t)off);
        CHECK(!d.present, "goto-def on predeclared: absent");
        wgsl_free(r);
    }
    /* Goto-def on whitespace → absent. */
    {
        WGSLResult *r = wgsl_check("fn f() {}\n");
        WGSLDefinition d = wgsl_definition_at(r, 4);
        CHECK(!d.present, "goto-def whitespace: absent");
        wgsl_free(r);
    }

    fprintf(stderr, "%s  test_public_lsp  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
