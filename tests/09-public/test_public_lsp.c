/**
 * Public LSP API tests for semantic tokens, hover, and go-to-def.
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
    /* Semantic tokens overlay. */
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

    /* Hover. */
    {
        const char *src = "fn f() { let x: i32 = 5; let y = x + 1; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "hover-src: ok");

        /* Hover on `x` at the first definition site. */
        int off_def = find_offset(src, "x: i32");
        WGSLHover h_def = wgsl_hover_at(r, (uint32_t)off_def);
        CHECK(h_def.present, "hover at def-x: present");
        CHECK(strcmp(h_def.type_repr, "i32") == 0, "hover at def-x: i32");
        CHECK(h_def.signature && h_def.signature[0] == '\0',
              "hover signature: reserved empty string");
        CHECK(h_def.doc && h_def.doc[0] == '\0',
              "hover doc: reserved empty string");

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
    /* Declaration-site lookup uses the decl AST key, so duplicate local
     * spellings in different functions do not collide. */
    {
        const char *src =
            "fn a(x: i32) -> i32 { return x; }\n"
            "fn b(x: f32) -> f32 { return x; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "hover dup decl src: ok");
        int off = find_offset(src, "x: f32");
        WGSLHover h = wgsl_hover_at(r, (uint32_t)off);
        CHECK(h.present, "hover dup decl: present");
        CHECK(strcmp(h.type_repr, "f32") == 0, "hover dup decl: f32");
        WGSLDefinition d = wgsl_definition_at(r, (uint32_t)off);
        CHECK(d.present && (int)d.offset == off && d.length == 1,
              "goto dup decl: same declaration");
        wgsl_free(r);
    }
    /* Hover on whitespace → absent. */
    {
        WGSLResult *r = wgsl_check("fn f() {  }\n");
        WGSLHover h = wgsl_hover_at(r, 8);   /* inside the body whitespace */
        CHECK(!h.present, "hover whitespace: absent");
        wgsl_free(r);
    }

    /* Go-to-definition. */
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

    /* §3.1 Document symbols (outline). */
    {
        const char *src =
            "struct S { a: i32, b: f32 }\n"
            "const C: i32 = 1;\n"
            "fn add(x: i32, y: i32) -> i32 { return x + y; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "outline src: ok");
        WGSLOutlineSymbol *syms = NULL;
        int n = 0;
        CHECK(wgsl_document_symbols(r, &syms, &n) == 1, "outline: rc");
        int saw_s = 0, saw_a = 0, saw_c = 0, saw_add = 0, saw_x = 0;
        int s_idx = -1, add_idx = -1;
        for (int i = 0; i < n; i++) {
            const WGSLOutlineSymbol *o = &syms[i];
            if (o->name_length == 1 && src[o->name_offset] == 'S' &&
                o->kind == WGSL_OUTLINE_STRUCT) { saw_s = 1; s_idx = i; }
            if (o->name_length == 1 && src[o->name_offset] == 'a' &&
                o->kind == WGSL_OUTLINE_FIELD && o->parent == s_idx) saw_a = 1;
            if (o->name_length == 1 && src[o->name_offset] == 'C' &&
                o->kind == WGSL_OUTLINE_CONSTANT) saw_c = 1;
            if (o->name_length == 3 && memcmp(src + o->name_offset, "add", 3) == 0 &&
                o->kind == WGSL_OUTLINE_FUNCTION) { saw_add = 1; add_idx = i; }
            if (o->name_length == 1 && src[o->name_offset] == 'x' &&
                o->kind == WGSL_OUTLINE_PARAMETER && o->parent == add_idx) saw_x = 1;
        }
        CHECK(saw_s && saw_a, "outline: struct S + field a");
        CHECK(saw_c, "outline: const C");
        CHECK(saw_add && saw_x, "outline: fn add + param x");
        wgsl_outline_free(syms);
        wgsl_free(r);
    }

    /* §3.1 References + prepare_rename. */
    {
        const char *src =
            "fn add(a: i32) -> i32 { return a; }\n"
            "fn caller() { _ = add(1); _ = add(2); }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "refs src: ok");
        int use = find_offset(src, "add(1)");
        WGSLReference *refs = NULL;
        int n = 0;
        CHECK(wgsl_references_at(r, (uint32_t)use, &refs, &n) == 1, "refs: rc");
        /* decl + 2 call sites */
        CHECK(n == 3, "refs: 3 sites for add");
        int defs = 0, uses = 0;
        for (int i = 0; i < n; i++) {
            if (refs[i].is_definition) defs++;
            else uses++;
            CHECK(refs[i].length == 3, "refs: name length 3");
        }
        CHECK(defs == 1 && uses == 2, "refs: 1 def + 2 uses");
        uint32_t no = 0, nl = 0;
        CHECK(wgsl_prepare_rename(r, (uint32_t)use, &no, &nl) == 1, "rename ok");
        CHECK(nl == 3 && memcmp(src + no, "add", 3) == 0, "rename span is add");
        /* predeclared cannot rename */
        int i32off = find_offset(src, "i32");
        CHECK(wgsl_prepare_rename(r, (uint32_t)i32off, &no, &nl) == 0,
              "rename predeclared rejected");
        wgsl_references_free(refs);
        wgsl_free(r);
    }

    /* §3.1 Folding ranges. */
    {
        const char *src =
            "fn f() {\n"
            "  if (true) { let x = 1; }\n"
            "  loop { break; }\n"
            "}\n"
            "struct S { a: i32 }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "fold src: ok");
        WGSLFoldingRange *fr = NULL;
        int n = 0;
        CHECK(wgsl_folding_ranges(r, &fr, &n) == 1, "fold: rc");
        CHECK(n >= 3, "fold: at least fn + if + loop (+struct)");
        for (int i = 0; i < n; i++)
            CHECK(fr[i].end_offset > fr[i].start_offset, "fold: non-empty range");
        wgsl_folding_free(fr);
        wgsl_free(r);
    }

    /* §3.1 Completions. */
    {
        const char *src =
            "fn helper() {}\n"
            "fn main() {\n"
            "  let local_v: i32 = 1;\n"
            "  let z = local_v;\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "compl src: ok");
        int off = find_offset(src, "local_v;");
        /* cursor mid-name → filter prefix "local_v" or at start of name */
        WGSLCompletionItem *items = NULL;
        int n = 0;
        CHECK(wgsl_completions_at(r, (uint32_t)off, &items, &n) == 1, "compl: rc");
        CHECK(n > 0, "compl: non-empty");
        int saw_local = 0, saw_helper = 0, saw_fn_kw = 0, saw_i32 = 0;
        for (int i = 0; i < n; i++) {
            if (strcmp(items[i].label, "local_v") == 0) saw_local = 1;
            if (strcmp(items[i].label, "helper") == 0) saw_helper = 1;
            if (strcmp(items[i].label, "fn") == 0) saw_fn_kw = 1;
            if (strcmp(items[i].label, "i32") == 0) saw_i32 = 1;
        }
        CHECK(saw_local, "compl: local_v visible in body");
        CHECK(saw_helper, "compl: module fn helper");
        CHECK(saw_fn_kw, "compl: keyword fn");
        CHECK(saw_i32, "compl: predeclared i32");
        /* Prefix filter: only "loc…" */
        wgsl_completions_free(items, n);
        items = NULL; n = 0;
        CHECK(wgsl_completions_at(r, (uint32_t)off + 3, &items, &n) == 1,
              "compl prefix: rc");  /* offset inside "local_v" after "loc" */
        int only_loc = 1, has_local = 0;
        for (int i = 0; i < n; i++) {
            if (strncmp(items[i].label, "loc", 3) != 0) only_loc = 0;
            if (strcmp(items[i].label, "local_v") == 0) has_local = 1;
        }
        CHECK(has_local && only_loc, "compl: prefix loc filters to local_v…");
        wgsl_completions_free(items, n);
        wgsl_free(r);
    }

    /* §3.1 Signature help. */
    {
        const char *src =
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n"
            "fn caller() { _ = add(1, 2); }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "sig src: ok");
        int off = find_offset(src, "1, 2");
        WGSLSignatureHelp sh = wgsl_signature_help_at(r, (uint32_t)off);
        CHECK(sh.present, "sig: present");
        CHECK(sh.parameter_count == 2, "sig: 2 params");
        CHECK(strstr(sh.label, "add") != NULL, "sig: label names add");
        CHECK(strstr(sh.label, "i32") != NULL, "sig: label has i32");
        CHECK(sh.active_parameter <= 1, "sig: active param in range");
        wgsl_free(r);
    }

    /* §3.1 Code actions (enable quickfix). */
    {
        const char *src = "fn f() { let x: f16 = 1.0h; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(!wgsl_ok(r), "ca src: not ok (needs enable f16)");
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1, "ca: rc");
        CHECK(n >= 1, "ca: at least one action");
        int saw = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strcmp(acts[i].kind, "quickfix.enable.f16") == 0) {
                saw = 1;
                CHECK(acts[i].edit_count == 1, "ca: one edit");
                CHECK(acts[i].edits[0].start_offset == 0, "ca: insert at 0");
                CHECK(strstr(acts[i].edits[0].new_text, "enable f16;") != NULL,
                      "ca: text is enable f16;");
            }
        }
        CHECK(saw, "ca: quickfix.enable.f16 present");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }
    /* Already enabled → no action. */
    {
        const char *src = "enable f16;\nfn f() { let x: f16 = 1.0h; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "ca enabled src: ok");
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1, "ca enabled: rc");
        int saw_enable = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strcmp(acts[i].kind, "quickfix.enable.f16") == 0)
                saw_enable = 1;
        }
        CHECK(!saw_enable, "ca enabled: no enable action");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }
    /* Insert after existing enable preamble. */
    {
        const char *src = "enable subgroups;\nfn f() { let x: f16 = 1.0h; }\n";
        WGSLResult *r = wgsl_check(src);
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1 && n >= 1, "ca after: has action");
        int ok_ins = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strstr(acts[i].kind, "f16")) {
                CHECK(acts[i].edits[0].start_offset > 0,
                      "ca after: insert after preamble");
                ok_ins = 1;
            }
        }
        CHECK(ok_ins, "ca after: f16 action found");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }

    /* §3.1 location auto-fix. */
    {
        const char *src =
            "@fragment\n"
            "fn fs(color: vec4f) -> vec4f { return color; }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(!wgsl_ok(r), "loc: not ok");
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1, "loc: rc");
        int loc_n = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strstr(acts[i].kind, "add_location")) {
                loc_n++;
                CHECK(acts[i].edit_count == 1, "loc: one edit");
                CHECK(acts[i].edits[0].new_text &&
                      strstr(acts[i].edits[0].new_text, "@location(") != NULL,
                      "loc: text is @location");
            }
        }
        CHECK(loc_n >= 2, "loc: param + return fixes");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }

    /* §3.1 align power-of-two auto-fix. */
    {
        const char *src =
            "struct S {\n"
            "  @align(3) x: i32,\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(!wgsl_ok(r), "align: not ok");
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1, "align: rc");
        int saw = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strstr(acts[i].kind, "align_pow2")) {
                saw = 1;
                CHECK(acts[i].edits[0].new_text &&
                      strcmp(acts[i].edits[0].new_text, "4") == 0,
                      "align: rewrites 3 → 4");
            }
        }
        CHECK(saw, "align: pow2 action present");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }

    /* Optimizer quickfix. */
    {
        const char *src =
            "@compute @workgroup_size(1)\n"
            "fn main() {\n"
            "  let x = 1;\n"
            "  let y = 2;\n"
            "  _ = y;\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "opt ca src: ok");
        WGSLCodeAction *acts = NULL;
        int n = 0;
        CHECK(wgsl_code_actions(r, &acts, &n) == 1, "opt ca: rc");
        int saw = 0;
        for (int i = 0; i < n; i++) {
            if (acts[i].kind && strcmp(acts[i].kind, "quickfix.opt.apply") == 0) {
                saw = 1;
                CHECK(acts[i].edit_count == 1, "opt ca: one edit");
                CHECK(acts[i].edits[0].start_offset == 0, "opt ca: full start");
                CHECK(acts[i].edits[0].end_offset == strlen(src),
                      "opt ca: full end");
                CHECK(strstr(acts[i].edits[0].new_text, "let x") == NULL,
                      "opt ca: optimized text removes x");
                WGSLResult *rr = wgsl_check(acts[i].edits[0].new_text);
                CHECK(rr != NULL && wgsl_ok(rr), "opt ca: edit checks");
                if (rr) wgsl_free(rr);
            }
        }
        CHECK(saw, "opt ca: quickfix.opt.apply present");
        wgsl_code_actions_free(acts, n);
        wgsl_free(r);
    }

    fprintf(stderr, "%s  test_public_lsp  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
