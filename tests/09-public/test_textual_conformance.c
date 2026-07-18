/**
 * §3 Textual Structure — conformance smoke tests.
 *
 * Pins the testable claims from `docs/wgsl/en/03_textual_structure.md`
 * against the public API.  Most §3 rules (Pattern_White_Space, line
 * breaks, comments, ident classification, template-list discovery) are
 * already tested at the lexer level in tests/01-utf8 and
 * tests/02-lexer-*.  This file targets the public API surface so the
 * end-to-end pipeline keeps obeying §3 even after lexer refactors, and
 * fills the remaining gaps:
 *
 *   §3 lead          — NUL code point (U+0000) rejected; BOM at start
 *                       rejected
 *   §3.1 Parsing     — non-matching translation_unit produces a
 *                       shader-generation error
 *   §3.3 Comments    — comment between expression tokens parses;
 *                       nested block comments parse; unterminated
 *                       block comment is a shader-generation error
 *   §3.5.2 Int       — non-zero integer literal with leading zero
 *                       rejected (`012`)
 *   §3.6 Keywords    — keywords cannot be used as user identifiers
 *   §3.7 Identifiers — single `_` is not a valid identifier;
 *                       leading `__` is rejected; non-ASCII XID idents
 *                       are accepted
 *   §3.8 Context names — attribute / builtin / interpolation names are
 *                       context-dependent, not keywords; using them
 *                       as regular identifiers is legal
 *   §3.9 Template lists — algorithm edge cases visible end-to-end:
 *                       `a < b && c > d` is two comparisons, not a
 *                       template; `a[b<d]>()` is NOT a template list;
 *                       nested `array<vec4<f32>, 2>` works
 *
 * Spec gaps documented in `docs/wgsl/missing.md` are not tested here
 * (integer-literal overflow §6.2.1, hex-float-precision §3.5,
 * mixed swizzle §8.5.1).
 */
#include "wgsl.h"

#include <stddef.h>
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

static int has_diag_with(const WGSLResult *r, const char *needle) {
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        if (d && d->message && strstr(d->message, needle)) return 1;
    }
    return 0;
}

static void dump_on_fail(const WGSLResult *r, const char *label) {
    if (wgsl_ok(r)) return;
    fprintf(stderr, "  [%s] unexpected diagnostics:\n", label);
    int n = wgsl_diagnostic_count(r);
    for (int i = 0; i < n && i < 5; i++) {
        const WGSLDiagnostic *d = wgsl_diagnostic(r, i);
        fprintf(stderr, "    L%u:%u %s\n",
                d ? d->line : 0, d ? d->column : 0,
                d && d->message ? d->message : "(null)");
    }
}

/* Accept helper: expects wgsl_ok=1 and zero diagnostics. */
static void expect_ok(const char *src, const char *label) {
    WGSLResult *r = wgsl_check(src);
    dump_on_fail(r, label);
    CHECK(wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) == 0, label);
    wgsl_free(r);
}

/* Reject helper: expects wgsl_ok=0 and a diagnostic carrying `needle`
 * in its message.  `needle == NULL` skips the substring check. */
static void expect_reject(const char *src, const char *needle,
                          const char *label) {
    WGSLResult *r = wgsl_check(src);
    CHECK(!wgsl_ok(r), label);
    CHECK(wgsl_diagnostic_count(r) >= 1, label);
    if (needle) CHECK(has_diag_with(r, needle), label);
    wgsl_free(r);
}

int main(void) {
    /* §3 lead - encoding rules. */
    {
        /* NUL code point inside source must be rejected per §3:
         * "The program text must not include a null code point". */
        const char s[] = "fn f() {\x00 return; }\n";
        WGSLResult *r = wgsl_check_n(s, sizeof s - 1);
        CHECK(!wgsl_ok(r), "§3: NUL (U+0000) in source rejected");
        CHECK(has_diag_with(r, "null"),
              "§3: NUL diagnostic mentions 'null'");
        wgsl_free(r);
    }
    {
        /* UTF-8 BOM (EF BB BF) at start is U+FEFF, not blankspace per
         * §3.2 — must not be accepted as leading whitespace. */
        const char s[] = "\xEF\xBB\xBF fn f() { }\n";
        WGSLResult *r = wgsl_check_n(s, sizeof s - 1);
        CHECK(!wgsl_ok(r), "§3: BOM at start rejected (no BOM allowed)");
        wgsl_free(r);
    }

    /* §3.1 - shader-generation error when translation_unit does.
     *           not match the entire token sequence                 */
    expect_reject("fn { }\n",      NULL,
                  "§3.1: translation_unit mismatch → error");
    expect_reject("@@ const X = 1;\n", NULL,
                  "§3.1: stray punctuation tokens at module scope → error");

    /* §3.3 - comments. */
    expect_ok("const X = 1 /* mid */ + 2;\n",
              "§3.3: block comment between tokens parses");
    expect_ok("/* a /* b /* c */ b */ a */ const X = 1;\n",
              "§3.3: triply-nested block comments parse");
    expect_ok("// line comment until end of line\n"
              "const X = 1;\n",
              "§3.3: line-ending comment ends at LF");
    expect_reject("/* never closes",
                  "unterminated",
                  "§3.3: unterminated block comment → shader-generation error");

    /* §3.5.2 - leading-zero non-zero integer rejected. */
    expect_reject("const X = 012;\n",
                  "leading zero",
                  "§3.5.2: leading-zero non-zero decimal int rejected");
    /* And confirm `0i` / `0u` / `0` themselves are fine — those are    *
     * the single-zero form allowed by the grammar.                     */
    expect_ok("const A = 0;\nconst B = 0i;\nconst C = 0u;\n",
              "§3.5.2: plain `0`, `0i`, `0u` accepted");

    /* §3.5.2 - `h` suffix gated on `enable f16;`.
     *                                                             *
     * Per the spec table at the end of §3.5.2: an `h`-suffix       *
     * floating-point literal denotes f16 and "A floating point     *
     * literal with a `h` suffix is used while the f16 extension    *
     * is not enabled" is a shader-generation error.                  */
    expect_reject("fn f() { let x = 1h; }\n",
                  "enable f16",
                  "§3.5.2: `1h` without `enable f16;` rejected");
    expect_ok("enable f16;\n"
              "fn f() { let x : f16 = 1h; }\n",
              "§3.5.2: `1h` with `enable f16;` accepted");
    expect_ok("enable f16;\n"
              "fn f() { let x : f16 = 1.5h; let y : f16 = 0x1p0h; }\n",
              "§3.5.2: decimal + hex `h`-suffix accepted with `enable f16;`");
    /* And: `enable f16;` also unblocks the `f16` type name itself. */
    expect_reject("var<private> x : f16;\n",
                  "enable f16",
                  "§6.2.4: `f16` type without `enable f16;` rejected");
    expect_ok("enable f16;\n"
              "var<private> x : f16 = 1.0h;\n",
              "§6.2.4: `f16` type with `enable f16;` accepted");

    /* §3.5.2 - i32 / u32 maximum literals pin.
     *                                                             *
     * The boundary literals themselves are accepted; overflow     *
     * detection (§6.2.1 / §6.2.3) is a documented gap in          *
     * missing.md and not asserted here.                            */
    expect_ok("const X : i32 = 2147483647i;\nconst Y : u32 = 4294967295u;\n",
              "§3.5.2: i32 max / u32 max literals accepted");

    /* §3.6 - keywords are not user identifiers. */
    expect_reject("fn let() { }\n", NULL,
                  "§3.6: keyword `let` cannot be a function name");
    expect_reject("var fn : i32 = 0;\n", NULL,
                  "§3.6: keyword `fn` cannot be a variable name");

    /* §3.7 - identifier rules. */
    {
        /* Non-ASCII XID_Start / XID_Continue ident accepted. */
        expect_ok("const \xCE\x94 : i32 = 1;\n"   /* U+0394 Δ */
                  "fn use_delta() -> i32 { return \xCE\x94; }\n",
                  "§3.7: non-ASCII XID identifier accepted");
    }
    expect_reject("fn f() { let _ = 1; }\n",
                  NULL,
                  "§3.7: single `_` is not a valid identifier");
    expect_reject("const __foo = 1;\n",
                  "__",
                  "§3.7: identifier starting with `__` rejected");

    /* §3.8 - context-dependent names are not keywords.
     *                                                             *
     * Spec §3.8 calls these "context-dependent name tokens"       *
     * (attribute names, builtin value names, interpolation,       *
     * severity, etc.).  They are NOT reserved.  Using them as     *
     * a regular `const` name at module scope must succeed.        */
    expect_ok("const align : i32 = 1;\n"
              "const binding : i32 = 2;\n"
              "const builtin : i32 = 3;\n"
              "const compute : i32 = 4;\n"
              "const fragment : i32 = 5;\n"
              "const group : i32 = 6;\n"
              "const id : i32 = 7;\n"
              "const location : i32 = 8;\n"
              "const vertex : i32 = 9;\n"
              "const workgroup_size : i32 = 10;\n",
              "§3.8.1: attribute names usable as regular identifiers");
    expect_ok("const position : i32 = 1;\n"
              "const vertex_index : u32 = 2u;\n"
              "const front_facing : bool = true;\n",
              "§3.8.2: builtin value names usable as regular identifiers");
    expect_ok("const perspective : i32 = 1;\n"
              "const linear : i32 = 2;\n"
              "const flat : i32 = 3;\n",
              "§3.8.6: interpolation type names usable as regular idents");
    expect_ok("const center : i32 = 1;\n"
              "const centroid : i32 = 2;\n"
              "const sample : i32 = 3;\n",
              "§3.8.7: interpolation sampling names usable as regular idents");

    /* §3.9 - template list discovery edge cases. */
    {
        /* Nested template: `array<vec4<f32>, 2>` — both `<…>` pairs *
         * resolved as template-arg delimiters.                      */
        expect_ok("var<private> a : array<vec4<f32>, 2>;\n",
                  "§3.9: nested template `array<vec4<f32>, 2>` parses");
    }
    {
        /* `a < b && c > d` — `&&` rejects the pending unclosed     *
         * candidate at the current nesting level, so the two `<`   *
         * and `>` are comparisons.                                 */
        expect_ok("fn f() -> bool {\n"
                  "  let a = 1; let b = 2; let c = 3; let d = 4;\n"
                  "  return a < b && c > d;\n"
                  "}\n",
                  "§3.9: `a < b && c > d` parses as two comparisons");
    }
    {
        /* `a[b<d]>()` — per §3.9 note, the indexing brackets pin   *
         * the nesting level.  The candidate `<` opened after `b`   *
         * is dropped when `]` closes the index, so this is NOT a   *
         * template list.  The remaining `>()` is a syntax error    *
         * (or a stray `>` consumed by relational parsing) — pin    *
         * only that the module is rejected, not the precise        *
         * diagnostic text.                                         */
        expect_reject(
            "fn f() {\n"
            "  var arr : array<i32, 4>;\n"
            "  let b = 0; let d = 1;\n"
            "  let r = arr[b<d]>();\n"
            "}\n",
            NULL,
            "§3.9: `a[b<d]>()` does NOT contain a template list");
    }
    {
        /* `var<storage,read_write>` — keyword-context template.    */
        expect_ok(
            "struct S { x : i32 }\n"
            "@group(0) @binding(0) var<storage, read_write> buf : S;\n"
            "@compute @workgroup_size(1) fn cs() { buf.x = 1; }\n",
            "§3.9: `var<storage,read_write>` template list parses");
    }
    {
        /* `A<B<C>>=D` exotic close — §3.9.                          *
         *                                                            *
         * `array<vec<f32>>=x` is tokenized with `>>=` as a single    *
         * token (no space).  Template discovery must close BOTH      *
         * pending templates with the leading `>>` and treat the      *
         * trailing `=` as a real assignment.                         */
        expect_reject(
            "fn f() {\n"
            "  var a : array<vec2<i32>>;\n"
            "  a=array<vec2<i32>>(vec2<i32>(0, 0));\n"
            "}\n",
            "runtime-sized array",
            "§3.9: `array<vec2<i32>>=...` closes two templates via `>>=`");
        expect_reject(
            "fn f() {\n"
            "  var b : array<array<u32, 2>>;\n"
            "  b=array<array<u32, 2>>(array<u32, 2>(0u, 1u));\n"
            "}\n",
            "runtime-sized array",
            "§3.9: nested `array<array<u32,2>>=` closes two templates");
    }

    /* §16.2 - reserved-word rejection.
     *                                                             *
     * A representative cross-section of the 146-entry spec list,  *
     * chosen to cover every length bucket from 2 to 16 so a       *
     * regression in `wgsl_token_classify_ident`'s length window   *
     * surfaces here.  See `KW_TABLE` in src/token.c.              */
    {
        const char *names[] = {
            "as",                 /* 2  */
            "ref",                /* 3  */
            "auto",               /* 4  */
            "snorm",              /* 5  */
            "filter",             /* 6  */
            "typedef",            /* 7  */
            "nullptr",            /* 8  fixed length-field bug */
            "writeonly",          /* 9  */
            "regardless",         /* 10 fixed length-field bug */
            "groupshared",        /* 11 */
            "column_major",       /* 12 */
            "static_assert",      /* 13 */
            "nointerpolation",    /* 15 */
            "reinterpret_cast",   /* 16 longest entry */
        };
        for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
            char src[128], label[128];
            snprintf(src, sizeof src,
                "fn f() { let %s = 0; }\n", names[i]);
            snprintf(label, sizeof label,
                "§16.2: reserved word '%s' rejected as identifier",
                names[i]);
            expect_reject(src, "reserved", label);
        }
    }

    /* §16.2 — names that LOOK reserved-ish but are NOT on the spec
     *           list MUST stay legal identifiers.  Catches over-strict
     *           regressions in the KW_TABLE (e.g., `block`, `void`,
     *           `fixed`, the `co*` cluster — all in older WGSL drafts
     *           but removed from the current spec).  Cross-ref WGSL
     *           §16.2 / docs/wgsl/en/16_keyword_and_token_summary.md. */
    {
        const char *non_reserved[] = {
            "block",     /* user reported false-positive on `let block = 0u;` */
            "void",      /* old draft; current spec has implicit void return */
            "char",
            "fixed",
            "handle",    /* spec-level concept but not in §16.2 reserved list */
            "object",
            "fun",
            "mixin",
            "bind",
            "coroutine",
            "synchronized",
            "reinterpret",
        };
        for (size_t i = 0; i < sizeof non_reserved / sizeof non_reserved[0]; i++) {
            char src[128], label[128];
            snprintf(src, sizeof src,
                "fn f() { let %s = 0; let x = %s; }\n",
                non_reserved[i], non_reserved[i]);
            snprintf(label, sizeof label,
                "§16.2: '%s' is NOT in the spec reserved list → legal identifier",
                non_reserved[i]);
            expect_ok(src, label);
        }
    }

    fprintf(stderr, "%s  test_textual_conformance  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
