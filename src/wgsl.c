/**
 * @file wgsl.c — public C API implementation (Phase 9).
 *
 * Iter A — WGSLResult lifecycle: `wgsl_check` / `wgsl_check_n` /
 *          `wgsl_free`; verdict + diagnostic accessors;
 *          process-init glue; tokenizer-only `wgsl_lex`.
 * Iter B — `wgsl_semantic_tokens` overlays resolver bindings on raw
 *          identifier tokens; `wgsl_hover_at` returns the resolved
 *          type for the symbol under a byte offset;
 *          `wgsl_definition_at` returns the declaration name span.
 *
 * Iter C will produce the module-summary JSON.
 */
#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/check.h"
#include "internal/consteval.h"
#include "internal/diag.h"
#include "internal/layout.h"
#include "internal/lexer.h"
#include "internal/parser.h"
#include "internal/resolver.h"
#include "internal/source.h"
#include "internal/project.h"
#include "internal/token.h"
#include "internal/types.h"
#include "internal/utf8.h"
#include "internal/validate.h"

/* ── Result handle ────────────────────────────────────────────────── */

struct WGSLResult {
    WGSLArena          arena;        /* parallel: AST + arena strings   */
    char              *src_copy;     /* heap-owned source bytes         */
    size_t             src_len;
    WGSLSource         source;
    WGSLDiagBag        diag;
    WGSLLexResult      lex;
    WGSLAst            ast;
    WGSLTypeStore      types;
    WGSLResolver       res;
    WGSLConstEvaluator cev;
    WGSLTypeChecker    tc;
    WGSLValidator      val;

    int                pipeline_ok;   /* lex && parse && diag-error-free */

    /* Lazy cache for `wgsl_module_json`.  Built on first call;
     * reused thereafter.  Empty string on failure.  Owned by the
     * arena, so freed in `wgsl_free` along with everything else. */
    char              *json_cache;
    size_t             json_cache_len;
    int                json_built;
};

static int add_size_checked(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

/* Stand-in result for failures that happen before a normal result exists. */
static WGSLResult *make_error_result(const char *message) {
    WGSLResult *r = (WGSLResult *)calloc(1, sizeof *r);
    if (!r) return NULL;
    wgsl_arena_init(&r->arena);
    wgsl_diag_init (&r->diag);
    if (wgsl_source_init(&r->source, "", 0)) {
        wgsl_diag_emit_at(&r->diag, &r->source, WGSL_DIAG_ERROR, 0, 0, NULL,
                          "%s", message ? message : "WGSL frontend error");
    }
    (void)wgsl_types_init(&r->types, &r->arena);
    return r;
}

/* ── Process lifecycle ────────────────────────────────────────────── */

void wgsl_init(void) {
    wgsl_utf8_init();
}

void wgsl_shutdown(void) {
    /* Unicode tables are file-static and have no cleanup hook in v1.
     * Implemented for symmetry. */
}

const char *wgsl_spec_pin(void) {
    return WGSL_SPEC_PIN;
}

const char *wgsl_unicode_version(void) {
    return "17.0.0";
}

/* ── Check ────────────────────────────────────────────────────────── */

WGSLResult *wgsl_check(const char *src) {
    return wgsl_check_n(src, src ? strlen(src) : 0);
}

WGSLResult *wgsl_check_n(const char *src, size_t src_len) {
    wgsl_utf8_init();

    if (src_len > UINT32_MAX) {
        return make_error_result("source is too large for 32-bit WGSL spans");
    }

    WGSLResult *r = (WGSLResult *)calloc(1, sizeof *r);
    if (!r) return NULL;

    /* Copy the source — diagnostics expose `WGSLDiagnostic.message`
     * as a borrowed pointer into our arena, but the source bytes
     * themselves are needed for hover / goto-def queries that look up
     * identifiers at byte offsets after the call returns. */
    r->src_len = (src && src_len > 0) ? src_len : 0;
    r->src_copy = (char *)malloc(r->src_len + 1);
    if (!r->src_copy) {
        free(r);
        return make_error_result("out of memory while copying WGSL source");
    }
    if (r->src_len > 0) memcpy(r->src_copy, src, r->src_len);
    r->src_copy[r->src_len] = '\0';

    wgsl_arena_init(&r->arena);
    wgsl_diag_init (&r->diag);
    if (!wgsl_source_init(&r->source, r->src_copy, r->src_len)) {
        wgsl_diag_destroy(&r->diag);
        wgsl_arena_destroy(&r->arena);
        free(r->src_copy);
        free(r);
        return make_error_result("out of memory while indexing WGSL source");
    }

    int lex_ok    = wgsl_tokenize(&r->source, &r->arena, &r->diag, &r->lex);
    if (!r->lex.tokens || r->lex.count == 0 ||
        r->lex.tokens[r->lex.count - 1].kind != WGSL_TOK_EOF)
    {
        if (!wgsl_diag_has_error(&r->diag)) {
            wgsl_diag_emit_at(&r->diag, &r->source, WGSL_DIAG_ERROR, 0, 0, NULL,
                              "lexer failed before producing EOF token");
        }
        r->pipeline_ok = 0;
        return r;
    }

    int parse_ok  = wgsl_parse(&r->lex, &r->source, &r->arena, &r->diag, &r->ast);
    if (!r->ast.root) {
        if (!wgsl_diag_has_error(&r->diag)) {
            wgsl_diag_emit_at(&r->diag, &r->source, WGSL_DIAG_ERROR, 0, 0, NULL,
                              "parser failed before producing an AST");
        }
        r->pipeline_ok = 0;
        return r;
    }

    if (!wgsl_types_init(&r->types, &r->arena)) {
        if (!wgsl_diag_has_error(&r->diag)) {
            wgsl_diag_emit_at(&r->diag, &r->source, WGSL_DIAG_ERROR, 0, 0, NULL,
                              "out of memory while initializing WGSL types");
        }
        r->pipeline_ok = 0;
        return r;
    }

    int res_ok = wgsl_resolve(
        &r->ast, &r->source, &r->arena, &r->diag, &r->types, &r->res);
    int ce_ok  = wgsl_consteval(
        &r->ast, &r->source, &r->arena, &r->diag, &r->types, &r->res, &r->cev);
    int tc_ok  = wgsl_typecheck(
        &r->ast, &r->source, &r->arena, &r->diag, &r->types, &r->res, &r->cev, &r->tc);
    int val_ok = wgsl_validate(
        &r->ast, &r->source, &r->arena, &r->diag, &r->types, &r->res, &r->cev, &r->tc, &r->val);

    r->pipeline_ok =
        lex_ok && parse_ok && res_ok && ce_ok && tc_ok && val_ok &&
        !wgsl_diag_has_error(&r->diag);
    return r;
}

WGSLResult *wgsl_check_with_preamble(
    const char *preamble, size_t preamble_len,
    const char *src,      size_t src_len)
{
    /* Concatenate into a single contiguous buffer; then delegate to
     * `wgsl_check_n`.  The synthesised buffer is freed before this
     * function returns — `wgsl_check_n` makes its own owned copy. */
    if (!preamble) preamble_len = 0;
    if (!src)      src_len      = 0;
    /* Ensure a separator newline between preamble and src so the
     * boundary doesn't fuse two tokens into one (e.g. `xyz` at
     * preamble end + `abc` at src start). */
    int need_sep = preamble_len > 0 && preamble[preamble_len - 1] != '\n';
    size_t total = 0;
    if (!add_size_checked(preamble_len, need_sep ? 1u : 0u, &total) ||
        !add_size_checked(total, src_len, &total) ||
        total == SIZE_MAX)
    {
        return make_error_result("source and preamble length overflow");
    }
    if (total > UINT32_MAX) {
        return make_error_result("combined source is too large for 32-bit WGSL spans");
    }

    char *joined = (char *)malloc(total + 1);
    if (!joined) return make_error_result("out of memory while joining WGSL preamble");
    if (preamble_len > 0) memcpy(joined, preamble, preamble_len);
    size_t off = preamble_len;
    if (need_sep) joined[off++] = '\n';
    if (src_len > 0) memcpy(joined + off, src, src_len);
    joined[total] = '\0';

    WGSLResult *r = wgsl_check_n(joined, total);
    free(joined);
    return r;
}

void wgsl_free(WGSLResult *r) {
    if (!r) return;
    wgsl_validator_destroy(&r->val);
    wgsl_typecheck_destroy(&r->tc);
    wgsl_consteval_destroy(&r->cev);
    wgsl_resolver_destroy (&r->res);
    wgsl_types_destroy    (&r->types);
    wgsl_diag_destroy     (&r->diag);
    wgsl_source_destroy   (&r->source);
    wgsl_arena_destroy    (&r->arena);
    free(r->src_copy);
    free(r);
}

/* ── Verdict + module summary ─────────────────────────────────────── */

int wgsl_ok(const WGSLResult *r) {
    return r ? (r->pipeline_ok != 0) : 0;
}

const char *wgsl_error(const WGSLResult *r) {
    if (!r) return "";
    for (size_t i = 0; i < r->diag.count; i++) {
        if (r->diag.items[i].severity == WGSL_DIAG_ERROR) {
            return r->diag.items[i].message ? r->diag.items[i].message : "";
        }
    }
    return "";
}

/* ── Module summary JSON — Iter C ─────────────────────────────────── */


#include "wgsl/module_json.inc"
#include "wgsl/lex_semantic.inc"
#include "wgsl/queries_project.inc"
