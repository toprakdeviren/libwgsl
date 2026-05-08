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

/* Stand-in result to honour the "always non-NULL" contract on OOM. */
static WGSLResult *make_empty_result(void) {
    WGSLResult *r = (WGSLResult *)calloc(1, sizeof *r);
    if (!r) return NULL;
    wgsl_arena_init(&r->arena);
    wgsl_diag_init (&r->diag);
    /* `wgsl_source_init` succeeds on a zero-length input. */
    wgsl_source_init(&r->source, "", 0);
    wgsl_types_init (&r->types, &r->arena);
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
        return make_empty_result();
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
        return make_empty_result();
    }

    int lex_ok    = wgsl_tokenize(&r->source, &r->arena, &r->diag, &r->lex);
    int parse_ok  = wgsl_parse(&r->lex, &r->source, &r->arena, &r->diag, &r->ast);

    if (!wgsl_types_init(&r->types, &r->arena)) {
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
    size_t total = preamble_len + (need_sep ? 1u : 0u) + src_len;

    char *joined = (char *)malloc(total + 1);
    if (!joined) return NULL;
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

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
    int    oom;
} JsonBuf;

static void jb_grow(JsonBuf *j, size_t need) {
    if (j->oom) return;
    if (j->len + need + 1 <= j->cap) return;
    size_t cap = j->cap ? j->cap : 256;
    while (cap < j->len + need + 1) cap *= 2;
    char *g = (char *)realloc(j->buf, cap);
    if (!g) { j->oom = 1; return; }
    j->buf = g;
    j->cap = cap;
}

static void jb_putc(JsonBuf *j, char c) {
    jb_grow(j, 1);
    if (j->oom) return;
    j->buf[j->len++] = c;
}

static void jb_putn(JsonBuf *j, const char *s, size_t n) {
    jb_grow(j, n);
    if (j->oom) return;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
}

static void jb_puts(JsonBuf *j, const char *s) {
    jb_putn(j, s, strlen(s));
}

static void jb_put_int(JsonBuf *j, long long v) {
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%lld", v);
    if (n > 0) jb_putn(j, tmp, (size_t)n);
}

/* JSON-escape a byte range. */
static void jb_put_jstr(JsonBuf *j, const char *s, size_t n) {
    jb_putc(j, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  jb_putn(j, "\\\"", 2); break;
        case '\\': jb_putn(j, "\\\\", 2); break;
        case '\b': jb_putn(j, "\\b", 2);  break;
        case '\f': jb_putn(j, "\\f", 2);  break;
        case '\n': jb_putn(j, "\\n", 2);  break;
        case '\r': jb_putn(j, "\\r", 2);  break;
        case '\t': jb_putn(j, "\\t", 2);  break;
        default:
            if (c < 0x20) {
                char tmp[8];
                int k = snprintf(tmp, sizeof tmp, "\\u%04x", c);
                if (k > 0) jb_putn(j, tmp, (size_t)k);
            } else {
                jb_putc(j, (char)c);
            }
        }
    }
    jb_putc(j, '"');
}

/* Convenience: emit a JSON string from a (offset, length) source span. */
static void jb_put_span(JsonBuf *j, const WGSLSource *src,
                        uint32_t off, uint32_t len)
{
    if (off + len > src->length) { jb_puts(j, "\"\""); return; }
    jb_put_jstr(j, src->bytes + off, len);
}

/* Format a TypeInfo into the JSON buffer as a quoted string. */
static void jb_put_type(JsonBuf *j, WGSLTypeInfo *t) {
    if (!t) { jb_puts(j, "\"\""); return; }
    char tmp[128];
    size_t n = wgsl_type_format(t, tmp, sizeof tmp);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    jb_put_jstr(j, tmp, n);
}

/* Resolve an attribute argument (an EXPR_*) to an integer.  Returns 0
 * if the arg isn't a const-eval'd integer; uses the result's existing
 * `WGSLConstEvaluator` (no second pass). */
static long long wgsl_attr_int_arg(WGSLResult *r, WGSLNode *arg) {
    if (!arg) return 0;
    /* Cast away const so we can reuse `wgsl_consteval_expr`; the
     * call only writes `*out`, not the AST. */
    WGSLValue v = {0};
    if (!wgsl_consteval_expr(&r->cev, arg, &v)) return 0;
    if (v.kind != WGSL_VAL_INT) return 0;
    return (long long)v.u.i;
}

/* Look up an attribute on `[parent->children + first .. + first+count)`
 * by name.  Returns the ATTRIBUTE node or NULL. */
static WGSLNode *json_find_attr(
    WGSLResult *r, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = (uint32_t)(nm->payload[0] & 0xFFFFFFFFu);
        uint32_t len = (uint32_t)(nm->payload[0] >> 32);
        if (len == wl &&
            memcmp(r->source.bytes + off, want, wl) == 0)
        {
            return a;
        }
    }
    return NULL;
}

static const char *json_stage_for(WGSLResult *r, WGSLNode *fn,
                                  uint32_t fn_attr_count)
{
    if (json_find_attr(r, fn, 0, fn_attr_count, "vertex"))   return "vertex";
    if (json_find_attr(r, fn, 0, fn_attr_count, "fragment")) return "fragment";
    if (json_find_attr(r, fn, 0, fn_attr_count, "compute"))  return "compute";
    return NULL;
}

static void emit_entry_points(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, "\"entry_points\":[");
    int first = 1;
    if (!r->ast.root) { jb_puts(j, "]"); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_FUNCTION) continue;

        uint32_t FA = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        const char *stage = json_stage_for(r, n, FA);
        if (!stage) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"stage\":\"");
        jb_puts(j, stage);
        jb_putc(j, '"');

        if (strcmp(stage, "compute") == 0) {
            WGSLNode *wgs = json_find_attr(r, n, 0, FA, "workgroup_size");
            if (wgs) {
                jb_puts(j, ",\"workgroup_size\":[");
                /* wgs->children: [name_ident, arg1, arg2?, arg3?] */
                long long dims[3] = { 1, 1, 1 };
                for (uint32_t k = 1; k < wgs->child_count && k <= 3; k++) {
                    dims[k - 1] = wgsl_attr_int_arg(r, wgs->children[k]);
                }
                jb_put_int(j, dims[0]); jb_putc(j, ',');
                jb_put_int(j, dims[1]); jb_putc(j, ',');
                jb_put_int(j, dims[2]);
                jb_putc(j, ']');
            }
        }
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void emit_resources(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"resources\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_VAR) continue;

        uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        WGSLNode *gp = json_find_attr(r, n, 0, A, "group");
        WGSLNode *bd = json_find_attr(r, n, 0, A, "binding");
        if (!gp || !bd) continue;       /* not a bound resource          */

        long long group   = (gp->child_count >= 2)
            ? wgsl_attr_int_arg(r, gp->children[1]) : 0;
        long long binding = (bd->child_count >= 2)
            ? wgsl_attr_int_arg(r, bd->children[1]) : 0;

        /* Address space + access mode from the template list. */
        uint32_t T = (uint32_t)(n->payload[1] >> 32);
        const char *as_name = "private";
        const char *acc_name = NULL;
        if (T >= 1 && A < n->child_count) {
            WGSLNode *as_node = n->children[A];
            if (as_node && as_node->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t off = (uint32_t)(as_node->payload[0] & 0xFFFFFFFFu);
                uint32_t len = (uint32_t)(as_node->payload[0] >> 32);
                if      (len == 7 && memcmp(r->source.bytes + off, "storage", 7) == 0) as_name = "storage";
                else if (len == 7 && memcmp(r->source.bytes + off, "uniform", 7) == 0) as_name = "uniform";
                else if (len == 9 && memcmp(r->source.bytes + off, "workgroup", 9) == 0) as_name = "workgroup";
                else if (len == 7 && memcmp(r->source.bytes + off, "private", 7) == 0) as_name = "private";
            }
        }
        if (T >= 2 && A + 1 < n->child_count) {
            WGSLNode *ac_node = n->children[A + 1];
            if (ac_node && ac_node->kind == WGSL_NODE_EXPR_IDENT) {
                uint32_t off = (uint32_t)(ac_node->payload[0] & 0xFFFFFFFFu);
                uint32_t len = (uint32_t)(ac_node->payload[0] >> 32);
                if      (len == 4 && memcmp(r->source.bytes + off, "read", 4) == 0) acc_name = "read";
                else if (len == 5 && memcmp(r->source.bytes + off, "write", 5) == 0) acc_name = "write";
                else if (len == 10 && memcmp(r->source.bytes + off, "read_write", 10) == 0) acc_name = "read_write";
            }
        }
        /* Spec defaults: storage → read, uniform → read. */
        if (!acc_name) acc_name = "read";

        WGSLSymbol *sym = NULL;
        for (size_t k = 0; k < r->res.all_decl_count; k++) {
            if (r->res.all_decls[k]->ast == n) {
                sym = r->res.all_decls[k]; break;
            }
        }

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"group\":");   jb_put_int(j, group);
        jb_puts(j, ",\"binding\":"); jb_put_int(j, binding);
        jb_puts(j, ",\"address_space\":\""); jb_puts(j, as_name); jb_putc(j, '"');
        jb_puts(j, ",\"access\":\"");        jb_puts(j, acc_name); jb_putc(j, '"');
        jb_puts(j, ",\"type\":");
        jb_put_type(j, sym ? sym->type : NULL);
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void emit_structs(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"structs\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_STRUCT) continue;

        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"members\":[");
        int mfirst = 1;
        for (uint32_t k = 0; k < n->child_count; k++) {
            WGSLNode *m = n->children[k];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            if (!mfirst) jb_putc(j, ',');
            mfirst = 0;
            jb_puts(j, "{\"name\":");
            uint32_t mno = (uint32_t)(m->payload[0] & 0xFFFFFFFFu);
            uint32_t mnl = (uint32_t)(m->payload[0] >> 32);
            jb_put_span(j, &r->source, mno, mnl);
            jb_puts(j, ",\"type\":");
            uint32_t MA = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            WGSLTypeInfo *mt = NULL;
            if (MA < m->child_count) {
                /* The TC's `resolve_type_spec` records the resolved
                 * type on the side-table for every type-spec node it
                 * visits, so we can pull it back here. */
                mt = wgsl_typecheck_type_of(&r->tc, m->children[MA]);
            }
            jb_put_type(j, mt);
            jb_putc(j, '}');
        }
        jb_puts(j, "]}");
    }
    jb_putc(j, ']');
}

static void emit_overrides(WGSLResult *r, JsonBuf *j) {
    jb_puts(j, ",\"overrides\":[");
    int first = 1;
    if (!r->ast.root) { jb_putc(j, ']'); return; }
    WGSLNode *tu = r->ast.root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n || n->kind != WGSL_NODE_DECL_OVERRIDE) continue;

        uint32_t A = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
        WGSLNode *id = json_find_attr(r, n, 0, A, "id");

        WGSLSymbol *sym = NULL;
        for (size_t k = 0; k < r->res.all_decl_count; k++) {
            if (r->res.all_decls[k]->ast == n) {
                sym = r->res.all_decls[k]; break;
            }
        }
        if (!first) jb_putc(j, ',');
        first = 0;
        jb_puts(j, "{\"name\":");
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        jb_put_span(j, &r->source, no, nl);
        jb_puts(j, ",\"type\":");
        jb_put_type(j, sym ? sym->type : NULL);
        jb_puts(j, ",\"id\":");
        if (id && id->child_count >= 2) {
            jb_put_int(j, wgsl_attr_int_arg(r, id->children[1]));
        } else {
            jb_puts(j, "null");
        }
        jb_putc(j, '}');
    }
    jb_putc(j, ']');
}

static void build_module_json(WGSLResult *r) {
    JsonBuf j = {0};
    jb_putc(&j, '{');
    jb_puts(&j, "\"spec_pin\":\""); jb_puts(&j, WGSL_SPEC_PIN); jb_putc(&j, '"');
    jb_putc(&j, ',');
    emit_entry_points(r, &j);
    emit_resources   (r, &j);
    emit_structs     (r, &j);
    emit_overrides   (r, &j);
    jb_putc(&j, '}');

    if (j.oom || !j.buf) {
        free(j.buf);
        r->json_cache     = "";
        r->json_cache_len = 0;
        r->json_built     = 1;
        return;
    }
    /* Copy into the arena so the buffer survives `wgsl_free` semantics
     * and lives alongside the rest of the result's owned strings. */
    char *out = (char *)wgsl_arena_alloc(&r->arena, j.len + 1);
    if (out) {
        memcpy(out, j.buf, j.len);
        out[j.len] = '\0';
        r->json_cache     = out;
        r->json_cache_len = j.len;
    } else {
        r->json_cache     = "";
        r->json_cache_len = 0;
    }
    free(j.buf);
    r->json_built = 1;
}

const char *wgsl_module_json(const WGSLResult *r) {
    if (!r) return "";
    /* Only emit JSON when the module passed the full check.  Per the
     * header contract: empty string when the check failed. */
    if (!r->pipeline_ok) return "";
    /* Cast away const for lazy initialisation — the cache is logically
     * derived state, and `wgsl_module_json` is documented as
     * idempotent. */
    WGSLResult *m = (WGSLResult *)r;
    if (!m->json_built) build_module_json(m);
    return m->json_cache ? m->json_cache : "";
}

size_t wgsl_module_json_len(const WGSLResult *r) {
    if (!r) return 0;
    if (!r->pipeline_ok) return 0;
    WGSLResult *m = (WGSLResult *)r;
    if (!m->json_built) build_module_json(m);
    return m->json_cache_len;
}

/* ── Diagnostics ──────────────────────────────────────────────────── */

int wgsl_diagnostic_count(const WGSLResult *r) {
    if (!r) return 0;
    return (int)r->diag.count;
}

const WGSLDiagnostic *wgsl_diagnostic(const WGSLResult *r, int index) {
    if (!r || index < 0 || (size_t)index >= r->diag.count) return NULL;
    return &r->diag.items[index];
}

/* ── Lexer (raw tokens for syntax highlighters) ───────────────────── */

/* Translate `WGSLTokenKind` → `WGSLLexTokenKind` for the public stream.
 * Iter B will additionally promote IDENT tokens to STRUCT / FUNCTION /
 * PARAMETER / FIELD / VAR / LET / CONST / OVERRIDE / TYPE_ALIAS based
 * on resolver bindings. */
static WGSLLexTokenKind classify_raw_token(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_IDENT:           return WGSL_LEX_IDENTIFIER;
    case WGSL_TOK_INT_LIT:
    case WGSL_TOK_FLOAT_LIT:       return WGSL_LEX_NUMBER;

    /* Keywords. */
    case WGSL_TOK_KW_ALIAS:
    case WGSL_TOK_KW_BREAK:
    case WGSL_TOK_KW_CASE:
    case WGSL_TOK_KW_CONST:
    case WGSL_TOK_KW_CONST_ASSERT:
    case WGSL_TOK_KW_CONTINUE:
    case WGSL_TOK_KW_CONTINUING:
    case WGSL_TOK_KW_DEFAULT:
    case WGSL_TOK_KW_DISCARD:
    case WGSL_TOK_KW_ELSE:
    case WGSL_TOK_KW_FN:
    case WGSL_TOK_KW_FOR:
    case WGSL_TOK_KW_IF:
    case WGSL_TOK_KW_LET:
    case WGSL_TOK_KW_LOOP:
    case WGSL_TOK_KW_OVERRIDE:
    case WGSL_TOK_KW_RETURN:
    case WGSL_TOK_KW_STRUCT:
    case WGSL_TOK_KW_SWITCH:
    case WGSL_TOK_KW_VAR:
    case WGSL_TOK_KW_WHILE:        return WGSL_LEX_KEYWORD;

    /* Boolean literals are spec-keywords; treat them as builtin values
     * so editors can render `true` / `false` distinctly from `if`. */
    case WGSL_TOK_KW_TRUE:
    case WGSL_TOK_KW_FALSE:        return WGSL_LEX_BUILTIN_VALUE;

    case WGSL_TOK_KW_ENABLE:
    case WGSL_TOK_KW_REQUIRES:
    case WGSL_TOK_KW_DIAGNOSTIC:   return WGSL_LEX_DIRECTIVE;

    /* Punctuation. */
    case WGSL_TOK_LPAREN:
    case WGSL_TOK_RPAREN:
    case WGSL_TOK_LBRACE:
    case WGSL_TOK_RBRACE:
    case WGSL_TOK_LBRACKET:
    case WGSL_TOK_RBRACKET:
    case WGSL_TOK_COMMA:
    case WGSL_TOK_SEMICOLON:
    case WGSL_TOK_COLON:
    case WGSL_TOK_DOT:
    case WGSL_TOK_UNDERSCORE:      return WGSL_LEX_PUNCT;

    case WGSL_TOK_AT:              return WGSL_LEX_ATTRIBUTE;

    /* Template-list start / end (post §3.9 disambiguation). */
    case WGSL_TOK_TEMPLATE_START:
    case WGSL_TOK_TEMPLATE_END:    return WGSL_LEX_TEMPLATE_DELIM;

    /* Operators (everything else is an operator at lex level). */
    case WGSL_TOK_AMP:
    case WGSL_TOK_AMP_AMP:
    case WGSL_TOK_AMP_EQUAL:
    case WGSL_TOK_PIPE:
    case WGSL_TOK_PIPE_PIPE:
    case WGSL_TOK_PIPE_EQUAL:
    case WGSL_TOK_CARET:
    case WGSL_TOK_CARET_EQUAL:
    case WGSL_TOK_TILDE:
    case WGSL_TOK_BANG:
    case WGSL_TOK_BANG_EQUAL:
    case WGSL_TOK_PLUS:
    case WGSL_TOK_PLUS_PLUS:
    case WGSL_TOK_PLUS_EQUAL:
    case WGSL_TOK_MINUS:
    case WGSL_TOK_MINUS_MINUS:
    case WGSL_TOK_MINUS_EQUAL:
    case WGSL_TOK_ARROW:
    case WGSL_TOK_STAR:
    case WGSL_TOK_STAR_EQUAL:
    case WGSL_TOK_SLASH:
    case WGSL_TOK_SLASH_EQUAL:
    case WGSL_TOK_PERCENT:
    case WGSL_TOK_PERCENT_EQUAL:
    case WGSL_TOK_LESS:
    case WGSL_TOK_LESS_EQUAL:
    case WGSL_TOK_LESS_LESS:
    case WGSL_TOK_LESS_LESS_EQUAL:
    case WGSL_TOK_GREATER:
    case WGSL_TOK_GREATER_EQUAL:
    case WGSL_TOK_GREATER_GREATER:
    case WGSL_TOK_GREATER_GREATER_EQUAL:
    case WGSL_TOK_EQUAL:
    case WGSL_TOK_EQUAL_EQUAL:     return WGSL_LEX_OPERATOR;

    case WGSL_TOK_LINE_COMMENT:
    case WGSL_TOK_BLOCK_COMMENT:   return WGSL_LEX_COMMENT;
    case WGSL_TOK_BLANKSPACE:      return WGSL_LEX_WHITESPACE;

    case WGSL_TOK_INVALID:
    case WGSL_TOK_EOF:
    default:                       return WGSL_LEX_UNKNOWN;
    }
}

/* Internal helper: copy a `WGSLLexResult` into a heap-owned
 * `WGSLLexToken[]`, applying `classify` to upgrade idents. */
typedef WGSLLexTokenKind (*ClassifyFn)(
    const WGSLToken *t, const WGSLSource *src, void *ctx);

static int collect_lex_tokens(
    const WGSLSource *source, const WGSLLexResult *lex,
    ClassifyFn classify, void *ctx,
    WGSLLexToken **out_tokens, int *out_count)
{
    /* Skip the trailing EOF sentinel; emit blankspace + comment + real
     * tokens.  Callers can filter trivia client-side if they want. */
    size_t total = lex->count;
    if (total > 0 && lex->tokens[total - 1].kind == WGSL_TOK_EOF) total -= 1;

    WGSLLexToken *out = NULL;
    if (total > 0) {
        out = (WGSLLexToken *)malloc(total * sizeof *out);
        if (!out) return 0;
    }

    for (size_t i = 0; i < total; i++) {
        const WGSLToken *t = &lex->tokens[i];
        out[i].kind   = classify(t, source, ctx);
        out[i].offset = t->span.offset;
        out[i].length = t->span.length;
        out[i].line   = t->line;
        out[i].column = t->column;
    }
    *out_tokens = out;
    *out_count  = (int)total;
    return 1;
}

static WGSLLexTokenKind raw_classify(
    const WGSLToken *t, const WGSLSource *src, void *ctx)
{
    (void)src; (void)ctx;
    return classify_raw_token(t->kind);
}

int wgsl_lex(const char *src, WGSLLexToken **out_tokens, int *out_count) {
    if (!out_tokens || !out_count) return 0;
    *out_tokens = NULL;
    *out_count  = 0;

    wgsl_utf8_init();

    WGSLArena   arena;     wgsl_arena_init(&arena);
    WGSLDiagBag diag;      wgsl_diag_init(&diag);
    WGSLSource  source;
    if (!wgsl_source_init(&source, src ? src : "", src ? strlen(src) : 0)) {
        wgsl_diag_destroy(&diag);
        wgsl_arena_destroy(&arena);
        return 0;
    }
    WGSLLexResult lex = {0};
    int lex_ok = wgsl_tokenize(&source, &arena, &diag, &lex);

    int rc = lex_ok ? collect_lex_tokens(
        &source, &lex, raw_classify, NULL, out_tokens, out_count) : 0;

    wgsl_diag_destroy(&diag);
    wgsl_source_destroy(&source);
    wgsl_arena_destroy(&arena);
    return rc;
}

/* ── Semantic-token overlay — Iter B ──────────────────────────────── */

/* Per-decl-name span we emit while walking the AST.  Callers merge
 * these into the raw token stream by exact (offset, length) match. */
typedef struct {
    uint32_t          offset;
    uint32_t          length;
    WGSLLexTokenKind  kind;
} SemSpan;

typedef struct {
    SemSpan *items;
    size_t   count;
    size_t   capacity;
} SemList;

static void sem_push(SemList *L, uint32_t off, uint32_t len, WGSLLexTokenKind k) {
    if (len == 0) return;
    if (L->count == L->capacity) {
        size_t cap = L->capacity ? L->capacity * 2 : 64;
        SemSpan *g = (SemSpan *)realloc(L->items, cap * sizeof *g);
        if (!g) return;
        L->items = g;
        L->capacity = cap;
    }
    L->items[L->count++] = (SemSpan){ off, len, k };
}

static WGSLLexTokenKind kind_for_sym(const WGSLSymbol *s) {
    switch ((WGSLSymKind)s->kind) {
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_PREDECLARED_TYPEGEN:    return WGSL_LEX_TYPE;
    case WGSL_SYM_PREDECLARED_FN:         return WGSL_LEX_BUILTIN_FUNC;
    case WGSL_SYM_PREDECLARED_VALUE:      return WGSL_LEX_BUILTIN_VALUE;
    case WGSL_SYM_PREDECLARED_ADDR_SPACE:
    case WGSL_SYM_PREDECLARED_ACCESS_MODE: return WGSL_LEX_KEYWORD;
    case WGSL_SYM_TYPE_ALIAS:             return WGSL_LEX_TYPE_ALIAS;
    case WGSL_SYM_STRUCT:                 return WGSL_LEX_STRUCT_NAME;
    case WGSL_SYM_FUNCTION:               return WGSL_LEX_FUNCTION_NAME;
    case WGSL_SYM_VAR:                    return WGSL_LEX_VAR;
    case WGSL_SYM_LET:                    return WGSL_LEX_LET;
    case WGSL_SYM_CONST:                  return WGSL_LEX_CONST;
    case WGSL_SYM_OVERRIDE:               return WGSL_LEX_OVERRIDE;
    case WGSL_SYM_PARAM:                  return WGSL_LEX_PARAMETER;
    default:                              return WGSL_LEX_IDENTIFIER;
    }
}

/* Walk the AST and emit `(offset, length, kind)` triples for:
 *   - every EXPR_IDENT bound to a resolver symbol — uses `payload[2]`
 *     as `WGSLSymbol *`
 *   - every decl-shaped node's name span — `payload[0]` carries
 *     `(offset | length << 32)` for DECL_FUNCTION / PARAM / VAR / LET /
 *     CONST / OVERRIDE / STRUCT / ALIAS / STRUCT_MEMBER. */
static void collect_sem_spans(WGSLNode *n, SemList *L) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *s = wgsl_node_resolved_symbol(n);
        if (s) {
            sem_push(L, n->span_offset, n->span_length, kind_for_sym(s));
        }
        break;
    }

    case WGSL_NODE_DECL_FUNCTION: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_FUNCTION_NAME);
        break;
    }
    case WGSL_NODE_DECL_PARAM: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_PARAMETER);
        break;
    }
    case WGSL_NODE_DECL_VAR: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_VAR);
        break;
    }
    case WGSL_NODE_DECL_LET: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_LET);
        break;
    }
    case WGSL_NODE_DECL_CONST: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_CONST);
        break;
    }
    case WGSL_NODE_DECL_OVERRIDE: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_OVERRIDE);
        break;
    }
    case WGSL_NODE_DECL_STRUCT: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_STRUCT_NAME);
        break;
    }
    case WGSL_NODE_DECL_STRUCT_MEMBER: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_FIELD);
        break;
    }
    case WGSL_NODE_DECL_ALIAS: {
        uint32_t no = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(n->payload[0] >> 32);
        sem_push(L, no, nl, WGSL_LEX_TYPE_ALIAS);
        break;
    }

    case WGSL_NODE_EXPR_MEMBER: {
        /* `obj.member` — children[0] = obj (recursed below), children[1] =
         * the member-name ident.  Tag the member as FIELD. */
        if (n->child_count >= 2 && n->children[1] &&
            n->children[1]->kind == WGSL_NODE_EXPR_IDENT)
        {
            WGSLNode *m = n->children[1];
            sem_push(L, m->span_offset, m->span_length, WGSL_LEX_FIELD);
        }
        if (n->child_count >= 1) collect_sem_spans(n->children[0], L);
        return;       /* children[1] handled inline; don't recurse into it */
    }

    default:
        break;
    }
    for (uint32_t i = 0; i < n->child_count; i++) {
        collect_sem_spans(n->children[i], L);
    }
}

typedef struct {
    const SemSpan *spans;
    size_t         count;
} SemCtx;

static int span_cmp_offset(const void *a, const void *b) {
    const SemSpan *x = (const SemSpan *)a;
    const SemSpan *y = (const SemSpan *)b;
    if (x->offset < y->offset) return -1;
    if (x->offset > y->offset) return  1;
    return 0;
}

static WGSLLexTokenKind sem_classify(
    const WGSLToken *t, const WGSLSource *src, void *vctx)
{
    (void)src;
    SemCtx *ctx = (SemCtx *)vctx;
    /* Binary search by offset for an exact-span match. */
    if (ctx->count > 0) {
        size_t lo = 0, hi = ctx->count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (ctx->spans[mid].offset < t->span.offset) lo = mid + 1;
            else                                          hi = mid;
        }
        for (size_t i = lo;
             i < ctx->count && ctx->spans[i].offset == t->span.offset;
             i++)
        {
            if (ctx->spans[i].length == t->span.length) {
                return ctx->spans[i].kind;
            }
        }
    }
    return classify_raw_token(t->kind);
}

int wgsl_semantic_tokens(const char *src, WGSLLexToken **out_tokens, int *out_count) {
    if (!out_tokens || !out_count) return 0;
    *out_tokens = NULL;
    *out_count  = 0;

    /* Run the full pipeline so the resolver bindings are in place; if
     * parsing or resolving fails we degrade gracefully to raw lex. */
    WGSLResult *r = wgsl_check(src);
    if (!r) return 0;

    SemList L = {0};
    if (r->ast.root) collect_sem_spans(r->ast.root, &L);
    /* Sort so we can binary-search by offset. */
    if (L.count > 1) qsort(L.items, L.count, sizeof *L.items, span_cmp_offset);

    SemCtx ctx = { L.items, L.count };
    int rc = collect_lex_tokens(
        &r->source, &r->lex, sem_classify, &ctx, out_tokens, out_count);

    free(L.items);
    wgsl_free(r);
    return rc;
}

void wgsl_lex_free(WGSLLexToken *tokens) {
    free(tokens);
}

/* ── Hover / goto-def — Iter B ────────────────────────────────────── */

/* Find the smallest EXPR_IDENT span that contains `offset`.  Returns
 * NULL when no ident covers the offset. */
static const WGSLNode *find_ident_at(const WGSLNode *n, uint32_t offset) {
    if (!n) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
    {
        return NULL;
    }
    /* Recurse into children first — the deepest hit is the smallest. */
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = find_ident_at(n->children[i], offset);
        if (hit) return hit;
    }
    if (n->kind == WGSL_NODE_EXPR_IDENT) return n;
    return NULL;
}

/* Find a user-declared symbol whose decl-name span contains `offset`.
 * Decl-name spans live in `payload[0] = (off | len << 32)` for every
 * kind the resolver binds. */
static WGSLSymbol *find_decl_at(const WGSLResolver *res, uint32_t offset) {
    if (!res) return NULL;
    for (size_t i = 0; i < res->all_decl_count; i++) {
        WGSLSymbol *s = res->all_decls[i];
        if (!s || !s->ast) continue;
        uint32_t no = (uint32_t)(s->ast->payload[0] & 0xFFFFFFFFu);
        uint32_t nl = (uint32_t)(s->ast->payload[0] >> 32);
        if (nl == 0) continue;
        if (offset >= no && offset < no + nl) return s;
    }
    return NULL;
}

/* Format a TypeInfo into an arena-owned NUL-terminated string. */
static const char *fmt_type(WGSLArena *arena, WGSLTypeInfo *t) {
    if (!t) return "";
    size_t need = wgsl_type_format(t, NULL, 0);
    char *buf = (char *)wgsl_arena_alloc(arena, need + 1);
    if (!buf) return "";
    wgsl_type_format(t, buf, need + 1);
    return buf;
}

/* Empty `WGSLHover` value with non-NULL string fields so callers can
 * dereference unconditionally. */
static WGSLHover empty_hover(void) {
    WGSLHover h = {0};
    h.type_repr = "";
    h.signature = "";
    h.doc       = "";
    return h;
}

WGSLHover wgsl_hover_at(const WGSLResult *r, uint32_t offset) {
    if (!r || !r->ast.root) return empty_hover();

    /* First try the use-site path: an EXPR_IDENT whose `payload[2]`
     * already carries the resolver-bound symbol. */
    WGSLSymbol *sym = NULL;
    uint32_t span_off = 0, span_len = 0;
    const WGSLNode *id = find_ident_at(r->ast.root, offset);
    if (id) {
        sym = wgsl_node_resolved_symbol(id);
        span_off = id->span_offset;
        span_len = id->span_length;
    }
    /* Fall back to the decl-name path (the `x` in `let x: i32 = …`,
     * the `Foo` in `struct Foo { … }`, etc.). */
    if (!sym) {
        sym = find_decl_at(&r->res, offset);
        if (sym && sym->ast) {
            span_off = (uint32_t)(sym->ast->payload[0] & 0xFFFFFFFFu);
            span_len = (uint32_t)(sym->ast->payload[0] >> 32);
        }
    }
    if (!sym) return empty_hover();

    WGSLHover h     = empty_hover();
    h.present       = 1;
    h.symbol_offset = span_off;
    h.symbol_length = span_len;
    if (sym->type) {
        h.type_repr = fmt_type((WGSLArena *)&r->arena, sym->type);
    }
    return h;
}

WGSLDefinition wgsl_definition_at(const WGSLResult *r, uint32_t offset) {
    WGSLDefinition d = {0};
    if (!r || !r->ast.root) return d;

    WGSLSymbol *sym = NULL;
    const WGSLNode *id = find_ident_at(r->ast.root, offset);
    if (id) sym = wgsl_node_resolved_symbol(id);
    if (!sym) sym = find_decl_at(&r->res, offset);
    if (!sym || !sym->ast) return d;     /* predeclared symbols → absent */

    WGSLNode *decl = sym->ast;
    uint32_t no = (uint32_t)(decl->payload[0] & 0xFFFFFFFFu);
    uint32_t nl = (uint32_t)(decl->payload[0] >> 32);
    if (nl == 0) return d;
    d.present = 1;
    d.offset  = no;
    d.length  = nl;
    return d;
}

/* ── Pointer-out wrappers (wasm FFI ergonomics) ───────────────────── */

void wgsl_hover_at_into(
    const WGSLResult *r, uint32_t offset, WGSLHover *out)
{
    if (!out) return;
    *out = wgsl_hover_at(r, offset);
}

void wgsl_definition_at_into(
    const WGSLResult *r, uint32_t offset, WGSLDefinition *out)
{
    if (!out) return;
    *out = wgsl_definition_at(r, offset);
}

/* ── Project / preamble injection (FS-aware variants gated) ───────── */

#ifndef WGSL_NO_FS

/* Read the entire file at `path` into a freshly-malloc'd buffer.
 * On success returns the buffer (caller free); writes the byte count
 * to `*out_len`.  On failure returns NULL. */
static char *read_file_all(const char *path, size_t *out_len) {
    if (out_len) *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[n] = '\0';
    if (out_len) *out_len = (size_t)n;
    return buf;
}

static char *path_join(const char *a, const char *b) {
    size_t la = a ? strlen(a) : 0;
    size_t lb = b ? strlen(b) : 0;
    int    sep = (la > 0 && a[la - 1] != '/') ? 1 : 0;
    char  *out = (char *)malloc(la + (size_t)sep + lb + 1);
    if (!out) return NULL;
    if (la > 0) memcpy(out, a, la);
    if (sep)    out[la] = '/';
    if (lb > 0) memcpy(out + la + (size_t)sep, b, lb);
    out[la + (size_t)sep + lb] = '\0';
    return out;
}

WGSLProject *wgsl_project_open(const char *root_dir, char *err, size_t err_len) {
    if (err && err_len > 0) err[0] = '\0';
    if (!root_dir) {
        if (err && err_len > 0) snprintf(err, err_len, "root_dir is NULL");
        return NULL;
    }
    char *toml_path = path_join(root_dir, "wgslconfig.toml");
    if (!toml_path) {
        if (err && err_len > 0) snprintf(err, err_len, "out of memory");
        return NULL;
    }
    size_t toml_len = 0;
    char *toml_text = read_file_all(toml_path, &toml_len);
    free(toml_path);
    if (!toml_text) {
        if (err && err_len > 0) snprintf(err, err_len, "could not read wgslconfig.toml");
        return NULL;
    }
    WGSLProject *p = wgsl_project_open_from_string(toml_text, toml_len, err, err_len);
    free(toml_text);
    return p;
}

WGSLResult *wgsl_check_in_project(
    const WGSLProject *p, const char *root_dir, const char *file_path)
{
    /* Read source. */
    char *src_full = path_join(root_dir, file_path);
    if (!src_full) return NULL;
    size_t src_len = 0;
    char *src = read_file_all(src_full, &src_len);
    free(src_full);
    if (!src) return NULL;

    /* Resolve preamble list for this file path. */
    int preamble_count = 0;
    const char *const *preamble_files =
        p ? wgsl_project_match(p, file_path, &preamble_count) : NULL;

    /* Concatenate all preamble files (with trailing newlines). */
    char  *preamble = NULL;
    size_t plen     = 0;
    for (int i = 0; i < preamble_count; i++) {
        char  *full = path_join(root_dir, preamble_files[i]);
        if (!full) { free(preamble); free(src); return NULL; }
        size_t one_len = 0;
        char  *one = read_file_all(full, &one_len);
        free(full);
        if (!one) { free(preamble); free(src); return NULL; }
        int sep = (one_len > 0 && one[one_len - 1] != '\n') ? 1 : 0;
        char *grown = (char *)realloc(preamble, plen + one_len + (size_t)sep);
        if (!grown) { free(one); free(preamble); free(src); return NULL; }
        preamble = grown;
        memcpy(preamble + plen, one, one_len);
        plen += one_len;
        if (sep) preamble[plen++] = '\n';
        free(one);
    }

    WGSLResult *r = wgsl_check_with_preamble(preamble, plen, src, src_len);
    free(preamble);
    free(src);
    return r;
}

#endif /* WGSL_NO_FS */
