/**
 * @file completions.c — completions at offset
 */
#include "wgsl.h"
#include "internal/result.h"
#include "internal/api_priv.h"
#include "internal/lsp_priv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"
#include "internal/ast.h"
#include "internal/resolver.h"
#include "internal/types.h"
#include "internal/check.h"
#include "internal/consteval.h"


/* completions. */

static const char *const LSP_KEYWORDS[] = {
    "fn", "var", "let", "const", "override", "struct", "alias",
    "if", "else", "switch", "case", "default", "loop", "for", "while",
    "break", "continue", "return", "discard", "continuing",
    "enable", "requires", "diagnostic", "const_assert",
    "true", "false", "bitcast",
    NULL
};

static WGSLCompletionKind completion_kind_for_sym(const WGSLSymbol *s) {
    if (!s) return WGSL_COMPLETION_VARIABLE;
    switch ((WGSLSymKind)s->kind) {
    case WGSL_SYM_FUNCTION:
    case WGSL_SYM_PREDECLARED_FN:
        return WGSL_COMPLETION_FUNCTION;
    case WGSL_SYM_PREDECLARED_TYPE:
    case WGSL_SYM_PREDECLARED_TYPEGEN:
    case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
    case WGSL_SYM_TYPE_ALIAS:
    case WGSL_SYM_STRUCT:
        return WGSL_COMPLETION_TYPE;
    case WGSL_SYM_PREDECLARED_VALUE:
        return WGSL_COMPLETION_VALUE;
    default:
        return WGSL_COMPLETION_VARIABLE;
    }
}

static int prefix_ok(const char *label, const char *pfx, size_t pfx_len) {
    if (!pfx || pfx_len == 0) return 1;
    if (!label) return 0;
    size_t ll = strlen(label);
    if (ll < pfx_len) return 0;
    return memcmp(label, pfx, pfx_len) == 0;
}

/* Type detail string (heap). */
static char *sym_detail(const WGSLSymbol *s) {
    if (!s || !s->type) return lsp_strdup("");
    char tmp[160];
    size_t n = wgsl_type_format(s->type, tmp, sizeof tmp);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    return lsp_strndup(tmp, n);
}

/* Find enclosing function node whose span contains offset. */
static const WGSLNode *enclosing_function(const WGSLNode *n, uint32_t offset,
                                          int depth)
{
    if (!n || depth > WGSL_MAX_AST_DEPTH) return NULL;
    if (offset < n->span_offset ||
        offset >= n->span_offset + n->span_length)
        return NULL;
    const WGSLNode *best = (n->kind == WGSL_NODE_DECL_FUNCTION) ? n : NULL;
    for (uint32_t i = 0; i < n->child_count; i++) {
        const WGSLNode *hit = enclosing_function(n->children[i], offset, depth + 1);
        if (hit) best = hit;
    }
    return best;
}

/* Identifier prefix under/just before the cursor for filtering. */
static void completion_prefix(const WGSLResult *r, uint32_t offset,
                              const char **pfx, size_t *pfx_len)
{
    *pfx = NULL; *pfx_len = 0;
    if (!r || !r->src_copy || offset > r->src_len) return;
    const char *s = r->src_copy;
    uint32_t i = offset;
    /* If sitting on an ident char, walk back to its start. */
    if (i > 0 && i <= r->src_len) {
        uint32_t j = i;
        if (j < r->src_len) {
            unsigned char c = (unsigned char)s[j];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_'))
            {
                /* past end of ident — use chars before cursor */
            } else {
                /* mid-ident */
            }
        }
        while (j > 0) {
            unsigned char c = (unsigned char)s[j - 1];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_')
                j--;
            else break;
        }
        if (j < i) {
            *pfx = s + j;
            *pfx_len = (size_t)(i - j);
        }
    }
}

static int add_sym_completion(CompBuf *b, const WGSLSymbol *s,
                              const char *pfx, size_t pfx_len)
{
    if (!s || !s->name || s->name_len == 0) return 1;
    char label[128];
    size_t n = s->name_len < sizeof label - 1 ? s->name_len : sizeof label - 1;
    memcpy(label, s->name, n);
    label[n] = '\0';
    if (!prefix_ok(label, pfx, pfx_len)) return 1;
    if (cb_has_label(b, label)) return 1;
    char *detail = sym_detail(s);
    int ok = cb_push(b, completion_kind_for_sym(s), label, detail ? detail : "");
    free(detail);
    return ok;
}

int wgsl_completions_at(const WGSLResult *r, uint32_t offset,
                        WGSLCompletionItem **out, int *out_count)
{
    if (out) *out = NULL;
    if (out_count) *out_count = 0;
    if (!r || !out || !out_count) return 0;

    const char *pfx = NULL;
    size_t pfx_len = 0;
    completion_prefix(r, offset, &pfx, &pfx_len);

    CompBuf b = {0};

    /* Keywords */
    for (int i = 0; LSP_KEYWORDS[i]; i++) {
        if (!prefix_ok(LSP_KEYWORDS[i], pfx, pfx_len)) continue;
        if (!cb_push(&b, WGSL_COMPLETION_KEYWORD, LSP_KEYWORDS[i], "")) {
            wgsl_completions_free(b.v, b.n); return 0;
        }
    }

    /* Predeclared + module scopes (still live on the result). */
    const WGSLScope *scopes[2] = { &r->res.predeclared, &r->res.module };
    for (int si = 0; si < 2; si++) {
        const WGSLScope *sc = scopes[si];
        for (size_t i = 0; i < sc->count; i++) {
            if (!add_sym_completion(&b, sc->items[i], pfx, pfx_len)) {
                wgsl_completions_free(b.v, b.n); return 0;
            }
        }
    }

    /* Locals: params + body decls of the enclosing function whose names
     * start before the cursor (so we don't suggest yet-undeclared lets). */
    if (r->ast.root) {
        const WGSLNode *fn = enclosing_function(r->ast.root, offset, 0);
        if (fn) {
            uint32_t FA = wgsl_fn_attr_count(fn);
            uint32_t P  = wgsl_fn_param_count(fn);
            for (uint32_t k = 0; k < P; k++) {
                uint32_t ci = FA + k;
                if (ci >= fn->child_count) break;
                const WGSLNode *p = fn->children[ci];
                if (!p || p->kind != WGSL_NODE_DECL_PARAM) continue;
                WGSLSymbol *s = wgsl_resolver_symbol_for_decl(&r->res, p);
                if (s) {
                    if (!add_sym_completion(&b, s, pfx, pfx_len)) {
                        wgsl_completions_free(b.v, b.n); return 0;
                    }
                }
            }
            /* Walk all_decls whose AST is nested under fn and name offset < cursor. */
            for (size_t di = 0; di < r->res.all_decl_count; di++) {
                WGSLSymbol *s = r->res.all_decls[di];
                if (!s || !s->ast) continue;
                if (s->kind != WGSL_SYM_LET && s->kind != WGSL_SYM_VAR &&
                    s->kind != WGSL_SYM_CONST)
                    continue;
                const WGSLNode *d = s->ast;
                if (d->span_offset < fn->span_offset ||
                    d->span_offset >= fn->span_offset + fn->span_length)
                    continue;
                uint32_t no, nl;
                decl_name_span(s, &no, &nl);
                if (nl == 0 || no >= offset) continue;
                if (!add_sym_completion(&b, s, pfx, pfx_len)) {
                    wgsl_completions_free(b.v, b.n); return 0;
                }
            }
        }
    }

    *out = b.v;
    *out_count = b.n;
    return 1;
}

void wgsl_completions_free(WGSLCompletionItem *items, int count) {
    if (!items) return;
    for (int i = 0; i < count; i++) {
        free(items[i].label);
        free(items[i].detail);
    }
    free(items);
}
