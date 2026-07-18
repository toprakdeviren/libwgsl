/**
 * @file helpers.c — shared LSP buffers and symbol lookup
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


/* Language-service queries.
 *
 * Document symbols, references/rename, folding ranges, completions,
 * and signature help.  Built on the same AST + resolver state as
 * hover / goto-def (scopes + all_decls live until `wgsl_free`). */

/* shared symbol lookup (same rules as hover / definition). */

WGSLSymbol *lsp_symbol_at(const WGSLResult *r, uint32_t offset) {
    if (!r || !r->ast.root) return NULL;
    const WGSLNode *id = wgsl_api_find_ident_at(r->ast.root, offset, 0);
    if (id) {
        WGSLSymbol *s = wgsl_node_resolved_symbol(id);
        if (s) return s;
    }
    return wgsl_api_find_decl_at(&r->res, r->ast.root, offset);
}

void decl_name_span(const WGSLSymbol *s, uint32_t *off, uint32_t *len) {
    *off = 0; *len = 0;
    if (!s || !s->ast) return;
    *off = wgsl_node_name_span(s->ast).offset;
    *len = wgsl_node_name_span(s->ast).length;
}

/* growable buffers. */

/* OutlineBuf: lsp_priv.h */

int ob_push(OutlineBuf *b, WGSLOutlineSymbol s) {
    if (b->n + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 16;
        WGSLOutlineSymbol *g = (WGSLOutlineSymbol *)realloc(b->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        b->v = g; b->cap = nc;
    }
    b->v[b->n++] = s;
    return 1;
}

/* RefBuf: lsp_priv.h */

int rb_push(RefBuf *b, uint32_t off, uint32_t len, int is_def) {
    if (b->n + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 16;
        WGSLReference *g = (WGSLReference *)realloc(b->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        b->v = g; b->cap = nc;
    }
    b->v[b->n].offset = off;
    b->v[b->n].length = len;
    b->v[b->n].is_definition = is_def;
    b->n++;
    return 1;
}

/* FoldBuf: lsp_priv.h */

int fb_push(FoldBuf *b, uint32_t start, uint32_t end) {
    if (end <= start) return 1;
    if (b->n + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 32;
        WGSLFoldingRange *g = (WGSLFoldingRange *)realloc(b->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        b->v = g; b->cap = nc;
    }
    b->v[b->n].start_offset = start;
    b->v[b->n].end_offset = end;
    b->n++;
    return 1;
}

/* CompBuf: lsp_priv.h */

char *lsp_strdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n + 1);
    return p;
}

char *lsp_strndup(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    if (n) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

int cb_push(CompBuf *b, WGSLCompletionKind kind,
                   const char *label, const char *detail)
{
    if (!label || !label[0]) return 1;
    if (b->n + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 64;
        WGSLCompletionItem *g = (WGSLCompletionItem *)realloc(
            b->v, (size_t)nc * sizeof *g);
        if (!g) return 0;
        b->v = g; b->cap = nc;
    }
    char *l = lsp_strdup(label);
    char *d = lsp_strdup(detail ? detail : "");
    if (!l || !d) { free(l); free(d); return 0; }
    b->v[b->n].kind = kind;
    b->v[b->n].label = l;
    b->v[b->n].detail = d;
    b->n++;
    return 1;
}

int cb_has_label(const CompBuf *b, const char *label) {
    if (!label) return 0;
    for (int i = 0; i < b->n; i++)
        if (strcmp(b->v[i].label, label) == 0) return 1;
    return 0;
}
