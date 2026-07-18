/**
 * @file lsp_priv.h — shared helpers among src/api/lsp/ TUs
 */
#ifndef WGSL_INTERNAL_LSP_PRIV_H
#define WGSL_INTERNAL_LSP_PRIV_H

#include "wgsl.h"
#include "internal/result.h"
#include "internal/ast.h"
#include "internal/resolver.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    WGSLOutlineSymbol *v;
    int n, cap;
} OutlineBuf;

typedef struct {
    WGSLReference *v;
    int n, cap;
} RefBuf;

typedef struct {
    WGSLFoldingRange *v;
    int n, cap;
} FoldBuf;

typedef struct {
    WGSLCompletionItem *v;
    int n, cap;
} CompBuf;

typedef struct {
    WGSLCodeAction *v;
    int n, cap;
} ActBuf;

WGSLSymbol *lsp_symbol_at(const WGSLResult *r, uint32_t offset);
void decl_name_span(const WGSLSymbol *s, uint32_t *off, uint32_t *len);
int ob_push(OutlineBuf *b, WGSLOutlineSymbol s);
int rb_push(RefBuf *b, uint32_t off, uint32_t len, int is_def);
int fb_push(FoldBuf *b, uint32_t start, uint32_t end);
char *lsp_strdup(const char *s);
char *lsp_strndup(const char *s, size_t n);
int cb_push(CompBuf *b, WGSLCompletionKind kind, const char *label, const char *detail);
int cb_has_label(const CompBuf *b, const char *label);

#endif /* WGSL_INTERNAL_LSP_PRIV_H */
