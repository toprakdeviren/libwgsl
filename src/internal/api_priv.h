/**
 * @file api_priv.h — shared helpers among public-API TUs under src/api/.
 *
 * Position → AST/symbol helpers used by both hover/definition and the
 * richer LSP query surface.
 */
#ifndef WGSL_INTERNAL_API_PRIV_H
#define WGSL_INTERNAL_API_PRIV_H

#include "internal/ast.h"
#include "internal/resolver.h"

#include <stdint.h>

const WGSLNode *wgsl_api_find_ident_at(
    const WGSLNode *n, uint32_t offset, int depth);
WGSLSymbol *wgsl_api_find_decl_at(
    const WGSLResolver *res, const WGSLNode *root, uint32_t offset);

#endif /* WGSL_INTERNAL_API_PRIV_H */
