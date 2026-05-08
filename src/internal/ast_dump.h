/**
 * @file ast_dump.h — Human-readable AST printer.
 *
 * Walks an AST rooted at `root` and writes an indented S-expression-ish
 * tree to `out`.  Used by the parser tests and by `wgsl-cli --dump-ast`
 * (added in Phase 9 LSP work).
 *
 * Source spans are embedded so the dump round-trips back to the byte
 * locations.  Snippet length is clamped to avoid noisy output.
 */
#ifndef WGSL_INTERNAL_AST_DUMP_H
#define WGSL_INTERNAL_AST_DUMP_H

#include <stdio.h>

#include "internal/ast.h"
#include "internal/source.h"

/** Pretty-print the AST.  Pass `stdout` or any open `FILE *`. */
void wgsl_ast_dump(const WGSLNode *root, const WGSLSource *src, FILE *out);

#endif /* WGSL_INTERNAL_AST_DUMP_H */
