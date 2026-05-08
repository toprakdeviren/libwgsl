/**
 * @file parser.h — WGSL recursive-descent + Pratt parser.
 *
 * Consumes a `WGSLLexResult` (post-§3.9) and produces a `WGSLAst`
 * rooted at a `WGSL_NODE_TRANSLATION_UNIT`.  Every node is allocated
 * from the supplied arena.  Errors are emitted on the diag bag; the
 * parser recovers at the next `;` or `}` to surface multiple errors
 * per pass.
 *
 * Round A scope (this iteration):
 *   - translation_unit  — collects function decls; directives are
 *                          accepted-and-skipped, struct/alias/const/
 *                          override/var owed to Round C.
 *   - function_decl     — `attribute* fn IDENT '(' params ')'
 *                          ('->' attribute* type)? compound_statement`
 *   - parameter         — `attribute* IDENT ':' type_specifier`
 *   - type_specifier    — template_elaborated_ident
 *   - statements        — compound, let, return, expression-statement
 *   - expressions       — Pratt subset:
 *                            literal · ident · paren · call · index
 *                            · member · unary · binary (+ - * / % == != etc.)
 *                            · template_elaborated_ident
 *
 * Round B will expand to the full §8.19 precedence table, more
 * statements (if/switch/loop/for/while), and other global decls.
 *
 * Returned AST nodes are 48-byte WGSLNode objects.  Children layout is
 * kind-dependent and documented per-parser; see also `ast_dump.c`.
 */
#ifndef WGSL_INTERNAL_PARSER_H
#define WGSL_INTERNAL_PARSER_H

#include "internal/arena.h"
#include "internal/ast.h"
#include "internal/diag.h"
#include "internal/lexer.h"
#include "internal/source.h"

/** A parsed WGSL module. */
typedef struct WGSLAst {
    WGSLNode *root;     /* WGSL_NODE_TRANSLATION_UNIT, never NULL on success */
} WGSLAst;

/**
 * Parse the token stream into an AST.
 *
 * @param tokens  Output of `wgsl_tokenize` (already template-discovered).
 * @param source  Source buffer (for diagnostic line/column translation).
 * @param arena   Arena that will own every AST node + child array.
 * @param diag    Bag to receive error diagnostics.
 * @param out     Populated with `root` on success.
 *
 * @return 1 if no error-severity diagnostics emitted; 0 otherwise.
 *         Always sets `out->root` to a valid (possibly partial) AST so
 *         downstream tooling does not need NULL checks.
 */
int wgsl_parse(
    const WGSLLexResult *tokens,
    const WGSLSource    *source,
    WGSLArena           *arena,
    WGSLDiagBag         *diag,
    WGSLAst             *out);

#endif /* WGSL_INTERNAL_PARSER_H */
