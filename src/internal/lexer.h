/**
 * @file lexer.h — WGSL tokenizer.
 *
 * One pass over the source bytes produces a flat `WGSLToken[]` plus
 * a final `WGSL_TOK_EOF` sentinel.  Blankspace and comment tokens are
 * emitted (per editor round-trip needs); the parser will skip them.
 *
 * Errors (invalid UTF-8, unterminated block comment, leading-zero
 * non-zero integer, forbidden `__` ident prefix, unknown character)
 * surface as diagnostics on the bag and the function returns 0.
 *
 * `wgsl_tokenize` also applies WGSL §3.9 template-list discovery before
 * returning the final token stream.
 */
#ifndef WGSL_INTERNAL_LEXER_H
#define WGSL_INTERNAL_LEXER_H

#include <stddef.h>

#include "internal/arena.h"
#include "internal/diag.h"
#include "internal/source.h"
#include "internal/token.h"

typedef struct WGSLLexResult {
    WGSLToken *tokens;   /* arena-allocated, contiguous */
    size_t     count;    /* includes the trailing WGSL_TOK_EOF */
} WGSLLexResult;

/**
 * Lex `source` into a token stream stored in `arena`.  Runs the full
 * tokenizer **and** §3.9 template-list discovery — `<` / `>` pairs that
 * delimit template lists are returned as `WGSL_TOK_TEMPLATE_START` /
 * `WGSL_TOK_TEMPLATE_END`; comparison and shift operators are
 * untouched.  Splits `>>`, `>=`, `>>=` when their leading `>` closes a
 * pending template (the trailing tail is re-emitted as a fresh token).
 *
 * @param source  Pre-initialized source buffer.
 * @param arena   Arena that will own the final token array.
 * @param diag    Bag to receive any error diagnostics.
 * @param out     On success, populated with `tokens` + `count`.
 *
 * @return 1 if no error-severity diagnostics were emitted; 0 otherwise.
 *         Returns 0 on out-of-memory as well (no tokens written).
 */
int wgsl_tokenize(
    const WGSLSource *source,
    WGSLArena        *arena,
    WGSLDiagBag      *diag,
    WGSLLexResult    *out);

/**
 * Apply the WGSL §3.9 template-list discovery algorithm to a
 * pre-tokenized stream.  Mutates `result` in place semantically — the
 * pointer in `result->tokens` is replaced with a fresh arena-allocated
 * array (the original is wasted but freed on arena destroy).
 *
 * Exposed mostly for the test suite.  Production code should call
 * `wgsl_tokenize` which already runs this internally.
 *
 * `diag`/`source` may be NULL (e.g. from tests that don't care about
 * diagnostics); when both are non-NULL an error is emitted if template
 * nesting exceeds the internal stack limit instead of silently dropping
 * the candidate `<`.
 *
 * @return 1 on success, 0 on out-of-memory.  A template-nesting-too-deep
 *         condition is reported via `diag` (not the return value); the
 *         higher-level check then fails on the error-severity diagnostic.
 */
int wgsl_discover_templates(
    WGSLLexResult    *result,
    WGSLArena        *arena,
    WGSLDiagBag      *diag,
    const WGSLSource *source);

#endif /* WGSL_INTERNAL_LEXER_H */
