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
 * Round B (next iteration) adds template-list discovery (§3.9) on top.
 * For Round A, `<`/`>` are tokenized as `LESS`/`GREATER` etc.
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
 * @return 1 on success, 0 on out-of-memory.
 */
int wgsl_discover_templates(
    WGSLLexResult *result,
    WGSLArena     *arena);

#endif /* WGSL_INTERNAL_LEXER_H */
