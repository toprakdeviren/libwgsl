/**
 * @file utf8.h — UTF-8 step decoder + Unicode predicates the lexer needs.
 *
 * The library carries Unicode 17.0.0 tables via `vendor/unicode/`.
 * `wgsl_utf8_init()` brings them up; subsequent calls are idempotent
 * (matches `decoder_init` semantics) and thread-safe.
 *
 * `wgsl_utf8_step` is a single-shot incremental decoder used by the
 * lexer.  Returns `consumed == 0` for end-of-input or invalid UTF-8.
 *
 * Identifier predicates (`is_ident_start`, `is_ident_continue`) match
 * WGSL §3.7 — Unicode UAX31 XID-equivalent — by delegating to the
 * vendor library.  Blankspace (§3.2) is a tight 11-code-point set,
 * hard-coded here to guarantee spec match independent of vendor's
 * broader whitespace definition.
 *
 * §3.7.1 Identifier Comparison (pinned CRD): two identifiers are the
 * same iff they are the same sequence of code points.  The spec Note
 * explicitly does **not** permit Unicode normalization for comparison,
 * so the resolver's raw `memcmp` on source UTF-8 bytes is correct.
 * Vendored `normalize.h` is available but unused for this purpose.
 */
#ifndef WGSL_INTERNAL_UTF8_H
#define WGSL_INTERNAL_UTF8_H

#include <stddef.h>
#include <stdint.h>

typedef struct WGSLUtf8Decode {
    uint32_t code_point;   /* valid only when consumed > 0 */
    int      consumed;     /* bytes consumed; 0 on error or end-of-input */
} WGSLUtf8Decode;

/** Bring up the Unicode tables.  Idempotent + thread-safe. */
void wgsl_utf8_init(void);

/**
 * Decode one code point starting at `bytes[0]`.
 *
 * Validates against:
 *   - truncated sequences,
 *   - bad continuation bytes,
 *   - overlong encodings,
 *   - UTF-16 surrogate halves (U+D800..U+DFFF),
 *   - code points above U+10FFFF.
 *
 * On any failure, returns `{ .consumed = 0 }`.
 */
WGSLUtf8Decode wgsl_utf8_step(const uint8_t *bytes, size_t len);

/** WGSL §3.2 Pattern_White_Space — exactly 11 code points. */
int wgsl_utf8_is_blankspace(uint32_t cp);

/** WGSL §3.7 — XID_Start equivalent (delegates to vendor). */
int wgsl_utf8_is_ident_start(uint32_t cp);

/** WGSL §3.7 — XID_Continue equivalent (delegates to vendor). */
int wgsl_utf8_is_ident_continue(uint32_t cp);

#endif /* WGSL_INTERNAL_UTF8_H */
