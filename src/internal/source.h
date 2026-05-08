/**
 * @file source.h — Source buffer + precomputed line index.
 *
 * One `WGSLSource` represents the bytes of one WGSL module.  The
 * struct **borrows** the byte range; the caller is responsible for
 * keeping the bytes alive until `wgsl_source_destroy`.
 *
 * The line index is built once at `init` time so that diagnostic
 * span → line/column conversion is O(log line_count).
 *
 * Line break definition follows W3C WGSL §3.2:
 *   LF (U+000A), VT (U+000B), FF (U+000C),
 *   CR (U+000D) alone or as part of CR+LF,
 *   NEL (U+0085), LS (U+2028), PS (U+2029).
 *
 * Column is reported in **bytes from line start** in v1 (LSP's
 * default `positionEncodingKind` of "utf-16" can be added in v1.x —
 * tracked by `docs/PLAN.md`).
 */
#ifndef WGSL_INTERNAL_SOURCE_H
#define WGSL_INTERNAL_SOURCE_H

#include <stddef.h>
#include <stdint.h>

typedef struct WGSLSource {
    const char *bytes;       /* borrowed; not owned */
    size_t      length;
    uint32_t   *line_starts; /* line N (1-based) starts at line_starts[N-1] */
    uint32_t    line_count;  /* always >= 1 (empty source = one empty line) */
} WGSLSource;

/**
 * Initialize.  Walks `bytes` once to populate `line_starts`.
 * On success returns 1; on OOM returns 0 and leaves `*s` zeroed.
 */
int wgsl_source_init(WGSLSource *s, const char *bytes, size_t length);

/** Free internal allocations.  Bytes are not freed (we don't own them). */
void wgsl_source_destroy(WGSLSource *s);

/**
 * Convert a byte offset into 1-based (line, column).
 *
 * Offset == length is allowed and maps to the position one past the
 * last byte (useful for end-of-file diagnostics).  Offset > length
 * returns 0.
 *
 * Line is 1-based.  Column is 1-based byte distance from line start.
 */
int wgsl_source_offset_to_line_col(
    const WGSLSource *s, uint32_t offset,
    uint32_t *out_line, uint32_t *out_column);

/** Number of lines (>= 1). */
uint32_t wgsl_source_line_count(const WGSLSource *s);

/**
 * Pointer + length of line `line` (1-based), excluding the line break.
 * Returns 1 on success, 0 if `line` is out of range.
 */
int wgsl_source_line_text(
    const WGSLSource *s, uint32_t line,
    const char **out_ptr, size_t *out_length);

#endif /* WGSL_INTERNAL_SOURCE_H */
