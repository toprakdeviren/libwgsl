/**
 * @file diag.h — Diagnostic accumulator (the "bag" pass results pour into).
 *
 * The bag owns:
 *   - a growable array of public `WGSLDiagnostic` records, and
 *   - a private arena for message and rule string storage so that the
 *     `const char *` pointers inside each record stay stable through the
 *     bag's lifetime.
 *
 * Every pass calls `wgsl_diag_emit_at` with a source span (byte offset +
 * length); the emit function looks up (line, column) via the provided
 * `WGSLSource` and the position becomes part of the public record.
 *
 * Filterable rule names (per WGSL §2.3.2 — `derivative_uniformity` etc.)
 * are passed in `rule`; pass NULL or "" for unconditional diagnostics.
 *
 * `wgsl_diag_destroy` invalidates every pointer the bag handed out.
 */
#ifndef WGSL_INTERNAL_DIAG_H
#define WGSL_INTERNAL_DIAG_H

#include <stddef.h>
#include <stdint.h>

#include "wgsl.h"
#include "internal/arena.h"
#include "internal/source.h"

/* Maximum number of active diagnostic filters (global + per-scope).
 * WGSL currently defines 2 filterable rule names; 8 is generous. */
#define WGSL_DIAG_FILTER_MAX 8
#define WGSL_DIAG_SCOPE_MAX 16

typedef struct {
    const char       *rule;    /* interned in msg_arena */
    WGSLDiagSeverity  severity;
} WGSLDiagFilter;

typedef struct WGSLDiagBag {
    WGSLDiagnostic *items;
    size_t          count;
    size_t          capacity;
    WGSLArena       msg_arena;     /* owns message + rule string bytes */
    int             error_count;   /* WGSL_DIAG_ERROR-severity items   */

    /* Diagnostic filters (§2.3 / §4.2).  Filters are stacked:
     * global filters are pushed first, then per-scope (fn) filters.
     * `scope_base` marks where per-scope filters start so they can
     * be popped cleanly. */
    WGSLDiagFilter   filters[WGSL_DIAG_FILTER_MAX];
    int              filter_count;
    int              scope_stack[WGSL_DIAG_SCOPE_MAX];
    int              scope_depth;
} WGSLDiagBag;

void wgsl_diag_init(WGSLDiagBag *b);
void wgsl_diag_destroy(WGSLDiagBag *b);

/**
 * Emit a diagnostic.  `fmt` is printf-style and may be NULL or "".
 * Returns 1 on success, 0 on allocation failure.
 *
 * @param src           Source for offset → line/column translation.
 * @param severity      WGSL_DIAG_ERROR / WARNING / INFO / HINT.
 * @param span_offset   Byte offset into `src`.
 * @param span_length   Byte length.
 * @param rule          Filterable rule name, or NULL / "" if unconditional.
 * @param fmt, ...      printf-style message; rendered into the bag's arena.
 */
int wgsl_diag_emit_at(
    WGSLDiagBag *b,
    const WGSLSource *src,
    WGSLDiagSeverity severity,
    uint32_t span_offset,
    uint32_t span_length,
    const char *rule,
    const char *fmt, ...);

size_t                wgsl_diag_count    (const WGSLDiagBag *b);
const WGSLDiagnostic *wgsl_diag_at       (const WGSLDiagBag *b, size_t idx);
int                   wgsl_diag_has_error(const WGSLDiagBag *b);

/**
 * Register a diagnostic filter.  `rule` is copied into the bag's arena.
 * Severity `WGSL_DIAG_HINT` means "off" (suppress entirely).
 * Filters are matched by rule-string equality during `wgsl_diag_emit_at`;
 * when matched, the diagnostic's severity is replaced by `severity`.
 * If severity is "off" (mapped to 0 by the caller), the diagnostic is
 * discarded entirely.
 */
void wgsl_diag_set_filter(
    WGSLDiagBag *b, const char *rule, size_t rule_len,
    int severity);   /* 0 = off, else WGSLDiagSeverity */

/** Push a scope boundary (e.g. entering a function with @diagnostic). */
void wgsl_diag_push_scope(WGSLDiagBag *b);

/** Pop the current scope, removing any filters added since the last push. */
void wgsl_diag_pop_scope(WGSLDiagBag *b);

#endif /* WGSL_INTERNAL_DIAG_H */
