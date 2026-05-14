/**
 * @file diag.c — Diagnostic accumulator.  See internal/diag.h.
 */
#include "internal/diag.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_DIAG_INITIAL_CAPACITY 8

void wgsl_diag_init(WGSLDiagBag *b) {
    b->items        = NULL;
    b->count        = 0;
    b->capacity     = 0;
    b->error_count  = 0;
    b->filter_count = 0;
    b->scope_depth  = 0;
    wgsl_arena_init(&b->msg_arena);
}

void wgsl_diag_destroy(WGSLDiagBag *b) {
    free(b->items);
    b->items        = NULL;
    b->count        = 0;
    b->capacity     = 0;
    b->error_count  = 0;
    b->filter_count = 0;
    b->scope_depth  = 0;
    wgsl_arena_destroy(&b->msg_arena);
}

static int wgsl_diag_grow(WGSLDiagBag *b) {
    if (b->capacity > SIZE_MAX / 2) return 0;
    size_t new_cap = b->capacity == 0 ? WGSL_DIAG_INITIAL_CAPACITY : b->capacity * 2;
    if (new_cap > SIZE_MAX / sizeof *b->items) return 0;
    WGSLDiagnostic *grown = (WGSLDiagnostic *)realloc(
        b->items, new_cap * sizeof *grown);
    if (!grown) return 0;
    b->items    = grown;
    b->capacity = new_cap;
    return 1;
}

static const char *wgsl_diag_intern(WGSLDiagBag *b, const char *src, size_t len) {
    /* +1 for terminating NUL */
    if (len == SIZE_MAX) return NULL;
    char *dst = (char *)wgsl_arena_alloc_aligned(&b->msg_arena, len + 1, 1);
    if (!dst) return NULL;
    if (len) memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

int wgsl_diag_emit_at(
    WGSLDiagBag *b,
    const WGSLSource *src,
    WGSLDiagSeverity severity,
    uint32_t span_offset,
    uint32_t span_length,
    const char *rule,
    const char *fmt, ...)
{
    /* §2.3 / §4.2: apply diagnostic filters.  Walk from the most
     * recently added filter (innermost scope) to the first, so
     * per-function @diagnostic overrides global diagnostic().
     * When a filter matches:
     *   - severity 0 ("off"): discard the diagnostic entirely.
     *   - otherwise: replace severity with the filter's value. */
    if (rule && *rule) {
        for (int i = b->filter_count - 1; i >= 0; i--) {
            if (b->filters[i].rule &&
                strcmp(b->filters[i].rule, rule) == 0)
            {
                int filt_sev = (int)b->filters[i].severity;
                if (filt_sev == 0) {
                    /* "off" — suppress entirely. */
                    return 1;
                }
                severity = (WGSLDiagSeverity)filt_sev;
                break;  /* innermost match wins */
            }
        }
    }

    if (b->count == b->capacity) {
        if (!wgsl_diag_grow(b)) return 0;
    }

    /* Render the message into the arena.  Two-pass: snprintf for size,
     * then alloc + render. */
    char stack_buf[256];
    char *msg_buf = stack_buf;
    int  needed = 0;

    if (fmt && *fmt) {
        va_list ap;
        va_start(ap, fmt);
        needed = vsnprintf(stack_buf, sizeof stack_buf, fmt, ap);
        va_end(ap);

        if (needed < 0) needed = 0;

        if ((size_t)needed >= sizeof stack_buf) {
            /* Need a bigger buffer.  Use a heap scratch then intern. */
            msg_buf = (char *)malloc((size_t)needed + 1);
            if (!msg_buf) return 0;
            va_start(ap, fmt);
            int written = vsnprintf(msg_buf, (size_t)needed + 1, fmt, ap);
            va_end(ap);
            if (written < 0) { free(msg_buf); return 0; }
        }
    }

    const char *msg_dst = wgsl_diag_intern(b, msg_buf, (size_t)needed);
    if (msg_buf != stack_buf) free(msg_buf);
    if (!msg_dst) return 0;

    /* Rule string — intern (or empty literal "" via arena). */
    const char *rule_dst;
    if (rule && *rule) {
        rule_dst = wgsl_diag_intern(b, rule, strlen(rule));
        if (!rule_dst) return 0;
    } else {
        rule_dst = wgsl_diag_intern(b, "", 0);
        if (!rule_dst) return 0;
    }

    /* Translate offset → 1-based (line, col). */
    uint32_t line = 1, col = 1, end_line = 1, end_col = 1;
    if (src) {
        uint32_t src_len = src->length > UINT32_MAX
            ? UINT32_MAX
            : (uint32_t)src->length;
        uint32_t span_start = span_offset > src_len ? src_len : span_offset;
        uint32_t span_end = span_offset > UINT32_MAX - span_length
            ? UINT32_MAX
            : span_offset + span_length;
        if (span_end > src_len) span_end = src_len;
        wgsl_source_offset_to_line_col(src, span_start, &line, &col);
        wgsl_source_offset_to_line_col(src, span_end, &end_line, &end_col);
    }

    WGSLDiagnostic *d = &b->items[b->count++];
    d->severity   = severity;
    d->line       = line;
    d->column     = col;
    d->end_line   = end_line;
    d->end_column = end_col;
    d->message    = msg_dst;
    d->rule       = rule_dst;

    if (severity == WGSL_DIAG_ERROR) b->error_count += 1;
    return 1;
}

size_t wgsl_diag_count(const WGSLDiagBag *b) {
    return b ? b->count : 0;
}

const WGSLDiagnostic *wgsl_diag_at(const WGSLDiagBag *b, size_t idx) {
    if (!b || idx >= b->count) return NULL;
    return &b->items[idx];
}

int wgsl_diag_has_error(const WGSLDiagBag *b) {
    return b ? (b->error_count > 0) : 0;
}

void wgsl_diag_set_filter(
    WGSLDiagBag *b, const char *rule, size_t rule_len,
    int severity)
{
    if (!b || !rule || rule_len == 0) return;
    if (b->filter_count >= WGSL_DIAG_FILTER_MAX) return;

    const char *interned = wgsl_diag_intern(b, rule, rule_len);
    if (!interned) return;

    b->filters[b->filter_count].rule     = interned;
    b->filters[b->filter_count].severity = (WGSLDiagSeverity)severity;
    b->filter_count++;
}

void wgsl_diag_push_scope(WGSLDiagBag *b) {
    if (!b) return;
    if (b->scope_depth >= WGSL_DIAG_SCOPE_MAX) return;
    b->scope_stack[b->scope_depth++] = b->filter_count;
}

void wgsl_diag_pop_scope(WGSLDiagBag *b) {
    if (!b) return;
    if (b->scope_depth <= 0) return;
    b->filter_count = b->scope_stack[--b->scope_depth];
}
