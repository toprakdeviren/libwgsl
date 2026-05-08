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
    b->items       = NULL;
    b->count       = 0;
    b->capacity    = 0;
    b->error_count = 0;
    wgsl_arena_init(&b->msg_arena);
}

void wgsl_diag_destroy(WGSLDiagBag *b) {
    free(b->items);
    b->items       = NULL;
    b->count       = 0;
    b->capacity    = 0;
    b->error_count = 0;
    wgsl_arena_destroy(&b->msg_arena);
}

static int wgsl_diag_grow(WGSLDiagBag *b) {
    size_t new_cap = b->capacity == 0 ? WGSL_DIAG_INITIAL_CAPACITY : b->capacity * 2;
    WGSLDiagnostic *grown = (WGSLDiagnostic *)realloc(
        b->items, new_cap * sizeof *grown);
    if (!grown) return 0;
    b->items    = grown;
    b->capacity = new_cap;
    return 1;
}

static const char *wgsl_diag_intern(WGSLDiagBag *b, const char *src, size_t len) {
    /* +1 for terminating NUL */
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
        wgsl_source_offset_to_line_col(src, span_offset, &line, &col);
        wgsl_source_offset_to_line_col(src, span_offset + span_length, &end_line, &end_col);
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
