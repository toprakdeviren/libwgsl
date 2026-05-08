/**
 * @file span.h — Source range (byte offset + byte length).
 *
 * Header-only.  Spans are tiny value types passed by-value everywhere.
 * They reference bytes in a `WGSLSource` but don't carry a pointer to
 * one — the source is implicit at the call site.
 *
 *     end()       — exclusive end byte (offset + length)
 *     contains()  — true if byte `at` is inside [offset, end)
 *     join()      — smallest span covering both arguments
 */
#ifndef WGSL_INTERNAL_SPAN_H
#define WGSL_INTERNAL_SPAN_H

#include <stdint.h>

typedef struct WGSLSpan {
    uint32_t offset;
    uint32_t length;
} WGSLSpan;

static inline WGSLSpan wgsl_span_make(uint32_t offset, uint32_t length) {
    WGSLSpan s = { offset, length };
    return s;
}

static inline uint32_t wgsl_span_end(WGSLSpan s) {
    return s.offset + s.length;
}

static inline int wgsl_span_contains(WGSLSpan s, uint32_t at) {
    return at >= s.offset && at < s.offset + s.length;
}

static inline WGSLSpan wgsl_span_join(WGSLSpan a, WGSLSpan b) {
    uint32_t lo = a.offset < b.offset ? a.offset : b.offset;
    uint32_t a_end = a.offset + a.length;
    uint32_t b_end = b.offset + b.length;
    uint32_t hi = a_end > b_end ? a_end : b_end;
    WGSLSpan out = { lo, hi - lo };
    return out;
}

static inline int wgsl_span_eq(WGSLSpan a, WGSLSpan b) {
    return a.offset == b.offset && a.length == b.length;
}

#endif /* WGSL_INTERNAL_SPAN_H */
