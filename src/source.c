/**
 * @file source.c — Line index for WGSL source buffers.  See source.h.
 */
#include "internal/source.h"

#include <stdlib.h>
#include <string.h>

/* W3C WGSL §3.2 line breaks.  Returns 1 if a break starts at index `i`,
 * with `*consumed` set to the byte length of the break (1, 2, or 3).
 * Returns 0 otherwise. */
static int line_break_at(const char *s, size_t i, size_t len, int *consumed) {
    unsigned char c = (unsigned char)s[i];

    /* Single-byte breaks: LF, VT, FF, CR (alone). */
    if (c == 0x0A || c == 0x0B || c == 0x0C) {
        *consumed = 1;
        return 1;
    }
    if (c == 0x0D) {
        if (i + 1 < len && (unsigned char)s[i + 1] == 0x0A) {
            *consumed = 2;          /* CRLF — combined break */
        } else {
            *consumed = 1;          /* lone CR */
        }
        return 1;
    }
    /* NEL = U+0085 = 0xC2 0x85 */
    if (c == 0xC2 && i + 1 < len && (unsigned char)s[i + 1] == 0x85) {
        *consumed = 2;
        return 1;
    }
    /* LS = U+2028 = 0xE2 0x80 0xA8;  PS = U+2029 = 0xE2 0x80 0xA9 */
    if (c == 0xE2 && i + 2 < len &&
        (unsigned char)s[i + 1] == 0x80 &&
        ((unsigned char)s[i + 2] == 0xA8 || (unsigned char)s[i + 2] == 0xA9))
    {
        *consumed = 3;
        return 1;
    }
    return 0;
}

int wgsl_source_init(WGSLSource *s, const char *bytes, size_t length) {
    memset(s, 0, sizeof *s);

    s->bytes  = bytes;
    s->length = length;

    /* Initial capacity: one entry per ~16 bytes, clamped. */
    size_t cap = length / 16 + 16;
    if (cap < 16)        cap = 16;
    if (cap > 1u << 20)  cap = 1u << 20;

    s->line_starts = (uint32_t *)malloc(cap * sizeof *s->line_starts);
    if (!s->line_starts) {
        memset(s, 0, sizeof *s);
        return 0;
    }

    /* Line 1 always starts at offset 0. */
    s->line_starts[0] = 0;
    s->line_count     = 1;

    if (bytes == NULL) {
        return 1;
    }

    size_t i = 0;
    while (i < length) {
        int consumed = 0;
        if (line_break_at(bytes, i, length, &consumed)) {
            uint32_t next = (uint32_t)(i + (size_t)consumed);

            if (s->line_count + 1 > cap) {
                size_t new_cap = cap * 2;
                uint32_t *grown = (uint32_t *)realloc(
                    s->line_starts, new_cap * sizeof *s->line_starts);
                if (!grown) {
                    free(s->line_starts);
                    memset(s, 0, sizeof *s);
                    return 0;
                }
                s->line_starts = grown;
                cap            = new_cap;
            }
            s->line_starts[s->line_count++] = next;
            i = (size_t)next;
        } else {
            i += 1;
        }
    }

    /* Trim allocation to actual size for memory hygiene; non-fatal on OOM. */
    if (s->line_count < cap) {
        uint32_t *trimmed = (uint32_t *)realloc(
            s->line_starts, s->line_count * sizeof *s->line_starts);
        if (trimmed) s->line_starts = trimmed;
    }
    return 1;
}

void wgsl_source_destroy(WGSLSource *s) {
    free(s->line_starts);
    memset(s, 0, sizeof *s);
}

int wgsl_source_offset_to_line_col(
    const WGSLSource *s, uint32_t offset,
    uint32_t *out_line, uint32_t *out_column)
{
    if (offset > s->length) return 0;

    /* Binary search for the largest line_starts[k] <= offset. */
    uint32_t lo = 0, hi = s->line_count;
    while (lo + 1 < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (s->line_starts[mid] <= offset) lo = mid;
        else                                hi = mid;
    }
    if (out_line)   *out_line   = lo + 1;                       /* 1-based */
    if (out_column) *out_column = (offset - s->line_starts[lo]) + 1;
    return 1;
}

uint32_t wgsl_source_line_count(const WGSLSource *s) {
    return s ? s->line_count : 0;
}

int wgsl_source_line_text(
    const WGSLSource *s, uint32_t line,
    const char **out_ptr, size_t *out_length)
{
    if (line == 0 || line > s->line_count) return 0;

    uint32_t start = s->line_starts[line - 1];
    uint32_t end   = (line == s->line_count)
                       ? (uint32_t)s->length
                       : s->line_starts[line];

    /* Strip the trailing line break from the slice we hand back. */
    if (end > start) {
        if (end >= 2 + start &&
            (unsigned char)s->bytes[end - 2] == 0x0D &&
            (unsigned char)s->bytes[end - 1] == 0x0A)
        {
            end -= 2;                                    /* CRLF */
        } else if (end > start) {
            int consumed = 0;
            uint32_t probe = end - 1;
            /* Try 1-byte break (LF/VT/FF/CR). */
            if (line_break_at(s->bytes, probe, s->length, &consumed)
                && consumed == 1) {
                end -= 1;
            } else if (end >= 2 + start &&
                       line_break_at(s->bytes, end - 2, s->length, &consumed)
                       && consumed == 2) {
                end -= 2;                                /* NEL */
            } else if (end >= 3 + start &&
                       line_break_at(s->bytes, end - 3, s->length, &consumed)
                       && consumed == 3) {
                end -= 3;                                /* LS / PS */
            }
        }
    }
    *out_ptr    = s->bytes + start;
    *out_length = (size_t)(end - start);
    return 1;
}
