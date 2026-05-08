/**
 * @file utf8.c — UTF-8 step decoder + Unicode predicates.  See internal/utf8.h.
 */
#include "internal/utf8.h"

#include <stdbool.h>

#include "core.h"           /* vendor: decoder_init */
#include "security.h"       /* vendor: decoder_is_identifier_start/continue */

void wgsl_utf8_init(void) {
    decoder_init();
}

/* WGSL §3.2 — Pattern_White_Space.  Exactly:
 *   U+0009 U+000A U+000B U+000C U+000D U+0020
 *   U+0085 U+200E U+200F U+2028 U+2029
 */
int wgsl_utf8_is_blankspace(uint32_t cp) {
    /* Cluster the ASCII members first — fast common path. */
    if (cp <= 0x20u) {
        return cp == 0x0009u || cp == 0x000Au || cp == 0x000Bu ||
               cp == 0x000Cu || cp == 0x000Du || cp == 0x0020u;
    }
    return cp == 0x0085u || cp == 0x200Eu || cp == 0x200Fu ||
           cp == 0x2028u || cp == 0x2029u;
}

int wgsl_utf8_is_ident_start(uint32_t cp) {
    /* ASCII fast path — `_` and [A-Za-z]. */
    if (cp < 0x80u) {
        return cp == '_' || (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    }
    return decoder_is_identifier_start(cp) ? 1 : 0;
}

int wgsl_utf8_is_ident_continue(uint32_t cp) {
    if (cp < 0x80u) {
        return cp == '_' ||
               (cp >= 'A' && cp <= 'Z') ||
               (cp >= 'a' && cp <= 'z') ||
               (cp >= '0' && cp <= '9');
    }
    return decoder_is_identifier_continue(cp) ? 1 : 0;
}

WGSLUtf8Decode wgsl_utf8_step(const uint8_t *bytes, size_t len) {
    WGSLUtf8Decode r = { 0u, 0 };
    if (len == 0) return r;

    uint8_t b0 = bytes[0];

    /* — 1-byte ASCII (most common) — */
    if (b0 < 0x80u) {
        r.code_point = b0;
        r.consumed   = 1;
        return r;
    }

    /* Determine sequence length and lead-byte payload. */
    int      need;     /* continuation bytes expected (1, 2, or 3) */
    uint32_t cp;

    if ((b0 & 0xE0u) == 0xC0u) {        /* 110xxxxx */
        need = 1;
        cp   = (uint32_t)(b0 & 0x1Fu);
    } else if ((b0 & 0xF0u) == 0xE0u) { /* 1110xxxx */
        need = 2;
        cp   = (uint32_t)(b0 & 0x0Fu);
    } else if ((b0 & 0xF8u) == 0xF0u) { /* 11110xxx */
        need = 3;
        cp   = (uint32_t)(b0 & 0x07u);
    } else {
        /* Lead byte starting with 10xxxxxx (orphan continuation) or
         * with five+ leading 1s — invalid. */
        return r;
    }

    if (len < (size_t)(1 + need)) return r;  /* truncated */

    for (int i = 0; i < need; i++) {
        uint8_t b = bytes[1 + i];
        if ((b & 0xC0u) != 0x80u) return r;  /* bad continuation */
        cp = (cp << 6) | (uint32_t)(b & 0x3Fu);
    }

    /* Reject overlong encodings: shortest-form rule. */
    if (need == 1 && cp < 0x80u)    return r;
    if (need == 2 && cp < 0x800u)   return r;
    if (need == 3 && cp < 0x10000u) return r;

    /* Reject UTF-16 surrogate halves and out-of-range code points. */
    if (cp >= 0xD800u && cp <= 0xDFFFu) return r;
    if (cp > 0x10FFFFu)                  return r;

    r.code_point = cp;
    r.consumed   = 1 + need;
    return r;
}
