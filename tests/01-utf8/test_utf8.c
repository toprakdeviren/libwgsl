/**
 * UTF-8 step decoder and Unicode predicate tests.
 *
 * Covers:
 *   - 1/2/3/4 byte valid sequences
 *   - end-of-input and truncation
 *   - bad continuation bytes
 *   - overlong encodings (2-, 3-, 4-byte forms)
 *   - UTF-16 surrogates rejected (0xED 0xA0 0x80 = U+D800)
 *   - out-of-range code points (>U+10FFFF)
 *   - Pattern_White_Space exact set (§3.2)
 *   - identifier_start / identifier_continue ASCII + non-ASCII
 *   - decoder_init via wgsl_utf8_init is idempotent
 */
#include "internal/utf8.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static void check_step(const uint8_t *b, size_t len,
                       uint32_t want_cp, int want_consumed,
                       const char *tag) {
    WGSLUtf8Decode r = wgsl_utf8_step(b, len);
    if (r.code_point != want_cp || r.consumed != want_consumed) {
        fprintf(stderr,
                "FAIL  %s: got cp=U+%04X consumed=%d, want U+%04X consumed=%d\n",
                tag, r.code_point, r.consumed, want_cp, want_consumed);
        fail += 1;
    }
}

int main(void) {
    wgsl_utf8_init();
    wgsl_utf8_init();   /* idempotent */

    /* — 1-byte ASCII — */
    check_step((const uint8_t *)"A", 1, 'A',    1, "ASCII A");
    check_step((const uint8_t *)"\x00", 1, 0u,  1, "NUL");
    check_step((const uint8_t *)"\x7F", 1, 0x7Fu, 1, "DEL");

    /* — 2-byte: U+00E9 (é) = C3 A9 — */
    check_step((const uint8_t *)"\xC3\xA9", 2, 0x00E9u, 2, "U+00E9 é");

    /* — 3-byte: U+2028 (LS) = E2 80 A8 — */
    check_step((const uint8_t *)"\xE2\x80\xA8", 3, 0x2028u, 3, "U+2028 LS");

    /* — 4-byte: U+1F600 (😀) = F0 9F 98 80 — */
    check_step((const uint8_t *)"\xF0\x9F\x98\x80", 4, 0x1F600u, 4, "U+1F600 😀");

    /* — End of input — */
    check_step((const uint8_t *)"", 0, 0u, 0, "EOF");

    /* — Truncated 2-byte — */
    check_step((const uint8_t *)"\xC3", 1, 0u, 0, "truncated 2-byte");
    /* — Truncated 3-byte — */
    check_step((const uint8_t *)"\xE2\x80", 2, 0u, 0, "truncated 3-byte");
    /* — Truncated 4-byte — */
    check_step((const uint8_t *)"\xF0\x9F\x98", 3, 0u, 0, "truncated 4-byte");

    /* — Bad continuation: lead C3 followed by ASCII — */
    check_step((const uint8_t *)"\xC3""A", 2, 0u, 0, "bad continuation");
    /* — Orphan continuation byte as lead — */
    check_step((const uint8_t *)"\x80""x", 2, 0u, 0, "orphan continuation");

    /* — Overlong: 2-byte form encoding U+0000 — C0 80 — */
    check_step((const uint8_t *)"\xC0\x80", 2, 0u, 0, "overlong 2-byte for U+0000");
    /* — Overlong: 3-byte form encoding U+0024 — E0 80 A4 — */
    check_step((const uint8_t *)"\xE0\x80\xA4", 3, 0u, 0, "overlong 3-byte");
    /* — Overlong: 4-byte form encoding U+10 — F0 80 80 90 — */
    check_step((const uint8_t *)"\xF0\x80\x80\x90", 4, 0u, 0, "overlong 4-byte");

    /* — Surrogate U+D800 = ED A0 80 — must reject — */
    check_step((const uint8_t *)"\xED\xA0\x80", 3, 0u, 0, "surrogate U+D800");
    /* — Surrogate U+DFFF = ED BF BF — */
    check_step((const uint8_t *)"\xED\xBF\xBF", 3, 0u, 0, "surrogate U+DFFF");

    /* — Out of range: encodes U+110000 = F4 90 80 80 — must reject — */
    check_step((const uint8_t *)"\xF4\x90\x80\x80", 4, 0u, 0, "U+110000 out of range");
    /* — Boundary: U+10FFFF = F4 8F BF BF — must accept — */
    check_step((const uint8_t *)"\xF4\x8F\xBF\xBF", 4, 0x10FFFFu, 4, "U+10FFFF boundary");

    /* — 5-byte form (FB ...) is illegal for any code point — */
    check_step((const uint8_t *)"\xFB\x80\x80\x80\x80", 5, 0u, 0, "5-byte FB lead");

    /* — Pattern_White_Space exact set — */
    CHECK( wgsl_utf8_is_blankspace(0x0009), "TAB is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x0020), "SP is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x000A), "LF is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x000D), "CR is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x0085), "NEL is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x200E), "LRM is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x200F), "RLM is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x2028), "LS is blankspace");
    CHECK( wgsl_utf8_is_blankspace(0x2029), "PS is blankspace");
    /* Negatives — common things that are NOT in Pattern_White_Space */
    CHECK(!wgsl_utf8_is_blankspace('A'),     "letter not blankspace");
    CHECK(!wgsl_utf8_is_blankspace(0x00A0),  "NBSP not Pattern_White_Space");
    CHECK(!wgsl_utf8_is_blankspace(0x3000),  "ideographic space not in set");

    /* — Identifier predicates ASCII fast path — */
    CHECK( wgsl_utf8_is_ident_start('a'),    "a is ident start");
    CHECK( wgsl_utf8_is_ident_start('Z'),    "Z is ident start");
    CHECK( wgsl_utf8_is_ident_start('_'),    "_ is ident start");
    CHECK(!wgsl_utf8_is_ident_start('0'),    "digit not ident start");
    CHECK(!wgsl_utf8_is_ident_start(' '),    "space not ident start");
    CHECK( wgsl_utf8_is_ident_continue('0'), "digit is ident continue");
    CHECK( wgsl_utf8_is_ident_continue('a'), "letter is ident continue");
    CHECK( wgsl_utf8_is_ident_continue('_'), "_ is ident continue");
    CHECK(!wgsl_utf8_is_ident_continue(' '), "space not ident continue");

    /* — Identifier predicates non-ASCII (delegates to vendor Unicode 17.0) — */
    CHECK( wgsl_utf8_is_ident_start(0x00E9),  "é (U+00E9) is ident start");
    CHECK( wgsl_utf8_is_ident_continue(0x00E9), "é is ident continue");
    CHECK( wgsl_utf8_is_ident_start(0x05D0),  "Hebrew alef is ident start");
    CHECK( wgsl_utf8_is_ident_start(0x4E2D),  "中 is ident start");
    CHECK(!wgsl_utf8_is_ident_start(0x0660),  "Arabic-Indic 0 is not ident start"); /* digit */
    CHECK( wgsl_utf8_is_ident_continue(0x0660), "Arabic-Indic 0 is ident continue");

    if (fail == 0) {
        printf("PASS  test_utf8\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_utf8  %d check(s) failed\n", fail);
    return 1;
}
