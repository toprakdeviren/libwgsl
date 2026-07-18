/**
 * `WGSLSource` line-index tests.
 *
 * Covers:
 *   - empty source
 *   - source without trailing newline
 *   - LF, CR, CRLF, VT, FF
 *   - NEL (0xC2 0x85)
 *   - LS (0xE2 0x80 0xA8)  and  PS (0xE2 0x80 0xA9)
 *   - offset → (line, col) round trip with line_text
 *   - end-of-file offset (offset == length)
 *   - non-CRLF "lone CR" not joined to LF when separated
 */
#include "internal/source.h"

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

static void check_pos(const WGSLSource *s, uint32_t off,
                      uint32_t want_line, uint32_t want_col,
                      const char *tag) {
    uint32_t line = 0, col = 0;
    int ok = wgsl_source_offset_to_line_col(s, off, &line, &col);
    if (!ok || line != want_line || col != want_col) {
        fprintf(stderr,
                "FAIL  %s: offset %u → line %u col %u (got ok=%d line=%u col=%u)\n",
                tag, off, want_line, want_col, ok, line, col);
        fail += 1;
    }
}

static void check_line(const WGSLSource *s, uint32_t line,
                       const char *want, const char *tag) {
    const char *p = NULL; size_t n = 0;
    int ok = wgsl_source_line_text(s, line, &p, &n);
    size_t want_len = strlen(want);
    if (!ok || n != want_len || memcmp(p, want, n) != 0) {
        fprintf(stderr,
                "FAIL  %s: line %u expected \"%s\" (%zu); "
                "got ok=%d len=%zu \"%.*s\"\n",
                tag, line, want, want_len, ok, n, (int)n, p ? p : "");
        fail += 1;
    }
}

int main(void) {
    /* — empty source — */
    {
        WGSLSource s; (void)wgsl_source_init(&s, "", 0);
        CHECK(wgsl_source_line_count(&s) == 1, "empty: 1 line");
        check_pos(&s, 0, 1, 1, "empty@0");
        check_line(&s, 1, "", "empty.line1");
        wgsl_source_destroy(&s);
    }

    /* — single line, no trailing newline — */
    {
        const char *src = "let x = 1;";
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 1, "no-trailing: 1 line");
        check_pos(&s, 0,           1, 1,                 "noNL@0");
        check_pos(&s, 4,           1, 5,                 "noNL@4");
        check_pos(&s, (uint32_t)strlen(src), 1, (uint32_t)strlen(src) + 1, "noNL@end");
        check_line(&s, 1, "let x = 1;", "noNL.line1");
        wgsl_source_destroy(&s);
    }

    /* — LF — */
    {
        const char *src = "abc\ndef\nghi";
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 3, "LF: 3 lines");
        check_pos(&s, 0, 1, 1, "lf@0");
        check_pos(&s, 3, 1, 4, "lf@\\n");        /* on the LF itself */
        check_pos(&s, 4, 2, 1, "lf:line2");
        check_pos(&s, 7, 2, 4, "lf@\\n2");
        check_pos(&s, 8, 3, 1, "lf:line3");
        check_line(&s, 1, "abc", "lf.line1");
        check_line(&s, 2, "def", "lf.line2");
        check_line(&s, 3, "ghi", "lf.line3");
        wgsl_source_destroy(&s);
    }

    /* — CR alone — */
    {
        const char *src = "abc\rdef";
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 2, "CR: 2 lines");
        check_pos(&s, 4, 2, 1, "CR:line2");
        check_line(&s, 1, "abc", "CR.line1");
        check_line(&s, 2, "def", "CR.line2");
        wgsl_source_destroy(&s);
    }

    /* — CRLF as one break — */
    {
        const char *src = "abc\r\ndef";
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 2, "CRLF: 2 lines");
        check_pos(&s, 5, 2, 1, "CRLF:line2");          /* skips both bytes */
        check_line(&s, 1, "abc", "CRLF.line1");
        check_line(&s, 2, "def", "CRLF.line2");
        wgsl_source_destroy(&s);
    }

    /* — CR followed by non-LF then LF must not be joined — */
    {
        const char *src = "a\rX\n";                    /* 4 bytes, 3 lines */
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 3, "CR+notLF+LF: 3 lines");
        check_line(&s, 1, "a",  "CRsep.line1");
        check_line(&s, 2, "X",  "CRsep.line2");
        check_line(&s, 3, "",   "CRsep.line3");
        wgsl_source_destroy(&s);
    }

    /* — VT and FF — */
    {
        const char *src = "a\vb\fc";
        WGSLSource s; wgsl_source_init(&s, src, strlen(src));
        CHECK(wgsl_source_line_count(&s) == 3, "VT/FF: 3 lines");
        wgsl_source_destroy(&s);
    }

    /* — NEL (U+0085) = 0xC2 0x85 — */
    {
        const char src[] = "a\xC2\x85" "b";            /* 4 bytes total */
        WGSLSource s; wgsl_source_init(&s, src, sizeof src - 1);
        CHECK(wgsl_source_line_count(&s) == 2, "NEL: 2 lines");
        check_pos(&s, 3, 2, 1, "NEL:line2");
        check_line(&s, 1, "a", "NEL.line1");
        check_line(&s, 2, "b", "NEL.line2");
        wgsl_source_destroy(&s);
    }

    /* — LS (U+2028) = 0xE2 0x80 0xA8 — */
    {
        const char src[] = "a\xE2\x80\xA8" "b";        /* 5 bytes total */
        WGSLSource s; wgsl_source_init(&s, src, sizeof src - 1);
        CHECK(wgsl_source_line_count(&s) == 2, "LS: 2 lines");
        check_pos(&s, 4, 2, 1, "LS:line2");
        check_line(&s, 1, "a", "LS.line1");
        check_line(&s, 2, "b", "LS.line2");
        wgsl_source_destroy(&s);
    }

    /* — PS (U+2029) = 0xE2 0x80 0xA9 — */
    {
        const char src[] = "a\xE2\x80\xA9" "b";
        WGSLSource s; wgsl_source_init(&s, src, sizeof src - 1);
        CHECK(wgsl_source_line_count(&s) == 2, "PS: 2 lines");
        check_line(&s, 2, "b", "PS.line2");
        wgsl_source_destroy(&s);
    }

    /* — Out-of-range offset is rejected — */
    {
        const char *src = "abc";
        WGSLSource s; wgsl_source_init(&s, src, 3);
        uint32_t l, c;
        int ok = wgsl_source_offset_to_line_col(&s, 999, &l, &c);
        CHECK(!ok, "out-of-range offset returns 0");
        wgsl_source_destroy(&s);
    }

    /* — Mixed: CRLF + LF + LS + EOF without break — */
    {
        const char src[] =
            "alpha\r\n"        /* 7 bytes */
            "beta\n"            /* 5 bytes — line 2 */
            "gamma\xE2\x80\xA8" /* 8 bytes — line 3 */
            "delta";            /* 5 bytes — line 4, no trailing break */
        size_t len = sizeof src - 1;
        WGSLSource s; wgsl_source_init(&s, src, len);
        CHECK(wgsl_source_line_count(&s) == 4, "mixed: 4 lines");
        check_line(&s, 1, "alpha", "mix.line1");
        check_line(&s, 2, "beta",  "mix.line2");
        check_line(&s, 3, "gamma", "mix.line3");
        check_line(&s, 4, "delta", "mix.line4");
        wgsl_source_destroy(&s);
    }

    if (fail == 0) {
        printf("PASS  test_source  (line breaks: LF/CR/CRLF/VT/FF/NEL/LS/PS)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_source  %d check(s) failed\n", fail);
    return 1;
}
