/**
 * @file lexer.c — WGSL tokenizer.  See internal/lexer.h.
 */
#include "internal/lexer.h"

#include "internal/utf8.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WGSL_LEX_INITIAL_CAPACITY 64

typedef struct {
    const WGSLSource *src;
    const uint8_t    *bytes;
    size_t            length;
    size_t            pos;

    WGSLArena        *arena;
    WGSLDiagBag      *diag;

    /* Heap-backed dynamic array during lex; copied into `arena` at end. */
    WGSLToken        *tokens;
    size_t            count;
    size_t            capacity;

    int               had_error;
} L;

/* — Helpers — */

static int is_dec_digit(uint8_t c) { return c >= '0' && c <= '9'; }
static int is_hex_digit(uint8_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
static int is_ascii_ident_start(uint8_t c) {
    return c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static int is_ascii_ident_continue(uint8_t c) {
    return is_ascii_ident_start(c) || is_dec_digit(c);
}
static int is_ascii_blankspace(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static int grow_tokens(L *l) {
    size_t new_cap = l->capacity ? l->capacity * 2 : WGSL_LEX_INITIAL_CAPACITY;
    WGSLToken *grown = (WGSLToken *)realloc(l->tokens, new_cap * sizeof *grown);
    if (!grown) return 0;
    l->tokens   = grown;
    l->capacity = new_cap;
    return 1;
}

static int push_tok(L *l, WGSLTokenKind kind,
                    uint32_t off, uint32_t len, uint32_t payload)
{
    if (l->count == l->capacity && !grow_tokens(l)) return 0;

    uint32_t line = 1, col = 1;
    wgsl_source_offset_to_line_col(l->src, off, &line, &col);

    WGSLToken *t = &l->tokens[l->count++];
    t->kind        = kind;
    t->span.offset = off;
    t->span.length = len;
    t->line        = line;
    t->column      = col;
    t->payload     = payload;
    return 1;
}

static void emit_error(L *l, uint32_t off, uint32_t len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[256];
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    wgsl_diag_emit_at(l->diag, l->src, WGSL_DIAG_ERROR, off, len, NULL, "%s", buf);
    l->had_error = 1;
}

/* — Sub-scanners — */

static int scan_blankspace(L *l) {
    size_t start = l->pos;
    while (l->pos < l->length) {
        uint8_t b = l->bytes[l->pos];
        if (b < 0x80u) {
            if (!is_ascii_blankspace(b)) break;
            l->pos++;
            continue;
        }
        WGSLUtf8Decode d = wgsl_utf8_step(l->bytes + l->pos, l->length - l->pos);
        if (!d.consumed || !wgsl_utf8_is_blankspace(d.code_point)) break;
        l->pos += (size_t)d.consumed;
    }
    return push_tok(l, WGSL_TOK_BLANKSPACE,
                    (uint32_t)start, (uint32_t)(l->pos - start), 0);
}

static int scan_line_comment(L *l) {
    size_t start = l->pos;
    l->pos += 2;  /* skip `//` */
    while (l->pos < l->length) {
        uint8_t b = l->bytes[l->pos];
        /* Stop at any line break.  Per §3.2: LF VT FF CR NEL LS PS. */
        if (b == 0x0Au || b == 0x0Bu || b == 0x0Cu || b == 0x0Du) break;
        if (b == 0xC2u && l->pos + 1 < l->length && l->bytes[l->pos + 1] == 0x85u) break;
        if (b == 0xE2u && l->pos + 2 < l->length &&
            l->bytes[l->pos + 1] == 0x80u &&
            (l->bytes[l->pos + 2] == 0xA8u || l->bytes[l->pos + 2] == 0xA9u)) break;
        l->pos += 1;
    }
    return push_tok(l, WGSL_TOK_LINE_COMMENT,
                    (uint32_t)start, (uint32_t)(l->pos - start), 0);
}

static int scan_block_comment(L *l) {
    size_t start = l->pos;
    l->pos += 2;  /* skip `/​*` */
    int depth = 1;
    while (l->pos < l->length && depth > 0) {
        if (l->pos + 1 < l->length) {
            if (l->bytes[l->pos] == '/' && l->bytes[l->pos + 1] == '*') {
                depth += 1;
                l->pos += 2;
                continue;
            }
            if (l->bytes[l->pos] == '*' && l->bytes[l->pos + 1] == '/') {
                depth -= 1;
                l->pos += 2;
                continue;
            }
        }
        /* Advance one code point; UTF-8-validate as we go. */
        if (l->bytes[l->pos] < 0x80u) {
            l->pos += 1;
        } else {
            WGSLUtf8Decode d = wgsl_utf8_step(l->bytes + l->pos, l->length - l->pos);
            if (!d.consumed) {
                emit_error(l, (uint32_t)l->pos, 1, "invalid UTF-8 in block comment");
                l->pos += 1;
                return 0;
            }
            l->pos += (size_t)d.consumed;
        }
    }
    if (depth > 0) {
        emit_error(l, (uint32_t)start, 2, "unterminated block comment");
        return 0;
    }
    return push_tok(l, WGSL_TOK_BLOCK_COMMENT,
                    (uint32_t)start, (uint32_t)(l->pos - start), 0);
}

static int scan_ident(L *l) {
    size_t start = l->pos;

    /* Consume the first code point — already validated as ident-start. */
    if (l->bytes[l->pos] < 0x80u) {
        l->pos += 1;
    } else {
        WGSLUtf8Decode d = wgsl_utf8_step(l->bytes + l->pos, l->length - l->pos);
        l->pos += (size_t)d.consumed;
    }
    /* Continue. */
    while (l->pos < l->length) {
        uint8_t b = l->bytes[l->pos];
        if (b < 0x80u) {
            if (!is_ascii_ident_continue(b)) break;
            l->pos += 1;
            continue;
        }
        WGSLUtf8Decode d = wgsl_utf8_step(l->bytes + l->pos, l->length - l->pos);
        if (!d.consumed || !wgsl_utf8_is_ident_continue(d.code_point)) break;
        l->pos += (size_t)d.consumed;
    }

    size_t length = l->pos - start;

    /* Lone `_` is the phony marker, not an identifier. */
    if (length == 1 && l->bytes[start] == '_') {
        return push_tok(l, WGSL_TOK_UNDERSCORE, (uint32_t)start, 1, 0);
    }

    /* §3.7: identifiers must not start with `__`. */
    if (length >= 2 && l->bytes[start] == '_' && l->bytes[start + 1] == '_') {
        emit_error(l, (uint32_t)start, (uint32_t)length,
                   "identifier must not start with '__' (WGSL §3.7)");
        /* Still emit a token so subsequent passes aren't disrupted. */
    }

    WGSLTokenKind kind = wgsl_token_classify_ident(
        (const char *)(l->bytes + start), length);
    return push_tok(l, kind, (uint32_t)start, (uint32_t)length, 0);
}

/* Numeric literal scanner.  Handles all four shapes:
 *   decimal_int    [0-9]+ ([iu])?
 *   decimal_float  [0-9]+ '.' [0-9]* ([eE][+-]?[0-9]+)? ([fh])?
 *                  | [0-9]* '.' [0-9]+ ([eE][+-]?[0-9]+)? ([fh])?
 *                  | [0-9]+ [eE][+-]?[0-9]+ ([fh])?
 *                  | [0-9]+ [fh]
 *   hex_int        0[xX] [0-9a-fA-F]+ ([iu])?
 *   hex_float      0[xX] [0-9a-fA-F]* '.' [0-9a-fA-F]+ ([pP][+-]?[0-9]+ ([fh])?)?
 *                  | 0[xX] [0-9a-fA-F]+ '.' [0-9a-fA-F]* ([pP][+-]?[0-9]+ ([fh])?)?
 *                  | 0[xX] [0-9a-fA-F]+ [pP][+-]?[0-9]+ ([fh])?
 *
 * Caller guarantees the first byte is digit or `.` followed by digit.
 */
static int scan_number(L *l) {
    size_t start = l->pos;
    int is_float = 0;
    int is_hex   = 0;
    uint32_t suffix = WGSL_NUM_ABSTRACT;

    /* Leading-dot decimal float, e.g. `.5`. */
    if (l->bytes[l->pos] == '.') {
        is_float = 1;
        l->pos += 1;
        while (l->pos < l->length && is_dec_digit(l->bytes[l->pos])) l->pos += 1;
        goto check_dec_exponent;
    }

    /* Hex prefix `0x` or `0X`. */
    if (l->bytes[l->pos] == '0' && l->pos + 1 < l->length &&
        (l->bytes[l->pos + 1] == 'x' || l->bytes[l->pos + 1] == 'X'))
    {
        is_hex = 1;
        l->pos += 2;
        size_t hex_int_start = l->pos;
        while (l->pos < l->length && is_hex_digit(l->bytes[l->pos])) l->pos += 1;
        size_t hex_int_len = l->pos - hex_int_start;

        size_t hex_frac_len = 0;
        int has_dot = 0, has_exp = 0;

        if (l->pos < l->length && l->bytes[l->pos] == '.') {
            has_dot = 1;
            l->pos += 1;
            size_t f0 = l->pos;
            while (l->pos < l->length && is_hex_digit(l->bytes[l->pos])) l->pos += 1;
            hex_frac_len = l->pos - f0;
        }
        if (l->pos < l->length && (l->bytes[l->pos] == 'p' || l->bytes[l->pos] == 'P')) {
            has_exp = 1;
            l->pos += 1;
            if (l->pos < l->length &&
                (l->bytes[l->pos] == '+' || l->bytes[l->pos] == '-')) l->pos += 1;
            size_t e0 = l->pos;
            while (l->pos < l->length && is_dec_digit(l->bytes[l->pos])) l->pos += 1;
            if (l->pos == e0) {
                emit_error(l, (uint32_t)start, (uint32_t)(l->pos - start),
                           "expected digits in hex float exponent");
                return 0;
            }
        }

        is_float = has_dot || has_exp;

        if (!is_float && hex_int_len == 0) {
            emit_error(l, (uint32_t)start, (uint32_t)(l->pos - start),
                       "hex integer must have at least one hex digit");
            return 0;
        }
        if (is_float && hex_int_len == 0 && hex_frac_len == 0) {
            emit_error(l, (uint32_t)start, (uint32_t)(l->pos - start),
                       "hex float must have at least one hex digit");
            return 0;
        }
        goto suffix_check;
    }

    /* Decimal integer or float. */
    {
        size_t int_start = l->pos;
        while (l->pos < l->length && is_dec_digit(l->bytes[l->pos])) l->pos += 1;
        size_t int_len = l->pos - int_start;

        /* §3.5.2: non-zero integer literals cannot have leading zeros.
         * The spec allows `0123.0` (float) but forbids `0123` (int).
         * Error iff the next char does NOT promote the literal to float. */
        if (int_len > 1 && l->bytes[int_start] == '0') {
            int next_promotes_to_float = 0;
            if (l->pos < l->length) {
                uint8_t c = l->bytes[l->pos];
                next_promotes_to_float =
                    (c == '.' || c == 'e' || c == 'E' || c == 'f' || c == 'h');
            }
            if (!next_promotes_to_float) {
                emit_error(l, (uint32_t)int_start, (uint32_t)int_len,
                           "non-zero integer literal cannot have leading zero");
                return 0;
            }
        }

        if (l->pos < l->length && l->bytes[l->pos] == '.') {
            is_float = 1;
            l->pos += 1;
            while (l->pos < l->length && is_dec_digit(l->bytes[l->pos])) l->pos += 1;
        }

check_dec_exponent:
        if (l->pos < l->length && (l->bytes[l->pos] == 'e' || l->bytes[l->pos] == 'E')) {
            is_float = 1;
            l->pos += 1;
            if (l->pos < l->length &&
                (l->bytes[l->pos] == '+' || l->bytes[l->pos] == '-')) l->pos += 1;
            size_t e0 = l->pos;
            while (l->pos < l->length && is_dec_digit(l->bytes[l->pos])) l->pos += 1;
            if (l->pos == e0) {
                emit_error(l, (uint32_t)start, (uint32_t)(l->pos - start),
                           "expected digits in float exponent");
                return 0;
            }
        }
    }

suffix_check:
    if (l->pos < l->length) {
        uint8_t c = l->bytes[l->pos];
        if (is_float) {
            if (c == 'f') { suffix = WGSL_NUM_SUFFIX_F; l->pos += 1; }
            else if (c == 'h') { suffix = WGSL_NUM_SUFFIX_H; l->pos += 1; }
        } else {
            if      (c == 'i') { suffix = WGSL_NUM_SUFFIX_I; l->pos += 1; }
            else if (c == 'u') { suffix = WGSL_NUM_SUFFIX_U; l->pos += 1; }
            else if (c == 'f') { suffix = WGSL_NUM_SUFFIX_F; l->pos += 1; is_float = 1; }
            else if (c == 'h') { suffix = WGSL_NUM_SUFFIX_H; l->pos += 1; is_float = 1; }
        }
    }

    if (is_hex) suffix |= WGSL_NUM_HEX;

    return push_tok(l,
                    is_float ? WGSL_TOK_FLOAT_LIT : WGSL_TOK_INT_LIT,
                    (uint32_t)start, (uint32_t)(l->pos - start), suffix);
}

/* Single-byte punctuation + maximal-munch operators.  Returns 0 on
 * unknown characters (with diagnostic emitted), 1 on success. */
static int scan_punct_or_op(L *l) {
    uint8_t b  = l->bytes[l->pos];
    uint8_t b2 = (l->pos + 1 < l->length) ? l->bytes[l->pos + 1] : 0;
    uint8_t b3 = (l->pos + 2 < l->length) ? l->bytes[l->pos + 2] : 0;
    uint32_t start = (uint32_t)l->pos;

    switch (b) {
    case '(': l->pos++; return push_tok(l, WGSL_TOK_LPAREN,    start, 1, 0);
    case ')': l->pos++; return push_tok(l, WGSL_TOK_RPAREN,    start, 1, 0);
    case '{': l->pos++; return push_tok(l, WGSL_TOK_LBRACE,    start, 1, 0);
    case '}': l->pos++; return push_tok(l, WGSL_TOK_RBRACE,    start, 1, 0);
    case '[': l->pos++; return push_tok(l, WGSL_TOK_LBRACKET,  start, 1, 0);
    case ']': l->pos++; return push_tok(l, WGSL_TOK_RBRACKET,  start, 1, 0);
    case ',': l->pos++; return push_tok(l, WGSL_TOK_COMMA,     start, 1, 0);
    case ';': l->pos++; return push_tok(l, WGSL_TOK_SEMICOLON, start, 1, 0);
    case ':': l->pos++; return push_tok(l, WGSL_TOK_COLON,     start, 1, 0);
    case '@': l->pos++; return push_tok(l, WGSL_TOK_AT,        start, 1, 0);
    case '~': l->pos++; return push_tok(l, WGSL_TOK_TILDE,     start, 1, 0);

    case '.':
        if (is_dec_digit(b2)) return scan_number(l);
        l->pos++;
        return push_tok(l, WGSL_TOK_DOT, start, 1, 0);

    case '&':
        if (b2 == '&') { l->pos += 2; return push_tok(l, WGSL_TOK_AMP_AMP,   start, 2, 0); }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_AMP_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_AMP, start, 1, 0);

    case '|':
        if (b2 == '|') { l->pos += 2; return push_tok(l, WGSL_TOK_PIPE_PIPE,  start, 2, 0); }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_PIPE_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_PIPE, start, 1, 0);

    case '^':
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_CARET_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_CARET, start, 1, 0);

    case '!':
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_BANG_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_BANG, start, 1, 0);

    case '+':
        if (b2 == '+') { l->pos += 2; return push_tok(l, WGSL_TOK_PLUS_PLUS,  start, 2, 0); }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_PLUS_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_PLUS, start, 1, 0);

    case '-':
        if (b2 == '-') { l->pos += 2; return push_tok(l, WGSL_TOK_MINUS_MINUS, start, 2, 0); }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_MINUS_EQUAL, start, 2, 0); }
        if (b2 == '>') { l->pos += 2; return push_tok(l, WGSL_TOK_ARROW,       start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_MINUS, start, 1, 0);

    case '*':
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_STAR_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_STAR, start, 1, 0);

    case '/':
        /* `//` and `/​*` were already handled by lex_step. */
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_SLASH_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_SLASH, start, 1, 0);

    case '%':
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_PERCENT_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_PERCENT, start, 1, 0);

    case '<':
        if (b2 == '<') {
            if (b3 == '=') { l->pos += 3; return push_tok(l, WGSL_TOK_LESS_LESS_EQUAL, start, 3, 0); }
            l->pos += 2;   return push_tok(l, WGSL_TOK_LESS_LESS,       start, 2, 0);
        }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_LESS_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_LESS, start, 1, 0);

    case '>':
        if (b2 == '>') {
            if (b3 == '=') { l->pos += 3; return push_tok(l, WGSL_TOK_GREATER_GREATER_EQUAL, start, 3, 0); }
            l->pos += 2;   return push_tok(l, WGSL_TOK_GREATER_GREATER,    start, 2, 0);
        }
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_GREATER_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_GREATER, start, 1, 0);

    case '=':
        if (b2 == '=') { l->pos += 2; return push_tok(l, WGSL_TOK_EQUAL_EQUAL, start, 2, 0); }
        l->pos++; return push_tok(l, WGSL_TOK_EQUAL, start, 1, 0);

    default:
        emit_error(l, start, 1,
                   "unexpected character '%c' (0x%02X)",
                   (b >= 0x20 && b < 0x7F) ? (char)b : '?', b);
        l->pos++;
        return 0;
    }
}

static int lex_step(L *l) {
    uint8_t b = l->bytes[l->pos];

    /* — Blankspace (ASCII fast path) — */
    if (is_ascii_blankspace(b)) return scan_blankspace(l);

    /* — Comments — */
    if (b == '/' && l->pos + 1 < l->length) {
        uint8_t b2 = l->bytes[l->pos + 1];
        if (b2 == '/') return scan_line_comment(l);
        if (b2 == '*') return scan_block_comment(l);
    }

    /* — Identifiers / keywords / lone `_` (ASCII fast path) — */
    if (is_ascii_ident_start(b)) return scan_ident(l);

    /* — Numeric literals — */
    if (is_dec_digit(b)) return scan_number(l);
    /* `.` followed by digit is a leading-dot float; handled by scan_punct_or_op. */

    /* — Punctuation / operators — */
    if (b < 0x80u) return scan_punct_or_op(l);

    /* — Non-ASCII path — */
    WGSLUtf8Decode d = wgsl_utf8_step(l->bytes + l->pos, l->length - l->pos);
    if (!d.consumed) {
        emit_error(l, (uint32_t)l->pos, 1, "invalid UTF-8 byte 0x%02X", b);
        l->pos += 1;
        return 0;
    }
    if (wgsl_utf8_is_blankspace(d.code_point)) return scan_blankspace(l);
    if (wgsl_utf8_is_ident_start(d.code_point)) return scan_ident(l);

    emit_error(l, (uint32_t)l->pos, (uint32_t)d.consumed,
               "unexpected character U+%04X", d.code_point);
    l->pos += (size_t)d.consumed;
    return 0;
}

int wgsl_tokenize(
    const WGSLSource *source,
    WGSLArena        *arena,
    WGSLDiagBag      *diag,
    WGSLLexResult    *out)
{
    if (!source || !arena || !diag || !out) return 0;

    L l;
    memset(&l, 0, sizeof l);
    l.src      = source;
    l.bytes    = (const uint8_t *)source->bytes;
    l.length   = source->length;
    l.arena    = arena;
    l.diag     = diag;
    l.capacity = WGSL_LEX_INITIAL_CAPACITY;
    l.tokens   = (WGSLToken *)malloc(l.capacity * sizeof *l.tokens);
    if (!l.tokens) return 0;

    while (l.pos < l.length) {
        if (!lex_step(&l)) {
            /* Diagnostic was emitted; continue scanning so we surface as
             * many errors as possible in one pass. */
        }
    }

    /* Always close with EOF, even on error, so consumers can rely on it. */
    push_tok(&l, WGSL_TOK_EOF, (uint32_t)l.length, 0, 0);

    /* Copy heap dynamic array into the long-lived arena. */
    WGSLToken *arena_tokens =
        (WGSLToken *)wgsl_arena_alloc(arena, l.count * sizeof *arena_tokens);
    if (!arena_tokens) {
        free(l.tokens);
        return 0;
    }
    memcpy(arena_tokens, l.tokens, l.count * sizeof *arena_tokens);
    free(l.tokens);

    out->tokens = arena_tokens;
    out->count  = l.count;

    /* §3.9 template-list discovery — runs after Round A so `<` / `>`
     * matching pairs are retagged TEMPLATE_START / TEMPLATE_END. */
    if (!wgsl_discover_templates(out, arena)) return 0;

    return l.had_error ? 0 : 1;
}

/* ─────────────────────────────────────────────────────────────────────
 *  WGSL §3.9 — template-list discovery on a pre-tokenized stream.
 * ───────────────────────────────────────────────────────────────────── */

/* Token kinds whose lexical form matches `ident_pattern_token`.  The
 * §3.9 algorithm pushes a candidate when one of these is followed by
 * `<`. */
static int is_ident_shaped(WGSLTokenKind k) {
    if (k == WGSL_TOK_IDENT)      return 1;
    if (k == WGSL_TOK_UNDERSCORE) return 1;
    /* All keywords match ident_pattern_token lexically. */
    if (k >= WGSL_TOK_KW_ALIAS && k <= WGSL_TOK_KW_WHILE) return 1;
    return 0;
}

static int is_trivia(WGSLTokenKind k) {
    return k == WGSL_TOK_BLANKSPACE
        || k == WGSL_TOK_LINE_COMMENT
        || k == WGSL_TOK_BLOCK_COMMENT;
}

/* Pending stack entry — points into the **output** array (so the LESS
 * we may eventually retag is reachable by index). */
typedef struct {
    size_t out_idx;     /* index of the LESS token in out[] */
    int    depth;       /* nesting at the time of push       */
} Pending;

#define WGSL_TPL_STACK_LIMIT 256

typedef struct {
    WGSLToken *out;
    size_t     count;
    size_t     capacity;
} TplOut;

static int tpl_grow(TplOut *o) {
    size_t new_cap = o->capacity ? o->capacity * 2 : 64;
    WGSLToken *grown = (WGSLToken *)realloc(o->out, new_cap * sizeof *grown);
    if (!grown) return 0;
    o->out = grown;
    o->capacity = new_cap;
    return 1;
}

static int tpl_append(TplOut *o, WGSLToken t) {
    if (o->count == o->capacity && !tpl_grow(o)) return 0;
    o->out[o->count++] = t;
    return 1;
}

/* Try to close a pending template at the current depth using the
 * just-appended token at out[idx], retagging it to TEMPLATE_END and
 * the matching LESS to TEMPLATE_START.  Returns 1 if a close happened. */
static int tpl_try_close(TplOut *o, Pending *stack, int *sp,
                         int depth, size_t close_idx)
{
    if (*sp > 0 && stack[*sp - 1].depth == depth) {
        size_t less_idx = stack[*sp - 1].out_idx;
        o->out[less_idx].kind  = WGSL_TOK_TEMPLATE_START;
        o->out[close_idx].kind = WGSL_TOK_TEMPLATE_END;
        *sp -= 1;
        return 1;
    }
    return 0;
}

/* Split `>>`, `>=`, `>>=`: the leading `>` closes a pending template,
 * the remainder is emitted as a fresh follow-up token.  Caller has
 * already determined a close is wanted; this routine writes both
 * tokens and recurses into a second close attempt for `>>` only.
 * Returns 1 on success, 0 on OOM. */
static int tpl_split_greater(
    TplOut *o, Pending *stack, int *sp, int depth,
    WGSLToken orig, WGSLTokenKind tail_kind, uint32_t tail_length)
{
    /* Emit head as TEMPLATE_END (length 1) */
    WGSLToken head = orig;
    head.kind        = WGSL_TOK_TEMPLATE_END;
    head.span.length = 1;
    if (!tpl_append(o, head)) return 0;
    size_t head_idx = o->count - 1;
    /* Retag matching LESS */
    if (*sp > 0) {
        size_t less_idx = stack[*sp - 1].out_idx;
        o->out[less_idx].kind = WGSL_TOK_TEMPLATE_START;
        *sp -= 1;
    }

    /* Emit tail */
    WGSLToken tail = orig;
    tail.kind        = tail_kind;
    tail.span.offset = orig.span.offset + 1;
    tail.span.length = tail_length;
    /* Column of the tail = column of head + 1 (same line guaranteed:
     * `>>` `>=` `>>=` are single-line tokens). */
    tail.column      = orig.column + 1;
    if (!tpl_append(o, tail)) return 0;

    /* For `>>`, the tail is GREATER; it might *also* close another
     * pending template at the same depth. */
    if (tail_kind == WGSL_TOK_GREATER) {
        size_t tail_idx = o->count - 1;
        (void)head_idx;
        tpl_try_close(o, stack, sp, depth, tail_idx);
    }
    return 1;
}

int wgsl_discover_templates(WGSLLexResult *result, WGSLArena *arena) {
    if (!result || !arena || result->count == 0) return 1;

    TplOut o;
    o.out = NULL;
    o.count = 0;
    o.capacity = 0;
    if (!tpl_grow(&o)) return 0;

    Pending stack[WGSL_TPL_STACK_LIMIT];
    int sp = 0;
    int depth = 0;
    int prev_ident_shaped = 0;

    for (size_t i = 0; i < result->count; i++) {
        WGSLToken t = result->tokens[i];
        WGSLTokenKind k = t.kind;

        if (is_trivia(k)) {
            if (!tpl_append(&o, t)) goto oom;
            continue;
        }

        if (is_ident_shaped(k)) {
            if (!tpl_append(&o, t)) goto oom;
            prev_ident_shaped = 1;
            continue;
        }

        switch (k) {
        case WGSL_TOK_LESS: {
            if (!tpl_append(&o, t)) goto oom;
            if (prev_ident_shaped && sp < WGSL_TPL_STACK_LIMIT) {
                stack[sp].out_idx = o.count - 1;
                stack[sp].depth   = depth;
                sp += 1;
            }
            break;
        }
        case WGSL_TOK_LESS_LESS:
        case WGSL_TOK_LESS_EQUAL:
        case WGSL_TOK_LESS_LESS_EQUAL:
            /* Operators — never templates.  Just append. */
            if (!tpl_append(&o, t)) goto oom;
            break;

        case WGSL_TOK_GREATER:
            if (!tpl_append(&o, t)) goto oom;
            tpl_try_close(&o, stack, &sp, depth, o.count - 1);
            break;

        case WGSL_TOK_GREATER_GREATER:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_GREATER, 1)) goto oom;
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_GREATER_EQUAL:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_EQUAL, 1)) goto oom;
                /* The trailing `=` is a lone assignment — clears stack. */
                sp = 0;
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_GREATER_GREATER_EQUAL:
            if (sp > 0 && stack[sp - 1].depth == depth) {
                if (!tpl_split_greater(&o, stack, &sp, depth, t,
                                       WGSL_TOK_GREATER_EQUAL, 2)) goto oom;
            } else {
                if (!tpl_append(&o, t)) goto oom;
            }
            break;

        case WGSL_TOK_LPAREN:
        case WGSL_TOK_LBRACKET:
            if (!tpl_append(&o, t)) goto oom;
            depth += 1;
            break;

        case WGSL_TOK_RPAREN:
        case WGSL_TOK_RBRACKET:
            if (!tpl_append(&o, t)) goto oom;
            /* Pop pending entries that were opened at or below this
             * depth — they cannot become templates anymore. */
            while (sp > 0 && stack[sp - 1].depth >= depth) sp -= 1;
            if (depth > 0) depth -= 1;
            break;

        case WGSL_TOK_SEMICOLON:
        case WGSL_TOK_LBRACE:
        case WGSL_TOK_COLON:
            if (!tpl_append(&o, t)) goto oom;
            sp = 0;
            depth = 0;
            break;

        case WGSL_TOK_AMP_AMP:
        case WGSL_TOK_PIPE_PIPE:
            /* Expression boundary at current depth. */
            if (!tpl_append(&o, t)) goto oom;
            while (sp > 0 && stack[sp - 1].depth >= depth) sp -= 1;
            break;

        case WGSL_TOK_EQUAL:
            /* Lone `=` → assignment.  Anything pending isn't a template. */
            if (!tpl_append(&o, t)) goto oom;
            sp = 0;
            break;

        default:
            if (!tpl_append(&o, t)) goto oom;
            break;
        }

        prev_ident_shaped = 0;
    }

    /* Copy the rebuilt stream into the arena, replacing the old. */
    WGSLToken *arena_out =
        (WGSLToken *)wgsl_arena_alloc(arena, o.count * sizeof *arena_out);
    if (!arena_out) goto oom;
    memcpy(arena_out, o.out, o.count * sizeof *arena_out);
    free(o.out);

    result->tokens = arena_out;
    result->count  = o.count;
    return 1;

oom:
    free(o.out);
    return 0;
}
