/**
 * @file token.h — Lexer token type + full WGSL token kind enum.
 *
 * Layout: 24 bytes.
 *
 *   offset  size  field
 *   ──────  ────  ─────────────────────────────────────────────
 *      0      4   kind            WGSLTokenKind
 *      4      4   span.offset
 *      8      4   span.length
 *     12      4   line            1-based
 *     16      4   column          1-based byte distance from line start
 *     20      4   payload         per-kind extra (literal-suffix flags, …)
 *   ────────────  24 bytes total
 *
 * Tokens are stored contiguously in an arena-backed array.  The
 * `payload` slot carries kind-specific extras such as the numeric
 * suffix flag set (i / u / f / h / abstract).
 *
 * Kind IDs are stable ABI for any future tooling that walks raw
 * token streams; never reorder or renumber existing entries.
 */
#ifndef WGSL_INTERNAL_TOKEN_H
#define WGSL_INTERNAL_TOKEN_H

#include <stddef.h>
#include <stdint.h>

#include "internal/span.h"

typedef enum {
    WGSL_TOK_INVALID = 0,
    WGSL_TOK_EOF,

    /* — Identifiers and literals — */
    WGSL_TOK_IDENT,                 /* user identifier or context-dep name */
    WGSL_TOK_INT_LIT,               /* int literal (decimal or hex)        */
    WGSL_TOK_FLOAT_LIT,             /* float literal (decimal or hex)      */
    /* — Boolean literals are the keywords WGSL_TOK_KW_TRUE / _FALSE — */

    /* — Keywords — exactly the 27 listed in WGSL §16.1 — */
    WGSL_TOK_KW_ALIAS,
    WGSL_TOK_KW_BREAK,
    WGSL_TOK_KW_CASE,
    WGSL_TOK_KW_CONST,
    WGSL_TOK_KW_CONST_ASSERT,
    WGSL_TOK_KW_CONTINUE,
    WGSL_TOK_KW_CONTINUING,
    WGSL_TOK_KW_DEFAULT,
    WGSL_TOK_KW_DIAGNOSTIC,
    WGSL_TOK_KW_DISCARD,
    WGSL_TOK_KW_ELSE,
    WGSL_TOK_KW_ENABLE,
    WGSL_TOK_KW_FALSE,
    WGSL_TOK_KW_FN,
    WGSL_TOK_KW_FOR,
    WGSL_TOK_KW_IF,
    WGSL_TOK_KW_LET,
    WGSL_TOK_KW_LOOP,
    WGSL_TOK_KW_OVERRIDE,
    WGSL_TOK_KW_REQUIRES,
    WGSL_TOK_KW_RETURN,
    WGSL_TOK_KW_STRUCT,
    WGSL_TOK_KW_SWITCH,
    WGSL_TOK_KW_TRUE,
    WGSL_TOK_KW_VAR,
    WGSL_TOK_KW_WHILE,

    /* — Punctuation (§16.3) — */
    WGSL_TOK_LPAREN,                /* (  */
    WGSL_TOK_RPAREN,                /* )  */
    WGSL_TOK_LBRACE,                /* {  */
    WGSL_TOK_RBRACE,                /* }  */
    WGSL_TOK_LBRACKET,              /* [  */
    WGSL_TOK_RBRACKET,              /* ]  */
    WGSL_TOK_COMMA,                 /* ,  */
    WGSL_TOK_SEMICOLON,             /* ;  */
    WGSL_TOK_COLON,                 /* :  */
    WGSL_TOK_DOT,                   /* .  */
    WGSL_TOK_AT,                    /* @  — attribute introducer            */
    WGSL_TOK_UNDERSCORE,            /* _  — phony assignment LHS            */

    /* — Operators (§16.3, all maximal-munch in lexer) — */
    WGSL_TOK_AMP,                   /* &      */
    WGSL_TOK_AMP_AMP,               /* &&     */
    WGSL_TOK_AMP_EQUAL,             /* &=     */
    WGSL_TOK_PIPE,                  /* |      */
    WGSL_TOK_PIPE_PIPE,             /* ||     */
    WGSL_TOK_PIPE_EQUAL,            /* |=     */
    WGSL_TOK_CARET,                 /* ^      */
    WGSL_TOK_CARET_EQUAL,           /* ^=     */
    WGSL_TOK_TILDE,                 /* ~      */
    WGSL_TOK_BANG,                  /* !      */
    WGSL_TOK_BANG_EQUAL,            /* !=     */
    WGSL_TOK_PLUS,                  /* +      */
    WGSL_TOK_PLUS_PLUS,             /* ++     */
    WGSL_TOK_PLUS_EQUAL,            /* +=     */
    WGSL_TOK_MINUS,                 /* -      */
    WGSL_TOK_MINUS_MINUS,           /* --     */
    WGSL_TOK_MINUS_EQUAL,           /* -=     */
    WGSL_TOK_ARROW,                 /* ->     */
    WGSL_TOK_STAR,                  /* *      */
    WGSL_TOK_STAR_EQUAL,            /* *=     */
    WGSL_TOK_SLASH,                 /* /      */
    WGSL_TOK_SLASH_EQUAL,           /* /=     */
    WGSL_TOK_PERCENT,               /* %      */
    WGSL_TOK_PERCENT_EQUAL,         /* %=     */
    WGSL_TOK_LESS,                  /* <  (may retag to TEMPLATE_START) */
    WGSL_TOK_LESS_EQUAL,            /* <=     */
    WGSL_TOK_LESS_LESS,             /* <<     */
    WGSL_TOK_LESS_LESS_EQUAL,       /* <<=    */
    WGSL_TOK_GREATER,               /* >  (may retag to TEMPLATE_END)   */
    WGSL_TOK_GREATER_EQUAL,         /* >=     */
    WGSL_TOK_GREATER_GREATER,       /* >>     */
    WGSL_TOK_GREATER_GREATER_EQUAL, /* >>=    */
    WGSL_TOK_EQUAL,                 /* =      */
    WGSL_TOK_EQUAL_EQUAL,           /* ==     */

    /* — Template-list disambiguation tags (§3.9) — */
    WGSL_TOK_TEMPLATE_START,        /* < retagged after §3.9              */
    WGSL_TOK_TEMPLATE_END,          /* > retagged after §3.9              */

    /* — Markers preserved for editor round-trip — */
    WGSL_TOK_LINE_COMMENT,          /* // …                               */
    WGSL_TOK_BLOCK_COMMENT,         /* /​* … *​/, possibly nested          */
    WGSL_TOK_BLANKSPACE,            /* run of WGSL §3.2 blankspace        */

    WGSL_TOK_KIND_COUNT,
} WGSLTokenKind;

/* — Numeric literal payload bits.  Set in WGSLToken.payload for INT/FLOAT
 * literal tokens.  Mutually exclusive within a token. — */
enum {
    WGSL_NUM_ABSTRACT = 0u,         /* no suffix — AbstractInt or AbstractFloat */
    WGSL_NUM_SUFFIX_I = 1u,         /* `i` → i32                                */
    WGSL_NUM_SUFFIX_U = 2u,         /* `u` → u32                                */
    WGSL_NUM_SUFFIX_F = 3u,         /* `f` → f32                                */
    WGSL_NUM_SUFFIX_H = 4u,         /* `h` → f16 (requires `enable f16`)        */
    WGSL_NUM_HEX      = 1u << 4,    /* hex notation (orthogonal to suffix)      */
};

typedef struct WGSLToken {
    WGSLTokenKind kind;
    WGSLSpan      span;
    uint32_t      line;     /* 1-based */
    uint32_t      column;   /* 1-based byte distance from line start */
    uint32_t      payload;  /* numeric-suffix flags etc.; see above  */
} WGSLToken;

_Static_assert(sizeof(WGSLToken) == 24,
    "WGSLToken must be exactly 24 bytes; see internal/token.h.");
_Static_assert(_Alignof(WGSLToken) == 4,
    "WGSLToken must be 4-byte aligned.");
_Static_assert(WGSL_TOK_KIND_COUNT < (1u << 16),
    "WGSLTokenKind must fit in 16 bits.");

/* — Helpers (defined in token.c) — */

/** Short identifier-style name for diagnostics (`"kw_fn"`, `"int_lit"`, …). */
const char *wgsl_token_kind_name(WGSLTokenKind k);

/**
 * Classify an identifier-shape byte range as a keyword if it matches one of
 * the 27 in §16.1; otherwise return WGSL_TOK_IDENT.  ASCII-only — non-ASCII
 * bytes can never form a keyword, so a fast byte-level match is correct.
 */
WGSLTokenKind wgsl_token_classify_ident(const char *bytes, size_t length);

#endif /* WGSL_INTERNAL_TOKEN_H */
