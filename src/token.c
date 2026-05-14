/**
 * @file token.c — Token kind names + keyword classifier.  See internal/token.h.
 */
#include "internal/token.h"

#include <stddef.h>
#include <string.h>

const char *wgsl_token_kind_name(WGSLTokenKind k) {
    switch (k) {
    case WGSL_TOK_INVALID:                   return "invalid";
    case WGSL_TOK_EOF:                       return "eof";
    case WGSL_TOK_IDENT:                     return "ident";
    case WGSL_TOK_INT_LIT:                   return "int_lit";
    case WGSL_TOK_FLOAT_LIT:                 return "float_lit";

    case WGSL_TOK_KW_ALIAS:                  return "kw_alias";
    case WGSL_TOK_KW_BREAK:                  return "kw_break";
    case WGSL_TOK_KW_CASE:                   return "kw_case";
    case WGSL_TOK_KW_CONST:                  return "kw_const";
    case WGSL_TOK_KW_CONST_ASSERT:           return "kw_const_assert";
    case WGSL_TOK_KW_CONTINUE:               return "kw_continue";
    case WGSL_TOK_KW_CONTINUING:             return "kw_continuing";
    case WGSL_TOK_KW_DEFAULT:                return "kw_default";
    case WGSL_TOK_KW_DIAGNOSTIC:             return "kw_diagnostic";
    case WGSL_TOK_KW_DISCARD:                return "kw_discard";
    case WGSL_TOK_KW_ELSE:                   return "kw_else";
    case WGSL_TOK_KW_ENABLE:                 return "kw_enable";
    case WGSL_TOK_KW_FALSE:                  return "kw_false";
    case WGSL_TOK_KW_FN:                     return "kw_fn";
    case WGSL_TOK_KW_FOR:                    return "kw_for";
    case WGSL_TOK_KW_IF:                     return "kw_if";
    case WGSL_TOK_KW_LET:                    return "kw_let";
    case WGSL_TOK_KW_LOOP:                   return "kw_loop";
    case WGSL_TOK_KW_OVERRIDE:               return "kw_override";
    case WGSL_TOK_KW_REQUIRES:               return "kw_requires";
    case WGSL_TOK_KW_RETURN:                 return "kw_return";
    case WGSL_TOK_KW_STRUCT:                 return "kw_struct";
    case WGSL_TOK_KW_SWITCH:                 return "kw_switch";
    case WGSL_TOK_KW_TRUE:                   return "kw_true";
    case WGSL_TOK_KW_VAR:                    return "kw_var";
    case WGSL_TOK_KW_WHILE:                  return "kw_while";
    case WGSL_TOK_RESERVED:                  return "reserved_word";

    case WGSL_TOK_LPAREN:                    return "(";
    case WGSL_TOK_RPAREN:                    return ")";
    case WGSL_TOK_LBRACE:                    return "{";
    case WGSL_TOK_RBRACE:                    return "}";
    case WGSL_TOK_LBRACKET:                  return "[";
    case WGSL_TOK_RBRACKET:                  return "]";
    case WGSL_TOK_COMMA:                     return ",";
    case WGSL_TOK_SEMICOLON:                 return ";";
    case WGSL_TOK_COLON:                     return ":";
    case WGSL_TOK_DOT:                       return ".";
    case WGSL_TOK_AT:                        return "@";
    case WGSL_TOK_UNDERSCORE:                return "_";

    case WGSL_TOK_AMP:                       return "&";
    case WGSL_TOK_AMP_AMP:                   return "&&";
    case WGSL_TOK_AMP_EQUAL:                 return "&=";
    case WGSL_TOK_PIPE:                      return "|";
    case WGSL_TOK_PIPE_PIPE:                 return "||";
    case WGSL_TOK_PIPE_EQUAL:                return "|=";
    case WGSL_TOK_CARET:                     return "^";
    case WGSL_TOK_CARET_EQUAL:               return "^=";
    case WGSL_TOK_TILDE:                     return "~";
    case WGSL_TOK_BANG:                      return "!";
    case WGSL_TOK_BANG_EQUAL:                return "!=";
    case WGSL_TOK_PLUS:                      return "+";
    case WGSL_TOK_PLUS_PLUS:                 return "++";
    case WGSL_TOK_PLUS_EQUAL:                return "+=";
    case WGSL_TOK_MINUS:                     return "-";
    case WGSL_TOK_MINUS_MINUS:               return "--";
    case WGSL_TOK_MINUS_EQUAL:               return "-=";
    case WGSL_TOK_ARROW:                     return "->";
    case WGSL_TOK_STAR:                      return "*";
    case WGSL_TOK_STAR_EQUAL:                return "*=";
    case WGSL_TOK_SLASH:                     return "/";
    case WGSL_TOK_SLASH_EQUAL:               return "/=";
    case WGSL_TOK_PERCENT:                   return "%";
    case WGSL_TOK_PERCENT_EQUAL:             return "%=";
    case WGSL_TOK_LESS:                      return "<";
    case WGSL_TOK_LESS_EQUAL:                return "<=";
    case WGSL_TOK_LESS_LESS:                 return "<<";
    case WGSL_TOK_LESS_LESS_EQUAL:           return "<<=";
    case WGSL_TOK_GREATER:                   return ">";
    case WGSL_TOK_GREATER_EQUAL:             return ">=";
    case WGSL_TOK_GREATER_GREATER:           return ">>";
    case WGSL_TOK_GREATER_GREATER_EQUAL:     return ">>=";
    case WGSL_TOK_EQUAL:                     return "=";
    case WGSL_TOK_EQUAL_EQUAL:               return "==";

    case WGSL_TOK_TEMPLATE_START:            return "<:tmpl";
    case WGSL_TOK_TEMPLATE_END:              return ">:tmpl";

    case WGSL_TOK_LINE_COMMENT:              return "line_comment";
    case WGSL_TOK_BLOCK_COMMENT:             return "block_comment";
    case WGSL_TOK_BLANKSPACE:                return "blankspace";

    case WGSL_TOK_KIND_COUNT:
        break;
    }
    return "<unknown>";
}

/* — Keyword table — sorted by length, then lexicographically.  All ASCII,
 * so a byte memcmp is sufficient. */
typedef struct { const char *text; uint8_t length; WGSLTokenKind kind; } KW;

static const KW KW_TABLE[] = {
    /* Sorted by length, then lexicographically.  Length field MUST
     * match strlen(text) — `wgsl_token_classify_ident` keys off it
     * for both the length-equality check and the memcmp range. */

    /* length 2 */
    { "as",                2, WGSL_TOK_RESERVED       },
    { "do",                2, WGSL_TOK_RESERVED       },
    { "fn",                2, WGSL_TOK_KW_FN          },
    { "i8",                2, WGSL_TOK_IDENT          },
    { "if",                2, WGSL_TOK_KW_IF          },
    { "of",                2, WGSL_TOK_RESERVED       },
    { "u8",                2, WGSL_TOK_IDENT          },
    /* length 3 */
    { "asm",               3, WGSL_TOK_RESERVED       },
    { "f16",               3, WGSL_TOK_IDENT          },
    { "f64",               3, WGSL_TOK_IDENT          },
    { "for",               3, WGSL_TOK_KW_FOR         },
    { "get",               3, WGSL_TOK_RESERVED       },
    { "i16",               3, WGSL_TOK_IDENT          },
    { "i64",               3, WGSL_TOK_IDENT          },
    { "let",               3, WGSL_TOK_KW_LET         },
    { "mod",               3, WGSL_TOK_RESERVED       },
    { "mut",               3, WGSL_TOK_RESERVED       },
    { "new",               3, WGSL_TOK_RESERVED       },
    { "nil",               3, WGSL_TOK_RESERVED       },
    { "pub",               3, WGSL_TOK_RESERVED       },
    { "ref",               3, WGSL_TOK_RESERVED       },
    { "set",               3, WGSL_TOK_RESERVED       },
    { "std",               3, WGSL_TOK_RESERVED       },
    { "try",               3, WGSL_TOK_RESERVED       },
    { "u16",               3, WGSL_TOK_IDENT          },
    { "u64",               3, WGSL_TOK_IDENT          },
    { "use",               3, WGSL_TOK_RESERVED       },
    { "var",               3, WGSL_TOK_KW_VAR         },
    /* length 4 */
    { "NULL",              4, WGSL_TOK_RESERVED       },
    { "Self",              4, WGSL_TOK_RESERVED       },
    { "auto",              4, WGSL_TOK_RESERVED       },
    { "bf16",              4, WGSL_TOK_IDENT          },
    { "case",              4, WGSL_TOK_KW_CASE        },
    { "cast",              4, WGSL_TOK_RESERVED       },
    { "else",              4, WGSL_TOK_KW_ELSE        },
    { "enum",              4, WGSL_TOK_RESERVED       },
    { "from",              4, WGSL_TOK_RESERVED       },
    { "goto",              4, WGSL_TOK_RESERVED       },
    { "impl",              4, WGSL_TOK_RESERVED       },
    { "loop",              4, WGSL_TOK_KW_LOOP        },
    { "lowp",              4, WGSL_TOK_RESERVED       },
    { "meta",              4, WGSL_TOK_RESERVED       },
    { "move",              4, WGSL_TOK_RESERVED       },
    { "null",              4, WGSL_TOK_RESERVED       },
    { "pass",              4, WGSL_TOK_RESERVED       },
    { "priv",              4, WGSL_TOK_RESERVED       },
    { "self",              4, WGSL_TOK_RESERVED       },
    { "this",              4, WGSL_TOK_RESERVED       },
    { "true",              4, WGSL_TOK_KW_TRUE        },
    { "type",              4, WGSL_TOK_RESERVED       },
    { "wgsl",              4, WGSL_TOK_RESERVED       },
    { "with",              4, WGSL_TOK_RESERVED       },
    /* length 5 */
    { "alias",             5, WGSL_TOK_KW_ALIAS       },
    { "async",             5, WGSL_TOK_RESERVED       },
    { "await",             5, WGSL_TOK_RESERVED       },
    { "break",             5, WGSL_TOK_KW_BREAK       },
    { "catch",             5, WGSL_TOK_RESERVED       },
    { "class",             5, WGSL_TOK_RESERVED       },
    { "const",             5, WGSL_TOK_KW_CONST       },
    { "crate",             5, WGSL_TOK_RESERVED       },
    { "false",             5, WGSL_TOK_KW_FALSE       },
    { "final",             5, WGSL_TOK_RESERVED       },
    { "highp",             5, WGSL_TOK_RESERVED       },
    { "macro",             5, WGSL_TOK_RESERVED       },
    { "match",             5, WGSL_TOK_RESERVED       },
    { "patch",             5, WGSL_TOK_RESERVED       },
    { "snorm",             5, WGSL_TOK_RESERVED       },
    { "super",             5, WGSL_TOK_RESERVED       },
    { "throw",             5, WGSL_TOK_RESERVED       },
    { "trait",             5, WGSL_TOK_RESERVED       },
    { "union",             5, WGSL_TOK_RESERVED       },
    { "unorm",             5, WGSL_TOK_RESERVED       },
    { "using",             5, WGSL_TOK_RESERVED       },
    { "where",             5, WGSL_TOK_RESERVED       },
    { "while",             5, WGSL_TOK_KW_WHILE       },
    { "yield",             5, WGSL_TOK_RESERVED       },
    /* length 6 */
    { "active",            6, WGSL_TOK_RESERVED       },
    { "become",            6, WGSL_TOK_RESERVED       },
    { "common",            6, WGSL_TOK_RESERVED       },
    { "delete",            6, WGSL_TOK_RESERVED       },
    { "demote",            6, WGSL_TOK_RESERVED       },
    { "enable",            6, WGSL_TOK_KW_ENABLE      },
    { "export",            6, WGSL_TOK_RESERVED       },
    { "extern",            6, WGSL_TOK_RESERVED       },
    { "filter",            6, WGSL_TOK_RESERVED       },
    { "friend",            6, WGSL_TOK_RESERVED       },
    { "import",            6, WGSL_TOK_RESERVED       },
    { "inline",            6, WGSL_TOK_RESERVED       },
    { "layout",            6, WGSL_TOK_RESERVED       },
    { "module",            6, WGSL_TOK_RESERVED       },
    { "public",            6, WGSL_TOK_RESERVED       },
    { "return",            6, WGSL_TOK_KW_RETURN      },
    { "shared",            6, WGSL_TOK_RESERVED       },
    { "sizeof",            6, WGSL_TOK_RESERVED       },
    { "smooth",            6, WGSL_TOK_RESERVED       },
    { "static",            6, WGSL_TOK_RESERVED       },
    { "struct",            6, WGSL_TOK_KW_STRUCT      },
    { "switch",            6, WGSL_TOK_KW_SWITCH      },
    { "target",            6, WGSL_TOK_RESERVED       },
    { "typeid",            6, WGSL_TOK_RESERVED       },
    { "typeof",            6, WGSL_TOK_RESERVED       },
    { "unless",            6, WGSL_TOK_RESERVED       },
    { "unsafe",            6, WGSL_TOK_RESERVED       },
    /* length 7 */
    { "alignas",           7, WGSL_TOK_RESERVED       },
    { "alignof",           7, WGSL_TOK_RESERVED       },
    { "compile",           7, WGSL_TOK_RESERVED       },
    { "concept",           7, WGSL_TOK_RESERVED       },
    { "default",           7, WGSL_TOK_KW_DEFAULT     },
    { "discard",           7, WGSL_TOK_KW_DISCARD     },
    { "extends",           7, WGSL_TOK_RESERVED       },
    { "finally",           7, WGSL_TOK_RESERVED       },
    { "fxgroup",           7, WGSL_TOK_RESERVED       },
    { "mediump",           7, WGSL_TOK_RESERVED       },
    { "mutable",           7, WGSL_TOK_RESERVED       },
    { "nullptr",           7, WGSL_TOK_RESERVED       },
    { "package",           7, WGSL_TOK_RESERVED       },
    { "precise",           7, WGSL_TOK_RESERVED       },
    { "require",           7, WGSL_TOK_RESERVED       },
    { "typedef",           7, WGSL_TOK_RESERVED       },
    { "unsized",           7, WGSL_TOK_RESERVED       },
    { "varying",           7, WGSL_TOK_RESERVED       },
    { "virtual",           7, WGSL_TOK_RESERVED       },
    /* length 8 */
    { "abstract",          8, WGSL_TOK_RESERVED       },
    { "co_await",          8, WGSL_TOK_RESERVED       },
    { "co_yield",          8, WGSL_TOK_RESERVED       },
    { "coherent",          8, WGSL_TOK_RESERVED       },
    { "continue",          8, WGSL_TOK_KW_CONTINUE    },
    { "debugger",          8, WGSL_TOK_RESERVED       },
    { "decltype",          8, WGSL_TOK_RESERVED       },
    { "explicit",          8, WGSL_TOK_RESERVED       },
    { "external",          8, WGSL_TOK_RESERVED       },
    { "noexcept",          8, WGSL_TOK_RESERVED       },
    { "noinline",          8, WGSL_TOK_RESERVED       },
    { "operator",          8, WGSL_TOK_RESERVED       },
    { "override",          8, WGSL_TOK_KW_OVERRIDE    },
    { "premerge",          8, WGSL_TOK_RESERVED       },
    { "readonly",          8, WGSL_TOK_RESERVED       },
    { "register",          8, WGSL_TOK_RESERVED       },
    { "requires",          8, WGSL_TOK_KW_REQUIRES    },
    { "resource",          8, WGSL_TOK_RESERVED       },
    { "restrict",          8, WGSL_TOK_RESERVED       },
    { "template",          8, WGSL_TOK_RESERVED       },
    { "typename",          8, WGSL_TOK_RESERVED       },
    { "volatile",          8, WGSL_TOK_RESERVED       },
    /* length 9 */
    { "attribute",         9, WGSL_TOK_RESERVED       },
    { "co_return",         9, WGSL_TOK_RESERVED       },
    { "consteval",         9, WGSL_TOK_RESERVED       },
    { "constexpr",         9, WGSL_TOK_RESERVED       },
    { "constinit",         9, WGSL_TOK_RESERVED       },
    { "interface",         9, WGSL_TOK_RESERVED       },
    { "namespace",         9, WGSL_TOK_RESERVED       },
    { "partition",         9, WGSL_TOK_RESERVED       },
    { "precision",         9, WGSL_TOK_RESERVED       },
    { "protected",         9, WGSL_TOK_RESERVED       },
    { "writeonly",         9, WGSL_TOK_RESERVED       },
    /* length 10 */
    { "const_cast",       10, WGSL_TOK_RESERVED       },
    { "continuing",       10, WGSL_TOK_KW_CONTINUING  },
    { "diagnostic",       10, WGSL_TOK_KW_DIAGNOSTIC  },
    { "implements",       10, WGSL_TOK_RESERVED       },
    { "instanceof",       10, WGSL_TOK_RESERVED       },
    { "packoffset",       10, WGSL_TOK_RESERVED       },
    { "regardless",       10, WGSL_TOK_RESERVED       },
    { "subroutine",       10, WGSL_TOK_RESERVED       },
    /* length 11 */
    { "fallthrough",      11, WGSL_TOK_RESERVED       },
    { "groupshared",      11, WGSL_TOK_RESERVED       },
    { "macro_rules",      11, WGSL_TOK_RESERVED       },
    { "noncoherent",      11, WGSL_TOK_RESERVED       },
    { "static_cast",      11, WGSL_TOK_RESERVED       },
    /* length 12 */
    { "asm_fragment",     12, WGSL_TOK_RESERVED       },
    { "column_major",     12, WGSL_TOK_RESERVED       },
    { "const_assert",     12, WGSL_TOK_KW_CONST_ASSERT},
    { "dynamic_cast",     12, WGSL_TOK_RESERVED       },
    { "non_coherent",     12, WGSL_TOK_RESERVED       },
    { "thread_local",     12, WGSL_TOK_RESERVED       },
    /* length 13 */
    { "noperspective",    13, WGSL_TOK_RESERVED       },
    { "pixelfragment",    13, WGSL_TOK_RESERVED       },
    { "static_assert",    13, WGSL_TOK_RESERVED       },
    /* length 15 */
    { "nointerpolation",  15, WGSL_TOK_RESERVED       },
    /* length 16 */
    { "compile_fragment", 16, WGSL_TOK_RESERVED       },
    { "demote_to_helper", 16, WGSL_TOK_RESERVED       },
    { "reinterpret_cast", 16, WGSL_TOK_RESERVED       },
};

#define KW_COUNT ((int)(sizeof(KW_TABLE) / sizeof(KW_TABLE[0])))

WGSLTokenKind wgsl_token_classify_ident(const char *bytes, size_t length) {
    /* Lengths range 2..12; quick reject. */
    /* Longest entry in KW_TABLE is 16 chars ("compile_fragment" /
     * "demote_to_helper" / "reinterpret_cast").  Anything outside
     * [2, 16] can't possibly match. */
    if (length < 2 || length > 16) return WGSL_TOK_IDENT;
    for (int i = 0; i < KW_COUNT; i++) {
        const KW *k = &KW_TABLE[i];
        if (k->length == (uint8_t)length && memcmp(k->text, bytes, length) == 0) {
            return k->kind;
        }
    }
    return WGSL_TOK_IDENT;
}
