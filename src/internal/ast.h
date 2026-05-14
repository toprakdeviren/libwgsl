/**
 * @file ast.h — WGSL AST node, locked at 48 bytes.
 *
 * Layout (one cache-friendly node, 8-byte aligned):
 *
 *   offset  size  field
 *   ──────  ────  ───────────────────────────────────────────────
 *      0      2   kind            WGSLNodeKind (16-bit fits 50+ kinds)
 *      2      2   flags           bit-packed analyzer state
 *      4      4   child_count
 *      8      4   span_offset     byte offset into source (0-based)
 *     12      4   span_length     byte length
 *     16      8   children        pointer into the same arena
 *     24     24   payload[3]      kind-specific 24-byte slot
 *   ──────────── 48 bytes total
 *
 * Type information, resolver bindings, and validator verdicts live in
 * **parallel arenas** keyed by node-arena offset; they never grow this
 * struct.  See `DESIGN.md` §5.
 *
 * The 48-byte size is enforced by the `_Static_assert` at the bottom
 * and by `tests/01-sizeof/test_sizeof.c` (Phase 1 sub-deliverable a).
 */
#ifndef WGSL_INTERNAL_AST_H
#define WGSL_INTERNAL_AST_H

#include <stdint.h>

typedef struct WGSLNode WGSLNode;

/** Kinds.  Stable IDs; never reorder, only append before _COUNT. */
typedef enum {
    WGSL_NODE_INVALID = 0,

    /* — Module structure — */
    WGSL_NODE_TRANSLATION_UNIT,

    /* — Declarations — */
    WGSL_NODE_DECL_STRUCT,
    WGSL_NODE_DECL_STRUCT_MEMBER,
    WGSL_NODE_DECL_FUNCTION,
    WGSL_NODE_DECL_PARAM,
    WGSL_NODE_DECL_ALIAS,
    WGSL_NODE_DECL_CONST,
    WGSL_NODE_DECL_OVERRIDE,
    WGSL_NODE_DECL_VAR,
    WGSL_NODE_DECL_LET,
    WGSL_NODE_DECL_CONST_ASSERT,

    /* — Statements — */
    WGSL_NODE_STMT_COMPOUND,
    WGSL_NODE_STMT_ASSIGN,
    WGSL_NODE_STMT_COMPOUND_ASSIGN,
    WGSL_NODE_STMT_PHONY_ASSIGN,
    WGSL_NODE_STMT_INCREMENT,
    WGSL_NODE_STMT_DECREMENT,
    WGSL_NODE_STMT_IF,
    WGSL_NODE_STMT_ELSE_IF,
    WGSL_NODE_STMT_ELSE,
    WGSL_NODE_STMT_SWITCH,
    WGSL_NODE_STMT_CASE_CLAUSE,
    WGSL_NODE_STMT_LOOP,
    WGSL_NODE_STMT_FOR,
    WGSL_NODE_STMT_WHILE,
    WGSL_NODE_STMT_BREAK,
    WGSL_NODE_STMT_BREAK_IF,
    WGSL_NODE_STMT_CONTINUE,
    WGSL_NODE_STMT_CONTINUING,
    WGSL_NODE_STMT_RETURN,
    WGSL_NODE_STMT_DISCARD,
    WGSL_NODE_STMT_FN_CALL,

    /* — Expressions — */
    WGSL_NODE_EXPR_LITERAL_BOOL,
    WGSL_NODE_EXPR_LITERAL_INT,
    WGSL_NODE_EXPR_LITERAL_FLOAT,
    WGSL_NODE_EXPR_IDENT,
    WGSL_NODE_EXPR_TEMPLATED_IDENT,     /* vec3<f32>, atomic<u32>, ... */
    WGSL_NODE_EXPR_PAREN,
    WGSL_NODE_EXPR_BINARY,
    WGSL_NODE_EXPR_UNARY,
    WGSL_NODE_EXPR_CALL,
    WGSL_NODE_EXPR_INDEX,
    WGSL_NODE_EXPR_MEMBER,              /* a.b   (also covers swizzle) */
    WGSL_NODE_EXPR_ADDR_OF,             /* &x */
    WGSL_NODE_EXPR_INDIRECTION,         /* *p */

    /* — Type specifiers and attributes are nested as templated_ident — */
    WGSL_NODE_ATTRIBUTE,                /* @group(0), @binding(2), ... */
    WGSL_NODE_TEMPLATE_ARG_LIST,        /* < ... > opaque list at parse time */

    WGSL_NODE_DIR_ENABLE,
    WGSL_NODE_DIR_REQUIRES,
    WGSL_NODE_DIR_DIAGNOSTIC,

    WGSL_NODE_KIND_COUNT,
} WGSLNodeKind;

/** Flag bits packed in `WGSLNode.flags`. */
enum {
    WGSL_FLAG_IS_REF        = 1u << 0,  /* type checker: value is a reference */
    WGSL_FLAG_HAS_ATTRS     = 1u << 1,  /* parser: node carries @attribute(s) */
    WGSL_FLAG_HAS_TEMPLATE  = 1u << 2,  /* parser: node carries < ... > args  */
    WGSL_FLAG_RESOLVED      = 1u << 3,  /* resolver visited this node          */
    WGSL_FLAG_TYPED         = 1u << 4,  /* type checker assigned a type        */
    WGSL_FLAG_CONST_EVALED  = 1u << 5,  /* const-evaluator produced a value    */
    WGSL_FLAG_TYPING        = 1u << 6,  /* type checker is currently typing    */
    WGSL_FLAG_CONST_EVALING = 1u << 7,  /* const-evaluator is currently typing */
    WGSL_FLAG_NONUNIFORM    = 1u << 8,  /* §15.2.9: expr is may-be-non-uniform */
    WGSL_FLAG_DIVERGENT_CF  = 1u << 9,  /* §15.2.9: stmt reached in divergent CF */
    WGSL_FLAG_ATTR_PARENS   = 1u << 10, /* parser: attribute used (...) syntax */
    WGSL_FLAG_SUBGROUP_UNIFORM = 1u << 11, /* §15.2.10: subgroup-uniform expr */
    /* — bits 12..15 reserved — */
};

struct WGSLNode {
    uint16_t   kind;          /* WGSLNodeKind                                */
    uint16_t   flags;         /* WGSL_FLAG_*                                 */
    uint32_t   child_count;   /* number of entries in `children`             */
    uint32_t   span_offset;   /* byte offset into source                     */
    uint32_t   span_length;   /* byte length                                 */
    WGSLNode **children;      /* arena-owned array; NULL when child_count==0 */
    uint64_t   payload[3];    /* kind-specific 24-byte slot (raw bytes)      */
};

/* — Phase 1 sub-deliverable (a): the size gate. — */
_Static_assert(sizeof(WGSLNode) == 48,
    "WGSLNode must be exactly 48 bytes; see docs/PLAN.md Phase 1 (a).");
_Static_assert(_Alignof(WGSLNode) == 8,
    "WGSLNode must be 8-byte aligned.");
_Static_assert(WGSL_NODE_KIND_COUNT < (1 << 16),
    "WGSLNodeKind must fit in 16 bits.");

#endif /* WGSL_INTERNAL_AST_H */
