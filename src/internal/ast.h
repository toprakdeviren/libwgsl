/**
 * @file ast.h — WGSL AST node, locked at 48 bytes.
 *
 * Layout (one cache-friendly node, 8-byte aligned):
 *
 *   offset  size  field
 *   ------  ----  -----------------------------------------------
 *      0      2   kind            WGSLNodeKind (16-bit fits 50+ kinds)
 *      2      2   flags           bit-packed analyzer state
 *      4      4   child_count
 *      8      4   span_offset     byte offset into source (0-based)
 *     12      4   span_length     byte length
 *     16      8   children        pointer into the same arena
 *     24     24   payload[3]      kind-specific 24-byte slot
 *   ------------ 48 bytes total
 *
 * Type information, resolver bindings, and validator verdicts live in
 * **parallel arenas** keyed by node-arena offset; they never grow this
 * struct.  See `docs/libwgsl.md` §3.
 *
 * The 48-byte size is enforced by the `_Static_assert` at the bottom
 * and by `tests/01-sizeof/test_sizeof.c`.
 */
#ifndef WGSL_INTERNAL_AST_H
#define WGSL_INTERNAL_AST_H

#include <stdint.h>
#include "internal/span.h"

typedef struct WGSLNode WGSLNode;

/* Maximum AST-node recursion depth for the downstream passes that walk the
 * tree recursively (resolver, const-evaluator, type-checker, validator,
 * ast-dump).  The parser caps its OWN recursion (WGSL_PARSE_MAX_DEPTH), but
 * left-leaning chains built iteratively (`a+a+…`, `a[0][0]…`, `a.b.b…`,
 * `a()()…`) produce an arbitrarily deep AST with no parser recursion, and
 * every downstream pass then recurses on that depth.  Each such walker
 * checks this cap and emits a "nested too deeply" diagnostic instead of
 * overflowing the call stack — a denial-of-service on untrusted WGSL.
 *
 * Chosen so the deepest guarded walk stays within the 2 MB wasm stack (the
 * studio runtime; see Makefile STACK_SIZE) with comfortable margin, and so
 * the value exceeds the deepest AST a parser-accepted program can produce
 * (~2 * WGSL_PARSE_MAX_DEPTH nodes) — a valid nested tree is thus never
 * rejected here; only truly unbounded left-leaning chains (`a+a+…` etc.,
 * which the parser does not nesting-cap) hit this limit.  Assumes a stack
 * of >= ~1 MB (native main threads are 8 MB, the wasm build 2 MB); embedders
 * running on a smaller worker-thread stack should define a lower value. */
#ifndef WGSL_MAX_AST_DEPTH
#define WGSL_MAX_AST_DEPTH 512
#endif

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

/* — Layout gate. — */
_Static_assert(sizeof(WGSLNode) == 48,
    "WGSLNode must be exactly 48 bytes.");
_Static_assert(_Alignof(WGSLNode) == 8,
    "WGSLNode must be 8-byte aligned.");
_Static_assert(WGSL_NODE_KIND_COUNT < (1 << 16),
    "WGSLNodeKind must fit in 16 bits.");

/* Iterative (recursion-free) AST-depth probe.  Returns the first node deeper
 * than `limit` (depth counted per AST node, root == depth 1), or NULL.
 *
 * Each downstream pass (resolver, const-eval, type-check, validate) calls
 * this at entry and refuses to run on a pathologically deep AST: those passes
 * contain many hand-rolled recursive walkers — only the hot ones route through
 * a depth-guarded chokepoint, so a left-leaning chain (`a+a+…`, `a.b.b…`) or
 * deep nesting could overflow one of the others.  Bailing once, up front, on
 * an over-deep tree bounds them all.  Uses a fixed, depth-bounded explicit
 * stack (no heap → no OOM, and the probe itself cannot overflow): a child
 * that would sit at depth limit+1 is reported without being pushed, so at
 * most `limit` frames are ever live.  Depth is counted the same way as the
 * ++depth>WGSL_MAX_AST_DEPTH chokepoint guards, so they all agree. */
static inline const WGSLNode *wgsl_ast_first_over_depth(
    const WGSLNode *root, int limit)
{
    if (!root || limit < 1) return NULL;
    struct { const WGSLNode *n; uint32_t next; } stack[WGSL_MAX_AST_DEPTH + 2];
    int sp = 0;
    stack[0].n = root;
    stack[0].next = 0;
    while (sp >= 0) {
        const WGSLNode *cur = stack[sp].n;
        if (stack[sp].next >= cur->child_count) { sp--; continue; }
        const WGSLNode *child = cur->children[stack[sp].next++];
        if (!child) continue;
        if (sp + 2 > limit) return child;   /* child would be at depth limit+1 */
        sp++;
        stack[sp].n = child;
        stack[sp].next = 0;
    }
    return NULL;
}

/*
 * payload[3] typed accessors — sanctioned decoders.
 *
 * Bit layout note: the parser remains the sole encoder (writes
 * `n->payload[i] = ...` directly).  Downstream passes must read through
 * these helpers so encoding changes stay one-touch.
 *
 * Slot ownership is pass-specific: the same payload slot can mean different
 * things after different compiler passes.  Accessors are therefore scoped by
 * kind and writer, not merely by kind.  The existing
 * `wgsl_node_resolved_symbol` (resolver.h / resolver.c) is the flag-gated
 * sibling for EXPR_IDENT payload[2] after resolve.
 *
 * Two attribute encodings (do not confuse them):
 *   - Decl attrs: leading children[0..count-1]; count lives in
 *     payload[1] (low).  Use wgsl_fn_attr_count / wgsl_param_attr_count /
 *     wgsl_member_attr_count / wgsl_varlike_attr_count.
 *   - Stmt attrs: side array pointed by payload[2]; count in payload[1]
 *     low; written by attach_statement_attrs.  Use wgsl_stmt_attr_*.
 */

/* Internal unpack helpers — not for call sites outside this header. */
static inline uint32_t wgsl_pld_lo32(uint64_t p) {
    return (uint32_t)(p & 0xFFFFFFFFu);
}
static inline uint32_t wgsl_pld_hi32(uint64_t p) {
    return (uint32_t)(p >> 32);
}
static inline WGSLSpan wgsl_pld_span(uint64_t p) {
    return wgsl_span_make(wgsl_pld_lo32(p), wgsl_pld_hi32(p));
}

/* Name span: payload[0] = offset | length<<32.
 * Kinds (parse-time, never overwritten by later passes for these):
 *   DECL_FUNCTION / PARAM / STRUCT / STRUCT_MEMBER / ALIAS /
 *   CONST / LET / OVERRIDE / VAR, EXPR_IDENT (name), ATTRIBUTE's
 *   IDENT child, EXPR_MEMBER's member IDENT child.
 * Not for: literals (payload[0]=value), binary/unary (op kind),
 *   case clause (sel_count|default), for (flags), directives. */
static inline WGSLSpan wgsl_node_name_span(const WGSLNode *n) {
    if (!n) return wgsl_span_make(0, 0);
    return wgsl_pld_span(n->payload[0]);
}

/* DECL_FUNCTION.
 * payload[1]: fn_attr_count | param_count<<32
 * payload[2]: ret_attr_count | has_return_type<<32
 * children:   [fn_attrs][params][ret_attrs][ret_type?][body] */
static inline uint32_t wgsl_fn_attr_count(const WGSLNode *fn) {
    return fn ? wgsl_pld_lo32(fn->payload[1]) : 0;
}
static inline uint32_t wgsl_fn_param_count(const WGSLNode *fn) {
    return fn ? wgsl_pld_hi32(fn->payload[1]) : 0;
}
static inline uint32_t wgsl_fn_ret_attr_count(const WGSLNode *fn) {
    return fn ? wgsl_pld_lo32(fn->payload[2]) : 0;
}
static inline uint32_t wgsl_fn_has_return_type(const WGSLNode *fn) {
    return fn ? wgsl_pld_hi32(fn->payload[2]) : 0;
}

/* DECL_PARAM.
 * payload[1]: attr_count (leading children).  children: [attrs][type] */
static inline uint32_t wgsl_param_attr_count(const WGSLNode *p) {
    return p ? wgsl_pld_lo32(p->payload[1]) : 0;
}

/* DECL_STRUCT_MEMBER.
 * payload[1]: attr_count (leading children).  children: [attrs][type] */
static inline uint32_t wgsl_member_attr_count(const WGSLNode *m) {
    return m ? wgsl_pld_lo32(m->payload[1]) : 0;
}

/* DECL_STRUCT.
 * payload[1]: member_count.  children: members[] */
static inline uint32_t wgsl_struct_member_count(const WGSLNode *s) {
    return s ? wgsl_pld_lo32(s->payload[1]) : 0;
}

/* DECL_VAR / DECL_OVERRIDE (shared "varlike" packing).
 * payload[1]: attr_count | tpl_count<<32   (override: tpl_count=0)
 * payload[2]: has_type   | has_init<<32
 * children:   [attrs][tpl_args…][type?][init?]
 * has_type/has_init are written as 0/1; decode with &1 / hi32 to match
 * historical sites that used either form. */
static inline uint32_t wgsl_varlike_attr_count(const WGSLNode *n) {
    return n ? wgsl_pld_lo32(n->payload[1]) : 0;
}
static inline uint32_t wgsl_varlike_tpl_count(const WGSLNode *n) {
    return n ? wgsl_pld_hi32(n->payload[1]) : 0;
}
static inline int wgsl_varlike_has_type(const WGSLNode *n) {
    return n ? (int)(n->payload[2] & 1u) : 0;
}
static inline int wgsl_varlike_has_init(const WGSLNode *n) {
    return n ? (int)wgsl_pld_hi32(n->payload[2]) : 0;
}

/* DECL_LET / DECL_CONST.
 * payload[1]: has_type (0/1).  No leading attrs.  children: [type?][init] */
static inline int wgsl_letlike_has_type(const WGSLNode *n) {
    return n ? (int)(n->payload[1] & 1u) : 0;
}

/* Statement attributes (written by attach_statement_attrs).
 * payload[1] low32 = attr_count (high32 preserved for other uses)
 * payload[2]       = (uintptr_t)WGSLNode** side array
 * Gated in practice by WGSL_FLAG_HAS_ATTRS.  NOT decl leading-children. */
static inline uint32_t wgsl_stmt_attr_count(const WGSLNode *n) {
    return n ? wgsl_pld_lo32(n->payload[1]) : 0;
}
static inline WGSLNode **wgsl_stmt_attrs(const WGSLNode *n) {
    if (!n) return NULL;
    return (WGSLNode **)(uintptr_t)n->payload[2];
}

/* Operator packing (binary / unary / compound_assign).
 * payload[0]: WGSLTokenKind of the operator
 * payload[1]: operator source span (offset | length<<32) */
static inline uint32_t wgsl_node_op_kind_u32(const WGSLNode *n) {
    return n ? (uint32_t)n->payload[0] : 0;
}
static inline WGSLSpan wgsl_node_op_span(const WGSLNode *n) {
    if (!n) return wgsl_span_make(0, 0);
    return wgsl_pld_span(n->payload[1]);
}

/* STMT_CASE_CLAUSE.
 * payload[0]: low32 = selector_count;
 *             bit 32 = is_default_alone (standalone `default:`);
 *             bit 33 = has_default_in_list (`case X, default, …`). */
static inline uint32_t wgsl_case_selector_count(const WGSLNode *n) {
    return n ? wgsl_pld_lo32(n->payload[0]) : 0;
}
static inline int wgsl_case_is_default_alone(const WGSLNode *n) {
    return n ? (int)((n->payload[0] >> 32) & 1u) : 0;
}
static inline int wgsl_case_has_default_in_list(const WGSLNode *n) {
    return n ? (int)((n->payload[0] >> 33) & 1u) : 0;
}
/* Full high half — tests historically compared >>32 to 0/1. Prefer the
 * bit accessors above for new code. */
static inline uint32_t wgsl_case_is_default(const WGSLNode *n) {
    return n ? wgsl_pld_hi32(n->payload[0]) : 0;
}

/* STMT_FOR.
 * payload[0]: flags (bit0=init, bit1=cond, bit2=update).  If HAS_ATTRS,
 * payload[1]/[2] hold stmt attrs; flags stay in payload[0]. */
static inline uint32_t wgsl_for_flags(const WGSLNode *n) {
    return n ? (uint32_t)n->payload[0] : 0;
}

/* Literals.
 * payload[0]: raw token payload (int/float bits) or 0/1 for bool. */
static inline uint64_t wgsl_lit_raw(const WGSLNode *n) {
    return n ? n->payload[0] : 0;
}
static inline int wgsl_lit_bool(const WGSLNode *n) {
    return n ? (n->payload[0] ? 1 : 0) : 0;
}

/* DIR_DIAGNOSTIC.
 * payload[0] low: rule-part count (1 = rule, 2 = rule.sub). */
static inline uint32_t wgsl_diag_rule_parts(const WGSLNode *n) {
    return n ? wgsl_pld_lo32(n->payload[0]) : 0;
}

/* DIR_ENABLE / DIR_REQUIRES.
 * payload[0]: extension / requirement name count (children = idents). */
static inline uint32_t wgsl_dir_name_count(const WGSLNode *n) {
    return n ? wgsl_pld_lo32(n->payload[0]) : 0;
}

#endif /* WGSL_INTERNAL_AST_H */
