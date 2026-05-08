/**
 * @file resolver.c — WGSL §5 name resolver + scope tree.
 *
 * Iter A — predeclared scope, scope ops, module-pass-1 (registration)
 *          and module-pass-2 (walking).
 * Iter B — function-body walk: parameters, blocks, control flow,
 *          identifier resolution.  Member-access RHS is deferred to
 *          the type checker (Phase 7); attribute argument idents are
 *          attempted but silent on miss.
 *
 * AST node carries the resolution result in `payload[2]` of every
 * `EXPR_IDENT` (and the leading ident of `EXPR_TEMPLATED_IDENT`).
 * `WGSL_FLAG_RESOLVED` is set on the same nodes for callers that want
 * a quick "did this get visited" probe.
 */
#include "internal/resolver.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Scope ops ──────────────────────────────────────────────────── */

#define SCOPE_INITIAL_CAP 16

static int scope_grow(WGSLScope *s) {
    size_t cap = s->capacity ? s->capacity * 2 : SCOPE_INITIAL_CAP;
    WGSLSymbol **g = (WGSLSymbol **)realloc(s->items, cap * sizeof(*g));
    if (!g) return 0;
    s->items    = g;
    s->capacity = cap;
    return 1;
}

static int scope_push_item(WGSLScope *s, WGSLSymbol *sym) {
    if (s->count == s->capacity && !scope_grow(s)) return 0;
    s->items[s->count++] = sym;
    return 1;
}

static WGSLSymbol *scope_find_local(
    const WGSLScope *s, const char *name, uint32_t name_len)
{
    for (size_t i = 0; i < s->count; i++) {
        WGSLSymbol *sym = s->items[i];
        if (sym->name_len == name_len &&
            memcmp(sym->name, name, name_len) == 0)
        {
            return sym;
        }
    }
    return NULL;
}

static WGSLSymbol *scope_find(
    const WGSLScope *s, const char *name, uint32_t name_len)
{
    for (const WGSLScope *cur = s; cur; cur = cur->parent) {
        WGSLSymbol *hit = scope_find_local(cur, name, name_len);
        if (hit) return hit;
    }
    return NULL;
}

static void scope_destroy(WGSLScope *s) {
    free(s->items);
    s->items = NULL; s->count = 0; s->capacity = 0;
}

/* ── Symbol construction ────────────────────────────────────────── */

static WGSLSymbol *make_sym(
    WGSLResolver *r,
    WGSLSymKind kind, const char *name, uint32_t name_len,
    WGSLNode *ast, WGSLTypeInfo *type)
{
    WGSLSymbol *s = (WGSLSymbol *)wgsl_arena_calloc(r->arena, 1, sizeof *s);
    if (!s) { r->had_error = 1; return NULL; }
    s->name     = name;
    s->name_len = name_len;
    s->kind     = (uint16_t)kind;
    s->ast      = ast;
    s->type     = type;
    return s;
}

/* ── Predeclared scope population ───────────────────────────────── */

static void add_pred_value(WGSLResolver *r, const char *name, WGSLSymKind kind) {
    WGSLSymbol *s = make_sym(r, kind, name, (uint32_t)strlen(name), NULL, NULL);
    if (s) scope_push_item(&r->predeclared, s);
}

static void add_pred_type(
    WGSLResolver *r, const char *name, WGSLSymKind kind, WGSLTypeInfo *t)
{
    WGSLSymbol *s = make_sym(r, kind, name, (uint32_t)strlen(name), NULL, t);
    if (s) scope_push_item(&r->predeclared, s);
}

static void populate_predeclared(WGSLResolver *r) {
    WGSLTypeStore *ts = r->types;

    /* Scalars (WGSL §6.2.2–5). */
    add_pred_type(r, "bool", WGSL_SYM_PREDECLARED_TYPE, ts->t_bool);
    add_pred_type(r, "i32",  WGSL_SYM_PREDECLARED_TYPE, ts->t_i32);
    add_pred_type(r, "u32",  WGSL_SYM_PREDECLARED_TYPE, ts->t_u32);
    add_pred_type(r, "f32",  WGSL_SYM_PREDECLARED_TYPE, ts->t_f32);
    add_pred_type(r, "f16",  WGSL_SYM_PREDECLARED_TYPE, ts->t_f16);

    /* Type generators (§6.2.6–9, §6.5).  No baked-in TypeInfo — they
     * resolve to a kind, the type checker materializes them with
     * template args. */
    static const char *typegens[] = {
        "vec2", "vec3", "vec4",
        "mat2x2", "mat2x3", "mat2x4",
        "mat3x2", "mat3x3", "mat3x4",
        "mat4x2", "mat4x3", "mat4x4",
        "atomic", "array", "ptr",
        /* Texture and sampler kinds (§6.5). */
        "texture_1d", "texture_2d", "texture_2d_array",
        "texture_3d", "texture_cube", "texture_cube_array",
        "texture_multisampled_2d",
        "texture_external",
        "texture_storage_1d", "texture_storage_2d",
        "texture_storage_2d_array", "texture_storage_3d",
        "texture_depth_2d", "texture_depth_2d_array",
        "texture_depth_cube", "texture_depth_cube_array",
        "texture_depth_multisampled_2d",
        "sampler", "sampler_comparison",
    };
    for (size_t i = 0; i < sizeof typegens / sizeof typegens[0]; i++) {
        add_pred_type(r, typegens[i], WGSL_SYM_PREDECLARED_TYPEGEN, NULL);
    }

    /* Predeclared type aliases (§6.9):
     *   vecN with elem in {i32, u32, f32, f16}                  → 12
     *   matCxR with C,R∈{2,3,4} and elem in {f32, f16}          → 18
     */
    static const struct { const char *name; uint8_t w; char k; } vec_aliases[] = {
        { "vec2i", 2, 'i' }, { "vec3i", 3, 'i' }, { "vec4i", 4, 'i' },
        { "vec2u", 2, 'u' }, { "vec3u", 3, 'u' }, { "vec4u", 4, 'u' },
        { "vec2f", 2, 'f' }, { "vec3f", 3, 'f' }, { "vec4f", 4, 'f' },
        { "vec2h", 2, 'h' }, { "vec3h", 3, 'h' }, { "vec4h", 4, 'h' },
    };
    for (size_t i = 0; i < sizeof vec_aliases / sizeof vec_aliases[0]; i++) {
        WGSLTypeInfo *e = NULL;
        switch (vec_aliases[i].k) {
        case 'i': e = ts->t_i32; break;
        case 'u': e = ts->t_u32; break;
        case 'f': e = ts->t_f32; break;
        case 'h': e = ts->t_f16; break;
        }
        WGSLTypeInfo *t = wgsl_type_vec(ts, vec_aliases[i].w, e);
        add_pred_type(r, vec_aliases[i].name, WGSL_SYM_PREDECLARED_TYPE_ALIAS, t);
    }
    for (uint8_t c = 2; c <= 4; c++) {
        for (uint8_t rr = 2; rr <= 4; rr++) {
            char namef[8], nameh[8];
            snprintf(namef, sizeof namef, "mat%ux%uf", c, rr);
            snprintf(nameh, sizeof nameh, "mat%ux%uh", c, rr);
            WGSLTypeInfo *tf = wgsl_type_mat(ts, c, rr, ts->t_f32);
            WGSLTypeInfo *th = wgsl_type_mat(ts, c, rr, ts->t_f16);
            /* names live in static storage on retainable strings — but
             * stack snprintf buffers expire.  Allocate from arena. */
            char *kf = (char *)wgsl_arena_alloc(r->arena, strlen(namef) + 1);
            char *kh = (char *)wgsl_arena_alloc(r->arena, strlen(nameh) + 1);
            if (kf) { strcpy(kf, namef); add_pred_type(r, kf, WGSL_SYM_PREDECLARED_TYPE_ALIAS, tf); }
            if (kh) { strcpy(kh, nameh); add_pred_type(r, kh, WGSL_SYM_PREDECLARED_TYPE_ALIAS, th); }
        }
    }

    /* Address spaces (§14.3). */
    static const char *addr_spaces[] = {
        "function", "private", "workgroup",
        "uniform", "storage", "handle", "push_constant",
    };
    for (size_t i = 0; i < sizeof addr_spaces / sizeof addr_spaces[0]; i++) {
        add_pred_value(r, addr_spaces[i], WGSL_SYM_PREDECLARED_ADDR_SPACE);
    }

    /* Access modes (§14.2). */
    static const char *access_modes[] = { "read", "write", "read_write" };
    for (size_t i = 0; i < 3; i++) {
        add_pred_value(r, access_modes[i], WGSL_SYM_PREDECLARED_ACCESS_MODE);
    }

    /* Builtin functions (§17).  Names only — Phase 7 will own
     * signatures via def/wgsl.def + codegen. */
    static const char *builtin_fns[] = {
        /* §17.2 Bit reinterpretation. */
        "bitcast",
        /* §17.3 Logical. */
        "all", "any", "select",
        /* §17.4 Array. */
        "arrayLength",
        /* §17.5 Numeric (63 fns). */
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
        "ceil", "clamp", "cos", "cosh",
        "countLeadingZeros", "countOneBits", "countTrailingZeros",
        "cross", "degrees", "determinant", "distance",
        "dot", "dot4U8Packed", "dot4I8Packed",
        "exp", "exp2", "extractBits",
        "faceForward", "firstLeadingBit", "firstTrailingBit",
        "floor", "fma", "fract", "frexp",
        "insertBits", "inverseSqrt", "ldexp", "length",
        "log", "log2", "max", "min", "mix", "modf",
        "normalize", "pow", "quantizeToF16", "radians",
        "reflect", "refract", "reverseBits", "round",
        "saturate", "sign", "sin", "sinh", "smoothstep",
        "sqrt", "step", "tan", "tanh", "transpose", "trunc",
        /* §17.6 Derivative. */
        "dpdx", "dpdy", "fwidth",
        "dpdxCoarse", "dpdyCoarse", "fwidthCoarse",
        "dpdxFine", "dpdyFine", "fwidthFine",
        /* §17.7 Texture (15). */
        "textureDimensions",
        "textureGather", "textureGatherCompare",
        "textureLoad",
        "textureNumLayers", "textureNumLevels", "textureNumSamples",
        "textureSample", "textureSampleBias",
        "textureSampleCompare", "textureSampleCompareLevel",
        "textureSampleGrad", "textureSampleLevel",
        "textureSampleBaseClampToEdge",
        "textureStore",
        /* §17.8 Atomic. */
        "atomicLoad", "atomicStore",
        "atomicAdd", "atomicSub",
        "atomicMax", "atomicMin",
        "atomicAnd", "atomicOr", "atomicXor",
        "atomicExchange", "atomicCompareExchangeWeak",
        /* §17.9–10 Pack / unpack. */
        "pack4x8snorm", "pack4x8unorm",
        "pack4xI8", "pack4xU8", "pack4xI8Clamp", "pack4xU8Clamp",
        "pack2x16snorm", "pack2x16unorm", "pack2x16float",
        "unpack4x8snorm", "unpack4x8unorm",
        "unpack4xI8", "unpack4xU8",
        "unpack2x16snorm", "unpack2x16unorm", "unpack2x16float",
        /* §17.11 Synchronization. */
        "storageBarrier", "textureBarrier",
        "workgroupBarrier", "workgroupUniformLoad",
        /* §17.12 Subgroup. */
        "subgroupAdd", "subgroupAll", "subgroupAnd", "subgroupAny",
        "subgroupBallot", "subgroupBroadcast", "subgroupBroadcastFirst",
        "subgroupElect",
        "subgroupExclusiveAdd", "subgroupExclusiveMul",
        "subgroupInclusiveAdd", "subgroupInclusiveMul",
        "subgroupMax", "subgroupMin", "subgroupMul",
        "subgroupOr", "subgroupShuffle",
        "subgroupShuffleDown", "subgroupShuffleUp", "subgroupShuffleXor",
        "subgroupXor",
        /* §17.13 Quad. */
        "quadBroadcast", "quadSwapDiagonal", "quadSwapX", "quadSwapY",
    };
    for (size_t i = 0; i < sizeof builtin_fns / sizeof builtin_fns[0]; i++) {
        add_pred_value(r, builtin_fns[i], WGSL_SYM_PREDECLARED_FN);
    }
}

/* ── Diagnostics ────────────────────────────────────────────────── */

static void resolve_error(
    WGSLResolver *r, const WGSLNode *n, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    wgsl_diag_emit_at(r->diag, r->src, WGSL_DIAG_ERROR,
                      n->span_offset, n->span_length, NULL, "%s", buf);
    r->had_error = 1;
}

/* ── AST → name extraction ──────────────────────────────────────── */

static void node_name(const WGSLResolver *r, const WGSLNode *n,
                      const char **out_ptr, uint32_t *out_len)
{
    uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
    uint32_t len = (uint32_t)(n->payload[0] >> 32);
    *out_ptr = r->src->bytes + off;
    *out_len = len;
}

/* Map an AST decl kind to the corresponding symbol kind. */
static WGSLSymKind sym_kind_for_decl(WGSLNodeKind k) {
    switch (k) {
    case WGSL_NODE_DECL_STRUCT:    return WGSL_SYM_STRUCT;
    case WGSL_NODE_DECL_FUNCTION:  return WGSL_SYM_FUNCTION;
    case WGSL_NODE_DECL_ALIAS:     return WGSL_SYM_TYPE_ALIAS;
    case WGSL_NODE_DECL_CONST:     return WGSL_SYM_CONST;
    case WGSL_NODE_DECL_OVERRIDE:  return WGSL_SYM_OVERRIDE;
    case WGSL_NODE_DECL_VAR:       return WGSL_SYM_VAR;
    case WGSL_NODE_DECL_LET:       return WGSL_SYM_LET;
    case WGSL_NODE_DECL_PARAM:     return WGSL_SYM_PARAM;
    default:                       return WGSL_SYM_INVALID;
    }
}

/* Bind a declaration's name into the given scope.  Emits "redeclaration"
 * if the same name already exists locally; emits "shadows predeclared"
 * if the predeclared scope has it (illegal per §5.1). */
static void bind_decl(
    WGSLResolver *r, WGSLScope *scope, WGSLNode *n)
{
    const char *name; uint32_t len;
    node_name(r, n, &name, &len);
    if (len == 0) return;

    /* Predeclared *type* names (scalar / typegen / alias / addr_space /
     * access_mode) live in the spec-reserved namespace and cannot be
     * shadowed at any scope.  Predeclared *function* names (`step`,
     * `length`, `dot`, …) are not reserved in practice — Tint/Naga
     * accept `let step = …` and so do real-world shaders.  Phase 8
     * (validator) owns the strict spec interpretation; the resolver
     * matches the de-facto behaviour. */
    WGSLSymbol *pred = scope_find_local(&r->predeclared, name, len);
    if (pred) {
        switch ((WGSLSymKind)pred->kind) {
        case WGSL_SYM_PREDECLARED_TYPE:
        case WGSL_SYM_PREDECLARED_TYPEGEN:
        case WGSL_SYM_PREDECLARED_TYPE_ALIAS:
        case WGSL_SYM_PREDECLARED_ADDR_SPACE:
        case WGSL_SYM_PREDECLARED_ACCESS_MODE:
            resolve_error(r, n,
                "'%.*s' shadows a predeclared identifier",
                (int)len, name);
            return;
        default:
            /* Predeclared fn / value — shadow is allowed locally. */
            break;
        }
    }
    if (scope_find_local(scope, name, len)) {
        resolve_error(r, n,
            "redeclaration of '%.*s' in the same scope",
            (int)len, name);
        return;
    }
    WGSLSymbol *s = make_sym(r, sym_kind_for_decl((WGSLNodeKind)n->kind),
                             name, len, n, NULL);
    if (!s) return;
    scope_push_item(scope, s);

    /* Mirror into the flat all-decls list for AST→symbol reverse
     * lookup by later passes. */
    if (r->all_decl_count == r->all_decl_capacity) {
        size_t cap = r->all_decl_capacity ? r->all_decl_capacity * 2 : 32;
        WGSLSymbol **g = (WGSLSymbol **)realloc(
            r->all_decls, cap * sizeof(*g));
        if (g) {
            r->all_decls = g;
            r->all_decl_capacity = cap;
        }
    }
    if (r->all_decl_count < r->all_decl_capacity) {
        r->all_decls[r->all_decl_count++] = s;
    }
}

/* ── Walker ─────────────────────────────────────────────────────── */

static void walk(WGSLResolver *r, WGSLNode *n);

static WGSLScope *push_scope(WGSLResolver *r) {
    WGSLScope *s = (WGSLScope *)wgsl_arena_calloc(r->arena, 1, sizeof *s);
    if (!s) { r->had_error = 1; return r->current; }
    s->parent  = r->current;
    r->current = s;
    return s;
}

static void pop_scope(WGSLResolver *r) {
    if (!r->current) return;
    WGSLScope *gone = r->current;
    r->current = gone->parent;
    free(gone->items);
    /* the scope struct itself lives in the arena */
}

static void store_resolution(WGSLNode *ident, WGSLSymbol *sym) {
    ident->payload[2] = (uint64_t)(uintptr_t)sym;
    ident->flags |= WGSL_FLAG_RESOLVED;
}

static void resolve_ident(WGSLResolver *r, WGSLNode *ident) {
    const char *name; uint32_t len;
    node_name(r, ident, &name, &len);
    if (len == 0) return;

    WGSLScope *start = r->current ? r->current : &r->module;
    WGSLSymbol *s = scope_find(start, name, len);
    if (!s) s = scope_find_local(&r->module, name, len);
    if (!s) s = scope_find_local(&r->predeclared, name, len);

    if (s) {
        store_resolution(ident, s);
        return;
    }

    /* Inside attribute arg lists, missing names are not errors here —
     * many enumerants (`vertex`, `position`, `linear`, …) are not in the
     * symbol scope; the validator owns attribute-arg semantics. */
    if (r->in_attribute) {
        ident->flags |= WGSL_FLAG_RESOLVED;
        return;
    }

    resolve_error(r, ident,
        "use of undeclared identifier '%.*s'",
        (int)len, name);
}

/* Walk the type-specifier sub-tree of a declaration: a single child
 * which is either an EXPR_IDENT or an EXPR_TEMPLATED_IDENT. */
static void walk_children(WGSLResolver *r, WGSLNode *n) {
    for (uint32_t i = 0; i < n->child_count; i++) {
        WGSLNode *c = n->children[i];
        if (c) walk(r, c);
    }
}

static void walk_function(WGSLResolver *r, WGSLNode *n) {
    /* Children: [fn_attrs...][params...][ret_attrs...][ret_type?][body] */
    uint32_t fn_attrs   = (uint32_t)(n->payload[1] & 0xFFFFFFFFu);
    uint32_t param_cnt  = (uint32_t)(n->payload[1] >> 32);
    uint32_t ret_attrs  = (uint32_t)(n->payload[2] & 0xFFFFFFFFu);
    uint32_t has_ret    = (uint32_t)(n->payload[2] >> 32);

    push_scope(r);              /* function scope */

    uint32_t i = 0;
    /* fn attrs first (resolved with in_attribute semantics). */
    for (uint32_t k = 0; k < fn_attrs && i < n->child_count; k++, i++) {
        walk(r, n->children[i]);
    }
    /* Params: walk type, then bind name into fn scope. */
    for (uint32_t k = 0; k < param_cnt && i < n->child_count; k++, i++) {
        WGSLNode *p = n->children[i];
        if (p && p->kind == WGSL_NODE_DECL_PARAM) {
            walk_children(r, p);          /* attrs + type */
            bind_decl(r, r->current, p);
        }
    }
    /* Return attrs. */
    for (uint32_t k = 0; k < ret_attrs && i < n->child_count; k++, i++) {
        walk(r, n->children[i]);
    }
    /* Return type. */
    if (has_ret && i < n->child_count) {
        walk(r, n->children[i++]);
    }
    /* Body — already a COMPOUND.  Walk its children directly so we don't
     * push a separate inner scope (per §11.1, params share fn scope with
     * the body). */
    if (i < n->child_count) {
        WGSLNode *body = n->children[i++];
        if (body && body->kind == WGSL_NODE_STMT_COMPOUND) {
            for (uint32_t k = 0; k < body->child_count; k++) {
                walk(r, body->children[k]);
            }
        } else if (body) {
            walk(r, body);
        }
    }
    pop_scope(r);
}

static void walk(WGSLResolver *r, WGSLNode *n) {
    if (!n) return;

    switch ((WGSLNodeKind)n->kind) {

    case WGSL_NODE_INVALID:
    case WGSL_NODE_TRANSLATION_UNIT:        /* not normally re-entered */
    case WGSL_NODE_DIR_ENABLE:
    case WGSL_NODE_DIR_REQUIRES:
    case WGSL_NODE_DIR_DIAGNOSTIC:
        return;

    case WGSL_NODE_DECL_FUNCTION:
        walk_function(r, n);
        return;

    case WGSL_NODE_DECL_PARAM:
        /* Top-level walk only — usually entered via walk_function. */
        walk_children(r, n);
        return;

    case WGSL_NODE_DECL_STRUCT:
        /* Members live in their own namespace; walk only their type-specs. */
        walk_children(r, n);
        return;

    case WGSL_NODE_DECL_STRUCT_MEMBER:
        walk_children(r, n);
        return;

    case WGSL_NODE_DECL_ALIAS:
        walk_children(r, n);
        return;

    case WGSL_NODE_DECL_CONST:
    case WGSL_NODE_DECL_LET:
    case WGSL_NODE_DECL_VAR:
        walk_children(r, n);
        if (r->current) bind_decl(r, r->current, n);
        return;

    case WGSL_NODE_DECL_OVERRIDE:
        walk_children(r, n);
        /* override is module-scope only (registered in pass 1). */
        return;

    case WGSL_NODE_DECL_CONST_ASSERT:
        walk_children(r, n);
        return;

    case WGSL_NODE_STMT_COMPOUND:
        push_scope(r);
        walk_children(r, n);
        pop_scope(r);
        return;

    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_WHILE:
        push_scope(r);
        walk_children(r, n);
        pop_scope(r);
        return;

    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_SWITCH:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_BREAK:
    case WGSL_NODE_STMT_BREAK_IF:
    case WGSL_NODE_STMT_CONTINUE:
    case WGSL_NODE_STMT_CONTINUING:
    case WGSL_NODE_STMT_RETURN:
    case WGSL_NODE_STMT_DISCARD:
    case WGSL_NODE_STMT_FN_CALL:
    case WGSL_NODE_STMT_ASSIGN:
    case WGSL_NODE_STMT_COMPOUND_ASSIGN:
    case WGSL_NODE_STMT_PHONY_ASSIGN:
    case WGSL_NODE_STMT_INCREMENT:
    case WGSL_NODE_STMT_DECREMENT:
        walk_children(r, n);
        return;

    case WGSL_NODE_EXPR_LITERAL_BOOL:
    case WGSL_NODE_EXPR_LITERAL_INT:
    case WGSL_NODE_EXPR_LITERAL_FLOAT:
        return;

    case WGSL_NODE_EXPR_IDENT:
        resolve_ident(r, n);
        return;

    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        if (n->child_count > 0) {
            WGSLNode *head = n->children[0];
            if (head && head->kind == WGSL_NODE_EXPR_IDENT) {
                resolve_ident(r, head);
            } else if (head) {
                walk(r, head);
            }
            for (uint32_t i = 1; i < n->child_count; i++) {
                walk(r, n->children[i]);
            }
        }
        return;

    case WGSL_NODE_EXPR_PAREN:
    case WGSL_NODE_EXPR_BINARY:
    case WGSL_NODE_EXPR_UNARY:
    case WGSL_NODE_EXPR_CALL:
    case WGSL_NODE_EXPR_INDEX:
    case WGSL_NODE_EXPR_ADDR_OF:
    case WGSL_NODE_EXPR_INDIRECTION:
    case WGSL_NODE_TEMPLATE_ARG_LIST:
        walk_children(r, n);
        return;

    case WGSL_NODE_EXPR_MEMBER:
        /* `obj.member` — resolve `obj`, leave `member` for type checker. */
        if (n->child_count >= 1 && n->children[0]) {
            walk(r, n->children[0]);
        }
        return;

    case WGSL_NODE_ATTRIBUTE: {
        int saved = r->in_attribute;
        r->in_attribute = 1;
        /* children[0] is the attribute name; skip it.  Walk the rest. */
        for (uint32_t i = 1; i < n->child_count; i++) {
            walk(r, n->children[i]);
        }
        r->in_attribute = saved;
        return;
    }

    case WGSL_NODE_KIND_COUNT:
        return;
    }
}

/* ── Module pre-pass: register all top-level names ──────────────── */

static void register_module_decls(WGSLResolver *r, WGSLNode *tu) {
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *d = tu->children[i];
        if (!d) continue;
        switch ((WGSLNodeKind)d->kind) {
        case WGSL_NODE_DECL_STRUCT:
        case WGSL_NODE_DECL_FUNCTION:
        case WGSL_NODE_DECL_ALIAS:
        case WGSL_NODE_DECL_CONST:
        case WGSL_NODE_DECL_OVERRIDE:
        case WGSL_NODE_DECL_VAR:
            bind_decl(r, &r->module, d);
            break;
        default:
            /* directives and const_asserts are walk-only */
            break;
        }
    }
}

/* ── Public entry ───────────────────────────────────────────────── */

int wgsl_resolve(
    WGSLAst         *ast,
    const WGSLSource *src,
    WGSLArena       *arena,
    WGSLDiagBag     *diag,
    WGSLTypeStore   *types,
    WGSLResolver    *out)
{
    if (!ast || !src || !arena || !diag || !types || !out) return 0;
    if (!ast->root || ast->root->kind != WGSL_NODE_TRANSLATION_UNIT) return 0;

    memset(out, 0, sizeof *out);
    out->arena   = arena;
    out->diag    = diag;
    out->types   = types;
    out->src     = src;
    out->current = NULL;

    populate_predeclared(out);

    WGSLNode *tu = ast->root;

    /* Pass 1 — register module-scope names. */
    register_module_decls(out, tu);

    /* Pass 2 — walk every top-level decl in source order.  Module-scope
     * walk does not push a new scope; the resolver consults
     * `out->module` directly when `current == NULL`. */
    for (uint32_t i = 0; i < tu->child_count; i++) {
        walk(out, tu->children[i]);
    }

    return out->had_error ? 0 : 1;
}

WGSLSymbol *wgsl_node_resolved_symbol(const WGSLNode *ident) {
    if (!ident) return NULL;
    if (!(ident->flags & WGSL_FLAG_RESOLVED)) return NULL;
    return (WGSLSymbol *)(uintptr_t)ident->payload[2];
}

void wgsl_resolver_destroy(WGSLResolver *r) {
    if (!r) return;
    /* Pop any remaining function/block scopes. */
    while (r->current) pop_scope(r);
    scope_destroy(&r->predeclared);
    scope_destroy(&r->module);
    free(r->all_decls);
    memset(r, 0, sizeof *r);
}
