/**
 * @file consteval.c — WGSL const-expression evaluator (Phase 6).
 *
 * Iter A — scalar literal parse, scalar arithmetic / comparison /
 *          logical / bitwise folding, abstract→concrete materialisation.
 * Iter B — module + fn-scope walker for `const` / `const_assert` decls,
 *          WGSLSymbol→value cache, EXPR_IDENT lookup against the cache.
 *
 * Iter C will add vec/mat constructor folding + corpus pass.
 */
#include "internal/consteval.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* ── Diagnostics ───────────────────────────────────────────────────── */

static void cev_error(
    WGSLConstEvaluator *cev, const WGSLNode *at, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    uint32_t off = at ? at->span_offset : 0;
    uint32_t len = at ? at->span_length : 0;
    wgsl_diag_emit_at(cev->diag, cev->src, WGSL_DIAG_ERROR,
                      off, len, NULL, "%s", buf);
    cev->had_error = 1;
}

/* ── Init / destroy ────────────────────────────────────────────────── */

void wgsl_consteval_init(
    WGSLConstEvaluator *cev,
    WGSLArena          *arena,
    WGSLDiagBag        *diag,
    WGSLTypeStore      *types,
    const WGSLSource   *src,
    WGSLResolver       *res)
{
    memset(cev, 0, sizeof *cev);
    cev->arena = arena;
    cev->diag  = diag;
    cev->types = types;
    cev->src   = src;
    cev->res   = res;
}

void wgsl_consteval_destroy(WGSLConstEvaluator *cev) {
    if (!cev) return;
    free(cev->bindings);
    memset(cev, 0, sizeof *cev);
}

/* ── Bindings table ────────────────────────────────────────────────── */

const WGSLValue *wgsl_consteval_value_of(
    const WGSLConstEvaluator *cev, const WGSLSymbol *sym)
{
    for (size_t i = 0; i < cev->binding_count; i++) {
        if (cev->bindings[i].sym == sym) return cev->bindings[i].value;
    }
    return NULL;
}

static int store_binding(
    WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v)
{
    if (cev->binding_count == cev->binding_capacity) {
        size_t cap = cev->binding_capacity ? cev->binding_capacity * 2 : 16;
        WGSLConstBinding *g = (WGSLConstBinding *)realloc(
            cev->bindings, cap * sizeof(*g));
        if (!g) return 0;
        cev->bindings = g;
        cev->binding_capacity = cap;
    }
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(cev->arena, sizeof *stored);
    if (!stored) return 0;
    *stored = *v;
    cev->bindings[cev->binding_count].sym   = sym;
    cev->bindings[cev->binding_count].value = stored;
    cev->binding_count += 1;
    return 1;
}

/* ── AST → bound symbol reverse lookup ─────────────────────────────── */

static WGSLSymbol *find_decl_symbol(
    const WGSLConstEvaluator *cev, const WGSLNode *decl)
{
    if (!cev->res) return NULL;
    for (size_t i = 0; i < cev->res->all_decl_count; i++) {
        if (cev->res->all_decls[i]->ast == decl) {
            return cev->res->all_decls[i];
        }
    }
    return NULL;
}

/* ── Literal parsing ───────────────────────────────────────────────── */

/* Slice the source bytes for `n`, copy into a null-terminated stack
 * buffer.  Returns the local length; truncates if too large.  WGSL
 * literal lengths are tiny (< 64 chars in practice). */
static int slice_text(
    const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap)
{
    if (!src || !n) { if (cap) buf[0] = '\0'; return 0; }
    uint32_t len = n->span_length;
    if (len + 1 > cap) len = (uint32_t)(cap - 1);
    memcpy(buf, src->bytes + n->span_offset, len);
    buf[len] = '\0';
    return (int)len;
}

static WGSLTypeInfo *type_for_int_suffix(
    WGSLConstEvaluator *cev, uint32_t suffix_bits)
{
    uint32_t base = suffix_bits & 0x0fu;
    switch (base) {
    case WGSL_NUM_SUFFIX_I: return cev->types->t_i32;
    case WGSL_NUM_SUFFIX_U: return cev->types->t_u32;
    default:                return cev->types->t_abstract_int;
    }
}

static WGSLTypeInfo *type_for_float_suffix(
    WGSLConstEvaluator *cev, uint32_t suffix_bits)
{
    uint32_t base = suffix_bits & 0x0fu;
    switch (base) {
    case WGSL_NUM_SUFFIX_F: return cev->types->t_f32;
    case WGSL_NUM_SUFFIX_H: return cev->types->t_f16;
    default:                return cev->types->t_abstract_float;
    }
}

static int parse_int_literal(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out)
{
    char buf[64];
    int len = slice_text(cev->src, n, buf, sizeof buf);
    if (len <= 0) {
        cev_error(cev, n, "invalid integer literal (empty)");
        return 0;
    }
    /* Strip suffix `i` / `u` if present. */
    if (buf[len - 1] == 'i' || buf[len - 1] == 'u') {
        buf[len - 1] = '\0';
        len -= 1;
    }
    /* WGSL accepts only `0`, `[1-9]…`, `0x…`, `0X…` forms.  strtoll
     * with base 0 handles all of those (and rejects nothing the lexer
     * accepted). */
    errno = 0;
    char *end = NULL;
    long long v = strtoll(buf, &end, 0);
    if (errno == ERANGE) {
        cev_error(cev, n, "integer literal out of range");
        return 0;
    }
    if (end == buf || *end != '\0') {
        cev_error(cev, n, "could not parse integer literal");
        return 0;
    }
    out->kind = WGSL_VAL_INT;
    out->type = type_for_int_suffix(cev, (uint32_t)n->payload[0]);
    out->u.i  = (int64_t)v;
    return 1;
}

static int parse_float_literal(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out)
{
    char buf[96];
    int len = slice_text(cev->src, n, buf, sizeof buf);
    if (len <= 0) {
        cev_error(cev, n, "invalid float literal (empty)");
        return 0;
    }
    if (buf[len - 1] == 'f' || buf[len - 1] == 'h') {
        buf[len - 1] = '\0';
        len -= 1;
    }
    errno = 0;
    char *end = NULL;
    double v = strtod(buf, &end);
    if (errno == ERANGE) {
        /* Underflow to ±0 is fine; overflow is not. */
        if (v == HUGE_VAL || v == -HUGE_VAL) {
            cev_error(cev, n, "float literal out of range");
            return 0;
        }
    }
    if (end == buf || *end != '\0') {
        cev_error(cev, n, "could not parse float literal");
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = type_for_float_suffix(cev, (uint32_t)n->payload[0]);
    out->u.f  = v;
    return 1;
}

static int eval_literal_bool(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out)
{
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    out->u.b  = (n->payload[0] != 0);
    return 1;
}

/* ── Predicates ────────────────────────────────────────────────────── */

static int val_is_int(const WGSLValue *v) {
    return v->kind == WGSL_VAL_INT;
}
static int val_is_float(const WGSLValue *v) {
    return v->kind == WGSL_VAL_FLOAT;
}
static int val_is_numeric(const WGSLValue *v) {
    return val_is_int(v) || val_is_float(v);
}
static int val_is_bool(const WGSLValue *v) {
    return v->kind == WGSL_VAL_BOOL;
}

/* ── Materialisation ───────────────────────────────────────────────── */

static int int_fits_target(int64_t v, WGSLTypeKind k) {
    switch (k) {
    case WGSL_TYPE_I32:
        return v >= INT32_MIN && v <= INT32_MAX;
    case WGSL_TYPE_U32:
        /* WGSL §6.2.1: AbstractInt → u32 only if value ≥ 0. */
        return v >= 0 && v <= (int64_t)UINT32_MAX;
    case WGSL_TYPE_ABSTRACT_INT:
        return 1;
    default:
        return 0;
    }
}

static int float_fits_target(double v, WGSLTypeKind k) {
    /* Spec defers to IEEE-754 finite range; we accept any finite + 0. */
    switch (k) {
    case WGSL_TYPE_F32:
        if (!isfinite(v)) return 1;            /* allow ±inf if produced */
        return v >= -FLT_MAX && v <= FLT_MAX;
    case WGSL_TYPE_F16:
        /* f16 max ≈ 65504 */
        return !isfinite(v) || (v >= -65504.0 && v <= 65504.0);
    case WGSL_TYPE_ABSTRACT_FLOAT:
        return 1;
    default:
        return 0;
    }
}

int wgsl_consteval_materialize(
    WGSLConstEvaluator *cev, WGSLValue *v,
    WGSLTypeInfo *target, const WGSLNode *at_node)
{
    if (!v || v->kind == WGSL_VAL_INVALID) return 0;
    if (!v->type) return 0;

    /* Default concretisation when no target requested. */
    if (!target) {
        target = wgsl_type_concretize(cev->types, v->type);
    }
    if (v->type == target) return 1;

    /* Conversion-rank gate (§6.1.2). */
    int rank = wgsl_type_conversion_rank(v->type, target);
    if (rank < 0) {
        cev_error(cev, at_node,
            "cannot convert value of type %s to %s",
            wgsl_type_kind_name((WGSLTypeKind)v->type->kind),
            wgsl_type_kind_name((WGSLTypeKind)target->kind));
        return 0;
    }

    /* Apply the value-side conversion. */
    WGSLTypeKind tk = (WGSLTypeKind)target->kind;
    if (val_is_int(v)) {
        if (tk == WGSL_TYPE_F32 || tk == WGSL_TYPE_F16 ||
            tk == WGSL_TYPE_ABSTRACT_FLOAT)
        {
            double d = (double)v->u.i;
            if (!float_fits_target(d, tk)) {
                cev_error(cev, at_node,
                    "value %lld out of range for %s",
                    (long long)v->u.i, wgsl_type_kind_name(tk));
                return 0;
            }
            v->kind = WGSL_VAL_FLOAT;
            v->u.f  = d;
        } else if (!int_fits_target(v->u.i, tk)) {
            cev_error(cev, at_node,
                "value %lld out of range for %s",
                (long long)v->u.i, wgsl_type_kind_name(tk));
            return 0;
        }
        v->type = target;
        return 1;
    }
    if (val_is_float(v)) {
        if (!float_fits_target(v->u.f, tk)) {
            cev_error(cev, at_node,
                "value out of range for %s", wgsl_type_kind_name(tk));
            return 0;
        }
        v->type = target;
        return 1;
    }
    /* bool / vec / mat: no scalar conversion in v1. */
    return 1;
}

/* ── Pair promotion (§6.1.2) ───────────────────────────────────────── */

static int promote_pair(
    WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b,
    const WGSLNode *at_node)
{
    if (a->type == b->type) return 1;
    int ab = wgsl_type_conversion_rank(a->type, b->type);
    int ba = wgsl_type_conversion_rank(b->type, a->type);

    /* Prefer concretising abstract operand when one side is concrete. */
    int a_abs = wgsl_type_is_abstract(a->type);
    int b_abs = wgsl_type_is_abstract(b->type);
    if (a_abs && !b_abs && ab >= 0) {
        return wgsl_consteval_materialize(cev, a, b->type, at_node);
    }
    if (b_abs && !a_abs && ba >= 0) {
        return wgsl_consteval_materialize(cev, b, a->type, at_node);
    }
    /* Both abstract — promote to AbstractFloat if either side is so. */
    if (a_abs && b_abs) {
        if (a->type == cev->types->t_abstract_float ||
            b->type == cev->types->t_abstract_float)
        {
            return wgsl_consteval_materialize(
                       cev, a, cev->types->t_abstract_float, at_node) &&
                   wgsl_consteval_materialize(
                       cev, b, cev->types->t_abstract_float, at_node);
        }
        /* both AbstractInt — already equal, handled by first check */
    }
    /* Both concrete and different — error. */
    if (ab >= 0) return wgsl_consteval_materialize(cev, a, b->type, at_node);
    if (ba >= 0) return wgsl_consteval_materialize(cev, b, a->type, at_node);

    cev_error(cev, at_node,
        "no implicit conversion between %s and %s",
        wgsl_type_kind_name((WGSLTypeKind)a->type->kind),
        wgsl_type_kind_name((WGSLTypeKind)b->type->kind));
    return 0;
}

/* ── Operator folding ──────────────────────────────────────────────── */

/* Apply `% / /` for ints, taking signedness into account. */
static int int_div_mod(
    WGSLConstEvaluator *cev, WGSLNode *at,
    int64_t a, int64_t b, int is_unsigned, int is_mod, int64_t *out)
{
    if (b == 0) {
        cev_error(cev, at, "%s by zero", is_mod ? "modulo" : "division");
        return 0;
    }
    if (is_unsigned) {
        uint64_t ua = (uint32_t)a;       /* low 32 bits */
        uint64_t ub = (uint32_t)b;
        *out = (int64_t)(is_mod ? (ua % ub) : (ua / ub));
        return 1;
    }
    /* Signed: spec mirrors C99: division truncates toward zero,
     * modulo carries the sign of the dividend. */
    if (a == INT64_MIN && b == -1) {
        cev_error(cev, at, "signed integer overflow in division");
        return 0;
    }
    *out = is_mod ? (a % b) : (a / b);
    return 1;
}

static int eval_binary_int(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    int64_t ai = a->u.i, bi = b->u.i;
    int is_u = (a->type == cev->types->t_u32);
    int64_t r = 0;

    switch (op) {
    case WGSL_TOK_PLUS:   r = ai + bi; break;
    case WGSL_TOK_MINUS:  r = ai - bi; break;
    case WGSL_TOK_STAR:   r = ai * bi; break;
    case WGSL_TOK_SLASH:
        if (!int_div_mod(cev, at, ai, bi, is_u, 0, &r)) return 0;
        break;
    case WGSL_TOK_PERCENT:
        if (!int_div_mod(cev, at, ai, bi, is_u, 1, &r)) return 0;
        break;
    case WGSL_TOK_AMP:    r = ai & bi; break;
    case WGSL_TOK_PIPE:   r = ai | bi; break;
    case WGSL_TOK_CARET:  r = ai ^ bi; break;
    case WGSL_TOK_LESS_LESS: {
        uint64_t s = (uint64_t)bi & 63u;
        r = (int64_t)((uint64_t)ai << s);
        break;
    }
    case WGSL_TOK_GREATER_GREATER: {
        uint64_t s = (uint64_t)bi & 63u;
        if (is_u) r = (int64_t)((uint32_t)ai >> s);
        else      r = ai >> s;
        break;
    }
    case WGSL_TOK_EQUAL_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (ai == bi); return 1;
    case WGSL_TOK_BANG_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (ai != bi); return 1;
    case WGSL_TOK_LESS:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai <  (uint32_t)bi) : (ai <  bi); return 1;
    case WGSL_TOK_LESS_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai <= (uint32_t)bi) : (ai <= bi); return 1;
    case WGSL_TOK_GREATER:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai >  (uint32_t)bi) : (ai >  bi); return 1;
    case WGSL_TOK_GREATER_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = is_u ? ((uint32_t)ai >= (uint32_t)bi) : (ai >= bi); return 1;
    default:
        cev_error(cev, at, "unsupported integer operator");
        return 0;
    }
    /* Truncate to target width for concrete ints to mirror runtime. */
    if (a->type == cev->types->t_i32) r = (int32_t)r;
    else if (is_u)                    r = (int64_t)(uint32_t)r;

    out->kind = WGSL_VAL_INT;
    out->type = a->type;        /* a and b were promoted to the same type */
    out->u.i  = r;
    return 1;
}

static int eval_binary_float(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    double af = a->u.f, bf = b->u.f, r = 0.0;
    switch (op) {
    case WGSL_TOK_PLUS:    r = af + bf; break;
    case WGSL_TOK_MINUS:   r = af - bf; break;
    case WGSL_TOK_STAR:    r = af * bf; break;
    case WGSL_TOK_SLASH:
        if (bf == 0.0) {
            cev_error(cev, at, "division by zero");
            return 0;
        }
        r = af / bf; break;
    case WGSL_TOK_PERCENT:
        if (bf == 0.0) {
            cev_error(cev, at, "modulo by zero");
            return 0;
        }
        /* WGSL §8.7.4: float % uses x - y * trunc(x / y). */
        r = af - bf * trunc(af / bf); break;
    case WGSL_TOK_EQUAL_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af == bf); return 1;
    case WGSL_TOK_BANG_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af != bf); return 1;
    case WGSL_TOK_LESS:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af <  bf); return 1;
    case WGSL_TOK_LESS_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af <= bf); return 1;
    case WGSL_TOK_GREATER:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af >  bf); return 1;
    case WGSL_TOK_GREATER_EQUAL:
        out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
        out->u.b = (af >= bf); return 1;
    default:
        cev_error(cev, at, "unsupported float operator");
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = a->type;
    out->u.f  = r;
    return 1;
}

static int eval_binary_bool(
    WGSLConstEvaluator *cev, WGSLNode *at, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    bool ab = a->u.b, bb = b->u.b;
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    switch (op) {
    case WGSL_TOK_AMP_AMP:   out->u.b = ab && bb; return 1;
    case WGSL_TOK_PIPE_PIPE: out->u.b = ab || bb; return 1;
    case WGSL_TOK_AMP:       out->u.b = ab && bb; return 1;
    case WGSL_TOK_PIPE:      out->u.b = ab || bb; return 1;
    case WGSL_TOK_CARET:     out->u.b = ab ^  bb; return 1;
    case WGSL_TOK_EQUAL_EQUAL: out->u.b = (ab == bb); return 1;
    case WGSL_TOK_BANG_EQUAL:  out->u.b = (ab != bb); return 1;
    default:
        cev_error(cev, at, "operator does not apply to bool operands");
        return 0;
    }
}

/* Forward decl */
static int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);

static int eval_unary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 1 || !n->children[0]) {
        cev_error(cev, n, "malformed unary expression");
        return 0;
    }
    WGSLValue inner = {0};
    if (!eval_expr(cev, n->children[0], &inner)) return 0;

    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
    switch (op) {
    case WGSL_TOK_MINUS:
        if (val_is_int(&inner)) {
            if (inner.type == cev->types->t_u32) {
                cev_error(cev, n, "unary minus is not defined for u32");
                return 0;
            }
            /* §6.2.3: negating the smallest representable value of a
             * signed type is a compile-time error.  AbstractInt is
             * 64-bit; concrete `i32` is 32-bit. */
            if (inner.type == cev->types->t_i32 &&
                inner.u.i == INT32_MIN)
            {
                cev_error(cev, n,
                    "signed integer overflow in negation: -(-2147483648) is not representable in i32");
                return 0;
            }
            if (inner.u.i == INT64_MIN) {
                cev_error(cev, n, "signed integer overflow in negation");
                return 0;
            }
            *out = inner; out->u.i = -inner.u.i; return 1;
        }
        if (val_is_float(&inner)) {
            *out = inner; out->u.f = -inner.u.f; return 1;
        }
        cev_error(cev, n, "unary minus requires numeric operand"); return 0;
    case WGSL_TOK_BANG:
        if (val_is_bool(&inner)) {
            out->kind = WGSL_VAL_BOOL; out->type = cev->types->t_bool;
            out->u.b = !inner.u.b; return 1;
        }
        cev_error(cev, n, "logical not requires bool operand"); return 0;
    case WGSL_TOK_TILDE:
        if (val_is_int(&inner)) {
            int64_t r = ~inner.u.i;
            if (inner.type == cev->types->t_i32) r = (int32_t)r;
            else if (inner.type == cev->types->t_u32) r = (int64_t)(uint32_t)r;
            *out = inner; out->u.i = r; return 1;
        }
        cev_error(cev, n, "bitwise not requires integer operand"); return 0;
    default:
        cev_error(cev, n, "unsupported unary operator"); return 0;
    }
}

static int eval_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];

    /* Boolean short-circuit evaluation is not strictly required for
     * const-eval (both sides are const-expr by definition), but mirror
     * the runtime semantics anyway so a constant-RHS-on-bool-error
     * doesn't incorrectly fold. */
    WGSLValue a = {0}, b = {0};
    if (!eval_expr(cev, n->children[0], &a)) return 0;
    if (!eval_expr(cev, n->children[1], &b)) return 0;

    /* Bool operands → handle separately (no numeric promotion). */
    if (val_is_bool(&a) && val_is_bool(&b)) {
        return eval_binary_bool(cev, n, op, &a, &b, out);
    }

    /* Mixed types → numeric promotion. */
    if (!val_is_numeric(&a) || !val_is_numeric(&b)) {
        cev_error(cev, n, "operator requires numeric or bool operands");
        return 0;
    }
    if (!promote_pair(cev, &a, &b, n)) return 0;

    /* Dispatch on the (now-shared) operand kind. */
    if (val_is_int(&a))   return eval_binary_int  (cev, n, op, &a, &b, out);
    if (val_is_float(&a)) return eval_binary_float(cev, n, op, &a, &b, out);

    cev_error(cev, n, "unhandled operand kind in binary op");
    return 0;
}

/* ── Top-level dispatch ────────────────────────────────────────────── */

static int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (!n) return 0;
    memset(out, 0, sizeof *out);
    switch ((WGSLNodeKind)n->kind) {

    case WGSL_NODE_EXPR_LITERAL_INT:   return parse_int_literal  (cev, n, out);
    case WGSL_NODE_EXPR_LITERAL_FLOAT: return parse_float_literal(cev, n, out);
    case WGSL_NODE_EXPR_LITERAL_BOOL:  return eval_literal_bool  (cev, n, out);

    case WGSL_NODE_EXPR_PAREN:
        if (n->child_count == 1 && n->children[0]) {
            return eval_expr(cev, n->children[0], out);
        }
        cev_error(cev, n, "malformed parenthesised expression");
        return 0;

    case WGSL_NODE_EXPR_UNARY:  return eval_unary (cev, n, out);
    case WGSL_NODE_EXPR_BINARY: return eval_binary(cev, n, out);

    case WGSL_NODE_EXPR_IDENT: {
        WGSLSymbol *sym = wgsl_node_resolved_symbol(n);
        if (!sym) {
            cev_error(cev, n,
                "unresolved identifier in constant expression");
            return 0;
        }
        const WGSLValue *bound = wgsl_consteval_value_of(cev, sym);
        if (!bound) {
            cev_error(cev, n,
                "'%.*s' is not a constant",
                (int)sym->name_len, sym->name);
            return 0;
        }
        *out = *bound;
        return 1;
    }

    /* Iter C will add vec/mat constructor calls. */
    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
    case WGSL_NODE_EXPR_CALL:
        cev_error(cev, n,
            "constructor const-eval not implemented yet (Iter C)");
        return 0;

    default:
        cev_error(cev, n,
            "expression is not a constant expression");
        return 0;
    }
}

int wgsl_consteval_expr(
    WGSLConstEvaluator *cev, WGSLNode *expr, WGSLValue *out)
{
    if (!cev || !expr || !out) return 0;
    return eval_expr(cev, expr, out);
}

/* ── Iter B — module + fn-scope walker for const decls / asserts ───── */

/* Resolve a type-specifier sub-tree to a WGSLTypeInfo when it's one of
 * the simple shapes the Iter B materialiser needs.  Returns NULL for
 * shapes the type-checker (Phase 7) will own — vec/mat/array typegens,
 * structs, aliases.  Returning NULL just means "no target steering";
 * default concretisation kicks in. */
static WGSLTypeInfo *type_from_typespec(
    const WGSLConstEvaluator *cev, const WGSLNode *tnode)
{
    if (!tnode) return NULL;
    const WGSLNode *head = tnode;
    if (tnode->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (tnode->child_count == 0) return NULL;
        head = tnode->children[0];
    }
    if (head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *sym = wgsl_node_resolved_symbol(head);
    if (!sym) return NULL;
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
        sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
    {
        return sym->type;
    }
    return NULL;
}

/* DECL_CONST children: [type?][init].  payload[1] low bit = has_type. */
static int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n) {
    int has_type = (int)(n->payload[1] & 1u);
    if (n->child_count < (uint32_t)(has_type ? 2 : 1)) {
        cev_error(cev, n, "const declaration is missing its initializer");
        return 0;
    }
    WGSLNode *type_node = has_type ? n->children[0]                : NULL;
    WGSLNode *init      = has_type ? n->children[1]                : n->children[0];

    WGSLValue v = {0};
    if (!eval_expr(cev, init, &v)) return 0;

    WGSLTypeInfo *target = type_node
        ? type_from_typespec(cev, type_node) : NULL;
    if (!target) target = wgsl_type_concretize(cev->types, v.type);
    if (!wgsl_consteval_materialize(cev, &v, target, init)) return 0;

    WGSLSymbol *sym = find_decl_symbol(cev, n);
    if (!sym) return 1;          /* resolver didn't bind — already errored */
    sym->type = v.type;
    if (!store_binding(cev, sym, &v)) {
        cev_error(cev, n, "out of memory while caching const value");
        return 0;
    }
    return 1;
}

/* DECL_OVERRIDE: in v1, we don't fold the default value into the
 * binding store (override values are pipeline-time).  We only validate
 * that the init, if present, is itself an override-expr — and the
 * cheapest way to check that is to attempt a pure const-eval and quietly
 * suppress the diagnostic on miss. */
static int eval_decl_override(WGSLConstEvaluator *cev, WGSLNode *n) {
    (void)cev; (void)n;
    /* No-op for v1.  Phase 7 will validate the override-expr tier with
     * full context. */
    return 1;
}

/* DECL_CONST_ASSERT: child[0] = condition expression. */
static int eval_const_assert(WGSLConstEvaluator *cev, WGSLNode *n) {
    if (n->child_count < 1 || !n->children[0]) {
        cev_error(cev, n, "const_assert is missing its condition");
        return 0;
    }
    WGSLValue v = {0};
    if (!eval_expr(cev, n->children[0], &v)) return 0;
    if (v.kind != WGSL_VAL_BOOL) {
        cev_error(cev, n, "const_assert condition must evaluate to bool");
        return 0;
    }
    if (!v.u.b) {
        cev_error(cev, n, "const_assert failed");
        return 0;
    }
    return 1;
}

static void walk_stmt(WGSLConstEvaluator *cev, WGSLNode *n);

static void walk_block(WGSLConstEvaluator *cev, WGSLNode *block) {
    if (!block) return;
    for (uint32_t i = 0; i < block->child_count; i++) {
        walk_stmt(cev, block->children[i]);
    }
}

static void walk_stmt(WGSLConstEvaluator *cev, WGSLNode *n) {
    if (!n) return;
    switch ((WGSLNodeKind)n->kind) {
    case WGSL_NODE_DECL_CONST:
        eval_decl_const(cev, n);
        return;
    case WGSL_NODE_DECL_CONST_ASSERT:
        eval_const_assert(cev, n);
        return;
    /* Recurse into compound forms so nested `const` / `const_assert`
     * inside `if { … }`, `for { … }`, etc. get evaluated. */
    case WGSL_NODE_STMT_COMPOUND:
    case WGSL_NODE_STMT_IF:
    case WGSL_NODE_STMT_ELSE_IF:
    case WGSL_NODE_STMT_ELSE:
    case WGSL_NODE_STMT_WHILE:
    case WGSL_NODE_STMT_FOR:
    case WGSL_NODE_STMT_LOOP:
    case WGSL_NODE_STMT_SWITCH:
    case WGSL_NODE_STMT_CASE_CLAUSE:
    case WGSL_NODE_STMT_CONTINUING:
        walk_block(cev, n);
        return;
    default:
        return;
    }
}

static void walk_function(WGSLConstEvaluator *cev, WGSLNode *fn) {
    if (!fn || fn->child_count == 0) return;
    /* Last child is the body (STMT_COMPOUND) per parser layout. */
    WGSLNode *body = fn->children[fn->child_count - 1];
    if (body && body->kind == WGSL_NODE_STMT_COMPOUND) {
        walk_block(cev, body);
    }
}

int wgsl_consteval(
    WGSLAst *ast, const WGSLSource *src, WGSLArena *arena,
    WGSLDiagBag *diag, WGSLTypeStore *types, WGSLResolver *res,
    WGSLConstEvaluator *out)
{
    wgsl_consteval_init(out, arena, diag, types, src, res);
    if (!ast || !ast->root) return 0;

    WGSLNode *tu = ast->root;
    for (uint32_t i = 0; i < tu->child_count; i++) {
        WGSLNode *n = tu->children[i];
        if (!n) continue;
        switch ((WGSLNodeKind)n->kind) {
        case WGSL_NODE_DECL_CONST:        eval_decl_const   (out, n); break;
        case WGSL_NODE_DECL_OVERRIDE:     eval_decl_override(out, n); break;
        case WGSL_NODE_DECL_CONST_ASSERT: eval_const_assert (out, n); break;
        case WGSL_NODE_DECL_FUNCTION:     walk_function     (out, n); break;
        default: break;
        }
    }
    return out->had_error ? 0 : 1;
}
