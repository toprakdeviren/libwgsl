/**
 * @file consteval.c — WGSL const-expression evaluator (Phase 6).
 *
 * Iter A — scalar literal parse, scalar arithmetic / comparison /
 *          logical / bitwise folding, abstract→concrete materialisation.
 * Iter B — module + fn-scope walker for `const` / `const_assert` decls,
 *          WGSLSymbol→value cache, EXPR_IDENT lookup against the cache.
 *
 * Iter C adds vector / matrix / array / struct constructor folding plus
 * broad @const numeric builtin fold-through for §15.7.7 domain checks.
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
        if (cev->binding_capacity > SIZE_MAX / 2) return 0;
        size_t cap = cev->binding_capacity ? cev->binding_capacity * 2 : 16;
        if (cap > SIZE_MAX / sizeof *cev->bindings) return 0;
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

/* WGSL §3.5.2: a decimal float literal preserves at most 20 significant
 * digits in its significand; a hex float literal preserves at most 16.
 * Beyond that, the spec lets the implementation pick between truncating
 * to zero and rounding up.  We pick truncation — the simpler and
 * cheaper choice — by replacing digits past the cutoff with '0'.
 *
 * Significance, per the spec: a digit is significant if it is non-zero,
 * or if there are non-zero significand digits both to its left and to
 * its right.  Equivalently, the run from the first to the last non-zero
 * significand digit is significant inclusive; everything outside that
 * run is not.  The decimal/hex point is skipped without resetting the
 * significant-digit count, and the exponent suffix (`e`/`E`/`p`/`P`)
 * marks the end of the significand. */
static void truncate_significand(char *buf, int len, int max_sig, int is_hex) {
    /* Locate the end of the significand portion. */
    char ec = is_hex ? 'p' : 'e';
    char EC = is_hex ? 'P' : 'E';
    int sig_end = len;
    for (int i = 0; i < len; i++) {
        if (buf[i] == ec || buf[i] == EC) { sig_end = i; break; }
    }
    int sig_start = (is_hex && len >= 2 &&
                     buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X')) ? 2 : 0;

    /* Find the first/last non-zero significand digits.  Hex digits a-f
     * and A-F are non-zero. */
    int first_nz = -1, last_nz = -1;
    for (int i = sig_start; i < sig_end; i++) {
        char c = buf[i];
        if (c == '.') continue;
        int nz = (c >= '1' && c <= '9') ||
                 (is_hex && ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')));
        if (nz) {
            if (first_nz < 0) first_nz = i;
            last_nz = i;
        }
    }
    if (first_nz < 0) return;  /* all-zero significand: nothing to do */

    /* Find the (max_sig + 1)-th significant digit and zero it (and every
     * subsequent significant position up through last_nz). */
    int count = 0;
    int cutoff = -1;
    for (int i = first_nz; i <= last_nz; i++) {
        if (buf[i] == '.') continue;
        count += 1;
        if (count == max_sig + 1) { cutoff = i; break; }
    }
    if (cutoff < 0) return;  /* significand has <= max_sig significant digits */
    for (int i = cutoff; i <= last_nz; i++) {
        if (buf[i] != '.') buf[i] = '0';
    }
}

static int parse_float_literal(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out)
{
    /* Mark the literal as visited up front so a later type-checker pass
     * over the same node skips its own consteval kick (which exists for
     * `let`/`var` paths that don't enter eval_decl_const).  The flag is
     * set regardless of success so a re-visit doesn't double-emit. */
    n->flags |= WGSL_FLAG_CONST_EVALED;

    /* Float literals can be arbitrarily long.  Spec §3.5.2 truncates
     * the significand to 20 (decimal) / 16 (hex) significant digits;
     * the buffer just needs to be large enough to hold a realistic
     * full-precision IEEE-encoded literal plus exponent + suffix.
     * 256 bytes is well past anything any real shader writes. */
    char buf[256];
    int len = slice_text(cev->src, n, buf, sizeof buf);
    if (len <= 0) {
        cev_error(cev, n, "invalid float literal (empty)");
        return 0;
    }
    uint32_t suffix = (uint32_t)n->payload[0] & 0x0fu;
    if (buf[len - 1] == 'f' || buf[len - 1] == 'h') {
        buf[len - 1] = '\0';
        len -= 1;
    }
    int is_hex = (len >= 2 && buf[0] == '0' &&
                  (buf[1] == 'x' || buf[1] == 'X'));
    truncate_significand(buf, len, is_hex ? 16 : 20, is_hex);
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
    /* §6.2.4 — per-suffix range check.  strtod's ERANGE branch above
     * only catches double-precision overflow (~1e308); a literal like
     * `1e100h` parses cleanly but exceeds the f16 representable range
     * (|v| ≤ 65504).  Same for `f` exceeding FLT_MAX. */
    if (suffix == WGSL_NUM_SUFFIX_F) {
        if (isfinite(v) && (v < -FLT_MAX || v > FLT_MAX)) {
            cev_error(cev, n, "value out of range for f32");
            return 0;
        }
    } else if (suffix == WGSL_NUM_SUFFIX_H) {
        /* IEEE 754 binary16 max is 65504. */
        if (isfinite(v) && (v < -65504.0 || v > 65504.0)) {
            cev_error(cev, n, "value out of range for f16");
            return 0;
        }
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

/* Forward decls needed by recursive aggregate materialization. */
static WGSLTypeInfo *type_from_typespec(
    const WGSLConstEvaluator *cev, const WGSLNode *tnode);
static int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
static int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n);
static uint32_t value_width(const WGSLValue *v);
static WGSLValue value_component(const WGSLValue *v, uint32_t i);
static int make_componentwise_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *vec_type,
    WGSLValue *elems, WGSLValue *out);

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
    if ((v->kind == WGSL_VAL_VEC && target->kind == WGSL_TYPE_VEC) ||
        (v->kind == WGSL_VAL_MAT && target->kind == WGSL_TYPE_MAT) ||
        (v->kind == WGSL_VAL_ARRAY && target->kind == WGSL_TYPE_ARRAY))
    {
        uint32_t expected = 0;
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        if (target->kind == WGSL_TYPE_VEC) {
            expected = target->width;
        } else if (target->kind == WGSL_TYPE_MAT) {
            expected = (uint32_t)target->width * (uint32_t)target->rows;
        } else {
            expected = target->array_len;
        }
        if (expected == 0 || v->u.agg.count != expected) {
            cev_error(cev, at_node, "aggregate constructor shape mismatch");
            return 0;
        }
        for (uint32_t i = 0; i < v->u.agg.count; i++) {
            if (!wgsl_consteval_materialize(
                    cev, &v->u.agg.elems[i], elem, at_node))
                return 0;
        }
        v->type = target;
        return 1;
    }

    if (v->kind == WGSL_VAL_STRUCT && target->kind == WGSL_TYPE_STRUCT) {
        WGSLNode *sd = (WGSLNode *)target->ref;
        if (!sd || v->type->ref != target->ref) {
            cev_error(cev, at_node, "struct constructor type mismatch");
            return 0;
        }
        uint32_t idx = 0;
        for (uint32_t i = 0; i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            if (idx >= v->u.agg.count) {
                cev_error(cev, at_node, "struct constructor shape mismatch");
                return 0;
            }
            uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            if (attrs >= m->child_count) return 0;
            WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
            if (mt && !wgsl_consteval_materialize(
                    cev, &v->u.agg.elems[idx], mt, at_node))
                return 0;
            idx++;
        }
        v->type = target;
        return idx == v->u.agg.count;
    }

    /* bool or non-convertible aggregate. */
    return target == v->type;
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

static int checked_abstract_int_result(
    WGSLConstEvaluator *cev, WGSLNode *at, __int128 value,
    const char *op_name, int64_t *out)
{
    if (value < (__int128)INT64_MIN || value > (__int128)INT64_MAX) {
        cev_error(cev, at,
            "signed integer overflow in %s for AbstractInt", op_name);
        return 0;
    }
    *out = (int64_t)value;
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
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            cev_error(cev, at, "signed integer overflow in division");
            return 0;
        }
        if (!int_div_mod(cev, at, ai, bi, is_u, 0, &r)) return 0;
        break;
    case WGSL_TOK_PERCENT:
        if (a->type == cev->types->t_i32 && ai == INT32_MIN && bi == -1) {
            cev_error(cev, at, "signed integer overflow in division");
            return 0;
        }
        if (!int_div_mod(cev, at, ai, bi, is_u, 1, &r)) return 0;
        break;
    case WGSL_TOK_AMP:    r = ai & bi; break;
    case WGSL_TOK_PIPE:   r = ai | bi; break;
    case WGSL_TOK_CARET:  r = ai ^ bi; break;
    case WGSL_TOK_LESS_LESS: {
        /* §8.9 — shift amount must be < bit-width of the target type.
         * For concrete i32 / u32 the width is 32; for AbstractInt the
         * spec allows the full 64-bit AbstractInt value but the result
         * must still fit when materialized.  We enforce the obvious
         * runtime case here (`1u << 33u` etc.). */
        uint8_t bit_width =
            (a->type == cev->types->t_i32 || a->type == cev->types->t_u32) ? 32 : 64;
        if (bi < 0 || (uint64_t)bi >= bit_width) {
            cev_error(cev, at,
                "shift amount %lld is out of range for %s",
                (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
            return 0;
        }
        uint64_t s = (uint64_t)bi;
        if (a->type == cev->types->t_u32) {
            uint64_t wr = (uint64_t)(uint32_t)ai << s;
            if (wr > UINT32_MAX) {
                cev_error(cev, at, "left shift result out of range for u32");
                return 0;
            }
            r = (int64_t)wr;
        } else {
            __int128 wr = (__int128)ai << s;
            __int128 lo = a->type == cev->types->t_i32
                ? (__int128)INT32_MIN : (__int128)INT64_MIN;
            __int128 hi = a->type == cev->types->t_i32
                ? (__int128)INT32_MAX : (__int128)INT64_MAX;
            if (wr < lo || wr > hi) {
                cev_error(cev, at, "left shift result out of range for %s",
                    wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
                return 0;
            }
            r = (int64_t)wr;
        }
        break;
    }
    case WGSL_TOK_GREATER_GREATER: {
        if (a->type == cev->types->t_abstract_int) {
            if (bi < 0) {
                cev_error(cev, at,
                    "shift amount %lld is out of range for AbstractInt",
                    (long long)bi);
                return 0;
            }
            if (bi >= 63) {
                r = ai < 0 ? -1 : 0;
            } else {
                r = ai >> (uint64_t)bi;
            }
            break;
        }
        uint8_t bit_width =
            (a->type == cev->types->t_i32 || a->type == cev->types->t_u32) ? 32 : 64;
        if (bi < 0 || (uint64_t)bi >= bit_width) {
            cev_error(cev, at,
                "shift amount %lld is out of range for %s",
                (long long)bi, wgsl_type_kind_name((WGSLTypeKind)a->type->kind));
            return 0;
        }
        uint64_t s = (uint64_t)bi;
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

static int round_float_value_to_type(
    WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out);

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
    /* §8.7 / §15.7.3 — finite-math assumption.  Inputs are finite at
     * this point (the consteval doesn't import non-finite literals);
     * any non-finite result therefore comes from a domain violation
     * (e.g., `inf - inf` derived through chained ops, or AbstractFloat
     * overflow at ~1e308 from `1e200 * 1e200`). */
    double rounded = 0.0;
    if (!round_float_value_to_type(cev, a->type, r, &rounded)) {
        if (isnan(r)) {
            cev_error(cev, at,
                "floating-point operation produced NaN "
                "(spec §15.7.3 finite-math assumption)");
        } else {
            cev_error(cev, at,
                "floating-point operation produced infinity "
                "(spec §15.7.3 finite-math assumption)");
        }
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = a->type;
    out->u.f  = rounded;
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
    case WGSL_TOK_EQUAL_EQUAL: out->u.b = (ab == bb); return 1;
    case WGSL_TOK_BANG_EQUAL:  out->u.b = (ab != bb); return 1;
    default:
        cev_error(cev, at, "operator does not apply to bool operands");
        return 0;
    }
}

static int eval_binary_componentwise(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (op == WGSL_TOK_AMP_AMP || op == WGSL_TOK_PIPE_PIPE) {
        cev_error(cev, n, "logical operator requires bool scalar operands");
        return 0;
    }

    uint32_t aw = value_width(a);
    uint32_t bw = value_width(b);
    uint32_t width = aw > bw ? aw : bw;
    if (aw != 1 && bw != 1 && aw != bw) {
        cev_error(cev, n, "vector operand width mismatch");
        return 0;
    }

    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue av = value_component(a, aw == 1 ? 0 : i);
        WGSLValue bv = value_component(b, bw == 1 ? 0 : i);
        if (val_is_bool(&av) || val_is_bool(&bv)) {
            if (!val_is_bool(&av) || !val_is_bool(&bv)) {
                cev_error(cev, n,
                    "operator does not apply to mixed bool and numeric operands");
                return 0;
            }
            if (!eval_binary_bool(cev, n, op, &av, &bv, &tmp[i])) return 0;
            continue;
        }
        if (!val_is_numeric(&av) || !val_is_numeric(&bv)) {
            cev_error(cev, n, "operator requires numeric or bool operands");
            return 0;
        }
        if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
            if (!val_is_int(&av) || !val_is_int(&bv)) {
                cev_error(cev, n, "shift requires integer operands");
                return 0;
            }
            if (bv.type != cev->types->t_u32 &&
                bv.type != cev->types->t_abstract_int)
            {
                cev_error(cev, n, "shift amount must be u32");
                return 0;
            }
            if (!eval_binary_int(cev, n, op, &av, &bv, &tmp[i])) return 0;
            continue;
        }
        if (!promote_pair(cev, &av, &bv, n)) return 0;
        if (val_is_int(&av)) {
            if (!eval_binary_int(cev, n, op, &av, &bv, &tmp[i])) return 0;
        } else if (val_is_float(&av)) {
            if (!eval_binary_float(cev, n, op, &av, &bv, &tmp[i])) return 0;
        } else {
            cev_error(cev, n, "unhandled operand kind in vector binary op");
            return 0;
        }
    }

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector binary op");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = tmp[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, tmp[0].type);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_numeric_scalar_binary(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, n)) return 0;
    if (val_is_int(a)) return eval_binary_int(cev, n, op, a, b, out);
    if (val_is_float(a)) return eval_binary_float(cev, n, op, a, b, out);
    cev_error(cev, n, "operator requires numeric operands");
    return 0;
}

static int make_matrix_binary_result(
    WGSLConstEvaluator *cev, WGSLNode *n,
    uint32_t cols, uint32_t rows, WGSLValue *elems, WGSLValue *out)
{
    if (cols < 2 || cols > 4 || rows < 2 || rows > 4) return 0;
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * cols * rows);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding matrix binary");
        return 0;
    }
    for (uint32_t i = 0; i < cols * rows; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(cev->types, (uint8_t)cols, (uint8_t)rows, elem);
    out->u.agg.count = cols * rows;
    out->u.agg.elems = stored;
    return 1;
}

static int make_vec_binary_result(
    WGSLConstEvaluator *cev, WGSLNode *n,
    uint32_t width, WGSLValue *elems, WGSLValue *out)
{
    if (width < 2 || width > 4) return 0;
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector binary");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_matrix_binary(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLTokenKind op,
    WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (op != WGSL_TOK_PLUS && op != WGSL_TOK_MINUS && op != WGSL_TOK_STAR)
        return 0;

    if (a->kind == WGSL_VAL_MAT && b->kind == WGSL_VAL_MAT) {
        if (!a->type || !b->type ||
            a->type->kind != WGSL_TYPE_MAT || b->type->kind != WGSL_TYPE_MAT)
            return 0;

        if (op == WGSL_TOK_PLUS || op == WGSL_TOK_MINUS) {
            if (a->type->width != b->type->width ||
                a->type->rows != b->type->rows)
                return 0;
            WGSLValue tmp[16] = {{0}};
            uint32_t count = a->type->width * a->type->rows;
            for (uint32_t i = 0; i < count; i++) {
                WGSLValue av = a->u.agg.elems[i];
                WGSLValue bv = b->u.agg.elems[i];
                if (!eval_numeric_scalar_binary(cev, n, op, &av, &bv, &tmp[i]))
                    return 0;
            }
            return make_matrix_binary_result(
                cev, n, a->type->width, a->type->rows, tmp, out);
        }

        /* mat<K,R> * mat<C,K> -> mat<C,R>. */
        if (a->type->width != b->type->rows) return 0;
        uint32_t K = a->type->width;
        uint32_t R = a->type->rows;
        uint32_t C = b->type->width;
        WGSLValue tmp[16] = {{0}};
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t r = 0; r < R; r++) {
                WGSLValue sum = {0};
                for (uint32_t k = 0; k < K; k++) {
                    WGSLValue av = a->u.agg.elems[k * R + r];
                    WGSLValue bv = b->u.agg.elems[c * K + k];
                    WGSLValue prod = {0};
                    if (!eval_numeric_scalar_binary(
                            cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                        return 0;
                    if (k == 0) {
                        sum = prod;
                    } else if (!eval_numeric_scalar_binary(
                                   cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                    {
                        return 0;
                    }
                }
                tmp[c * R + r] = sum;
            }
        }
        return make_matrix_binary_result(cev, n, C, R, tmp, out);
    }

    if ((a->kind == WGSL_VAL_MAT && val_is_numeric(b)) ||
        (b->kind == WGSL_VAL_MAT && val_is_numeric(a)))
    {
        WGSLValue *mat = a->kind == WGSL_VAL_MAT ? a : b;
        WGSLValue *scalar = a->kind == WGSL_VAL_MAT ? b : a;
        if (!mat->type || mat->type->kind != WGSL_TYPE_MAT) return 0;
        WGSLValue tmp[16] = {{0}};
        uint32_t count = mat->type->width * mat->type->rows;
        for (uint32_t i = 0; i < count; i++) {
            WGSLValue av = a->kind == WGSL_VAL_MAT
                ? mat->u.agg.elems[i] : *scalar;
            WGSLValue bv = a->kind == WGSL_VAL_MAT
                ? *scalar : mat->u.agg.elems[i];
            if (!eval_numeric_scalar_binary(cev, n, op, &av, &bv, &tmp[i]))
                return 0;
        }
        return make_matrix_binary_result(
            cev, n, mat->type->width, mat->type->rows, tmp, out);
    }

    if (op == WGSL_TOK_STAR &&
        a->kind == WGSL_VAL_MAT && b->kind == WGSL_VAL_VEC &&
        a->type && b->type &&
        a->type->kind == WGSL_TYPE_MAT && b->type->kind == WGSL_TYPE_VEC &&
        a->type->width == b->type->width)
    {
        uint32_t C = a->type->width;
        uint32_t R = a->type->rows;
        WGSLValue tmp[4] = {{0}};
        for (uint32_t r = 0; r < R; r++) {
            WGSLValue sum = {0};
            for (uint32_t c = 0; c < C; c++) {
                WGSLValue av = a->u.agg.elems[c * R + r];
                WGSLValue bv = b->u.agg.elems[c];
                WGSLValue prod = {0};
                if (!eval_numeric_scalar_binary(
                        cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                    return 0;
                if (c == 0) {
                    sum = prod;
                } else if (!eval_numeric_scalar_binary(
                               cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                {
                    return 0;
                }
            }
            tmp[r] = sum;
        }
        return make_vec_binary_result(cev, n, R, tmp, out);
    }

    if (op == WGSL_TOK_STAR &&
        a->kind == WGSL_VAL_VEC && b->kind == WGSL_VAL_MAT &&
        a->type && b->type &&
        a->type->kind == WGSL_TYPE_VEC && b->type->kind == WGSL_TYPE_MAT &&
        a->type->width == b->type->rows)
    {
        uint32_t C = b->type->width;
        uint32_t R = b->type->rows;
        WGSLValue tmp[4] = {{0}};
        for (uint32_t c = 0; c < C; c++) {
            WGSLValue sum = {0};
            for (uint32_t r = 0; r < R; r++) {
                WGSLValue av = a->u.agg.elems[r];
                WGSLValue bv = b->u.agg.elems[c * R + r];
                WGSLValue prod = {0};
                if (!eval_numeric_scalar_binary(
                        cev, n, WGSL_TOK_STAR, &av, &bv, &prod))
                    return 0;
                if (r == 0) {
                    sum = prod;
                } else if (!eval_numeric_scalar_binary(
                               cev, n, WGSL_TOK_PLUS, &sum, &prod, &sum))
                {
                    return 0;
                }
            }
            tmp[c] = sum;
        }
        return make_vec_binary_result(cev, n, C, tmp, out);
    }

    return 0;
}

static int eval_unary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *n,
    WGSLTokenKind op, WGSLValue inner, WGSLValue *out)
{
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

static int eval_unary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 1 || !n->children[0]) {
        cev_error(cev, n, "malformed unary expression");
        return 0;
    }
    WGSLValue inner = {0};
    if (!eval_expr(cev, n->children[0], &inner)) return 0;

    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];
    if (inner.kind == WGSL_VAL_VEC) {
        WGSLValue tmp[4] = {{0}};
        uint32_t width = inner.u.agg.count;
        for (uint32_t i = 0; i < width; i++) {
            if (!eval_unary_scalar(
                    cev, n, op, inner.u.agg.elems[i], &tmp[i]))
            {
                return 0;
            }
        }
        return make_componentwise_result(cev, n, width, inner.type, tmp, out);
    }
    return eval_unary_scalar(cev, n, op, inner, out);
}

static int eval_binary(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed binary expression");
        return 0;
    }
    WGSLTokenKind op = (WGSLTokenKind)n->payload[0];

    /* §8.1 — `&&` and `||` short-circuit at const-eval: when the LHS
     * already determines the result, the RHS must not be evaluated.
     * This matters when the RHS would otherwise trigger a const-eval
     * diagnostic (`false && (5/0 == 0)` must not flag div-by-zero). */
    WGSLValue a = {0}, b = {0};
    if (!eval_expr(cev, n->children[0], &a)) return 0;
    if (val_is_bool(&a)) {
        if (op == WGSL_TOK_AMP_AMP && a.u.b == false) {
            out->kind = WGSL_VAL_BOOL;
            out->type = cev->types->t_bool;
            out->u.b  = false;
            return 1;
        }
        if (op == WGSL_TOK_PIPE_PIPE && a.u.b == true) {
            out->kind = WGSL_VAL_BOOL;
            out->type = cev->types->t_bool;
            out->u.b  = true;
            return 1;
        }
    }
    if (!eval_expr(cev, n->children[1], &b)) return 0;

    if (a.kind == WGSL_VAL_MAT || b.kind == WGSL_VAL_MAT) {
        return eval_matrix_binary(cev, n, op, &a, &b, out);
    }

    if (a.kind == WGSL_VAL_VEC || b.kind == WGSL_VAL_VEC) {
        return eval_binary_componentwise(cev, n, op, &a, &b, out);
    }

    /* Bool operands → handle separately (no numeric promotion). */
    if (val_is_bool(&a) && val_is_bool(&b)) {
        return eval_binary_bool(cev, n, op, &a, &b, out);
    }

    /* Mixed types → numeric promotion. */
    if (!val_is_numeric(&a) || !val_is_numeric(&b)) {
        cev_error(cev, n, "operator requires numeric or bool operands");
        return 0;
    }
    if (op == WGSL_TOK_LESS_LESS || op == WGSL_TOK_GREATER_GREATER) {
        if (!val_is_int(&a) || !val_is_int(&b)) {
            cev_error(cev, n, "shift requires integer operands");
            return 0;
        }
        if (b.type != cev->types->t_u32 &&
            b.type != cev->types->t_abstract_int)
        {
            cev_error(cev, n, "shift amount must be u32");
            return 0;
        }
        return eval_binary_int(cev, n, op, &a, &b, out);
    }
    if (!promote_pair(cev, &a, &b, n)) return 0;

    /* Dispatch on the (now-shared) operand kind. */
    if (val_is_int(&a))   return eval_binary_int  (cev, n, op, &a, &b, out);
    if (val_is_float(&a)) return eval_binary_float(cev, n, op, &a, &b, out);

    cev_error(cev, n, "unhandled operand kind in binary op");
    return 0;
}

/* ── Builtin call folding (§15.7.7 domain checks) ─────────────────── */

static int node_ident_text(
    const WGSLConstEvaluator *cev, const WGSLNode *n,
    const char **out, uint32_t *len)
{
    if (!n || n->kind != WGSL_NODE_EXPR_IDENT) return 0;
    uint32_t off = (uint32_t)(n->payload[0] & 0xFFFFFFFFu);
    uint32_t l   = (uint32_t)(n->payload[0] >> 32);
    if (!cev->src || off + l > cev->src->length) return 0;
    *out = cev->src->bytes + off;
    *len = l;
    return 1;
}

static int name_eq(const char *nm, uint32_t len, const char *want) {
    size_t wl = strlen(want);
    return len == wl && memcmp(nm, want, wl) == 0;
}

static double round_even(double x) {
    double lo = floor(x);
    double f = x - lo;
    if (f < 0.5) return lo;
    if (f > 0.5) return lo + 1.0;
    return fmod(lo, 2.0) == 0.0 ? lo : lo + 1.0;
}

static int round_float_value_to_type(
    WGSLConstEvaluator *cev, WGSLTypeInfo *type, double x, double *out)
{
    (void)cev;
    if (!isfinite(x)) return 0;
    if (type == cev->types->t_f32) {
        float xf = (float)x;
        if (!isfinite((double)xf)) return 0;
        *out = (double)xf;
        return 1;
    }
    if (type == cev->types->t_f16) {
        double ax = fabs(x);
        if (ax == 0.0) {
            *out = signbit(x) ? -0.0 : 0.0;
            return 1;
        }
        /* binary16 overflows past the midpoint between max finite
         * (65504) and +inf; values at/below that round to max finite. */
        if (ax > 65520.0) return 0;
        double rounded = 0.0;
        if (ax < ldexp(1.0, -14)) {
            double step = ldexp(1.0, -24);
            rounded = round_even(ax / step) * step;
        } else {
            int e = (int)floor(log2(ax));
            double step = ldexp(1.0, e - 10);
            rounded = round_even(ax / step) * step;
            if (rounded > 65504.0) rounded = 65504.0;
        }
        *out = signbit(x) ? -rounded : rounded;
        return 1;
    }
    *out = x;
    return 1;
}

static int finish_float_builtin_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLTypeInfo *result_type, double r, WGSLValue *out)
{
    if (!isfinite(r)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    if (result_type == cev->types->t_f32 &&
        (r < -FLT_MAX || r > FLT_MAX))
    {
        cev_error(cev, call, "builtin '%.*s' result out of range for f32",
                  (int)len, nm);
        return 0;
    }
    if (result_type == cev->types->t_f16 &&
        (r < -65504.0 || r > 65504.0))
    {
        cev_error(cev, call, "builtin '%.*s' result out of range for f16",
                  (int)len, nm);
        return 0;
    }

    out->kind = WGSL_VAL_FLOAT;
    out->type = result_type;
    out->u.f = r;
    return 1;
}

static int finish_float_builtin_result_relaxed_range(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLTypeInfo *result_type, double r, WGSLValue *out)
{
    if (!isfinite(r)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    out->kind = WGSL_VAL_FLOAT;
    out->type = result_type;
    out->u.f = r;
    return 1;
}

static int require_float_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *v)
{
    if (v->kind == WGSL_VAL_INT && v->type == cev->types->t_abstract_int) {
        return wgsl_consteval_materialize(
            cev, v, cev->types->t_abstract_float, call);
    }
    if (v->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, call,
            "'%.*s' requires a floating-point argument", (int)len, nm);
        return 0;
    }
    return 1;
}

static int promote_three(
    WGSLConstEvaluator *cev, WGSLValue *a, WGSLValue *b, WGSLValue *c,
    const WGSLNode *at)
{
    return promote_pair(cev, a, b, at) &&
           promote_pair(cev, a, c, at) &&
           promote_pair(cev, b, c, at);
}

static uint32_t value_width(const WGSLValue *v) {
    return v->kind == WGSL_VAL_VEC ? v->u.agg.count : 1;
}

static WGSLValue value_component(const WGSLValue *v, uint32_t i) {
    return v->kind == WGSL_VAL_VEC ? v->u.agg.elems[i] : *v;
}

static int componentwise_shape(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *vals, int argc, int scalar_broadcast_arg,
    uint32_t *out_width, WGSLTypeInfo **out_vec_type)
{
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    for (int i = 0; i < argc; i++) {
        if (vals[i].kind != WGSL_VAL_VEC) continue;
        if (!vals[i].type || vals[i].type->kind != WGSL_TYPE_VEC) {
            cev_error(cev, call,
                "builtin '%.*s' requires scalar/vector numeric arguments",
                (int)len, nm);
            return 0;
        }
        if (!vec_type) {
            vec_type = vals[i].type;
            width = vals[i].u.agg.count;
        } else if (vals[i].u.agg.count != width) {
            cev_error(cev, call,
                "builtin '%.*s' vector width mismatch in const-eval",
                (int)len, nm);
            return 0;
        }
    }
    if (vec_type) {
        for (int i = 0; i < argc; i++) {
            if (vals[i].kind == WGSL_VAL_VEC) continue;
            if (i == scalar_broadcast_arg) continue;
            cev_error(cev, call,
                "builtin '%.*s' shape mismatch in const-eval",
                (int)len, nm);
            return 0;
        }
    }
    *out_width = width;
    *out_vec_type = vec_type;
    return 1;
}

static int make_componentwise_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *vec_type,
    WGSLValue *elems, WGSLValue *out)
{
    if (!vec_type) {
        *out = elems[0];
        return 1;
    }
    WGSLTypeInfo *elem = elems[0].type;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector builtin");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int make_vector_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *elem_type,
    WGSLValue *elems, WGSLValue *out)
{
    if (width < 2 || width > 4 || !elem_type) return 0;
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector builtin");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, (uint8_t)width, elem_type);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int require_same_width_vectors(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *vals, uint32_t argc, uint32_t *out_width)
{
    if (argc == 0 || vals[0].kind != WGSL_VAL_VEC) {
        cev_error(cev, call,
            "builtin '%.*s' requires vector arguments", (int)len, nm);
        return 0;
    }
    uint32_t width = vals[0].u.agg.count;
    for (uint32_t i = 1; i < argc; i++) {
        if (vals[i].kind != WGSL_VAL_VEC || vals[i].u.agg.count != width) {
            cev_error(cev, call,
                "builtin '%.*s' requires same-width vector arguments",
                (int)len, nm);
            return 0;
        }
    }
    *out_width = width;
    return 1;
}

static int vector_width_from_name(const char *nm, uint32_t len) {
    if (len == 4 && memcmp(nm, "vec", 3) == 0 &&
        nm[3] >= '2' && nm[3] <= '4')
    {
        return nm[3] - '0';
    }
    return 0;
}

static int matrix_dims_from_name(
    const char *nm, uint32_t len, uint8_t *cols, uint8_t *rows)
{
    if (len == 6 && memcmp(nm, "mat", 3) == 0 && nm[4] == 'x' &&
        nm[3] >= '2' && nm[3] <= '4' &&
        nm[5] >= '2' && nm[5] <= '4')
    {
        *cols = (uint8_t)(nm[3] - '0');
        *rows = (uint8_t)(nm[5] - '0');
        return 1;
    }
    return 0;
}

static WGSLSymbol *callee_head_symbol(const WGSLNode *callee) {
    if (!callee) return NULL;
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        return wgsl_node_resolved_symbol((WGSLNode *)callee);
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
        callee->child_count >= 1 && callee->children[0])
    {
        return wgsl_node_resolved_symbol(callee->children[0]);
    }
    return NULL;
}

static uint32_t struct_member_count(const WGSLNode *sd) {
    uint32_t count = 0;
    if (!sd) return 0;
    for (uint32_t i = 0; i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (m && m->kind == WGSL_NODE_DECL_STRUCT_MEMBER) count++;
    }
    return count;
}

static int make_zero_value(
    WGSLConstEvaluator *cev, WGSLTypeInfo *target, const WGSLNode *at,
    WGSLValue *out)
{
    if (!target) return 0;
    memset(out, 0, sizeof *out);
    out->type = target;
    switch ((WGSLTypeKind)target->kind) {
    case WGSL_TYPE_BOOL:
        out->kind = WGSL_VAL_BOOL;
        out->u.b = false;
        return 1;
    case WGSL_TYPE_ABSTRACT_INT:
    case WGSL_TYPE_I32:
    case WGSL_TYPE_U32:
        out->kind = WGSL_VAL_INT;
        out->u.i = 0;
        return 1;
    case WGSL_TYPE_ABSTRACT_FLOAT:
    case WGSL_TYPE_F32:
    case WGSL_TYPE_F16:
        out->kind = WGSL_VAL_FLOAT;
        out->u.f = 0.0;
        return 1;
    case WGSL_TYPE_VEC:
    case WGSL_TYPE_MAT:
    case WGSL_TYPE_ARRAY: {
        uint32_t count = 0;
        if (target->kind == WGSL_TYPE_VEC) {
            count = target->width;
            out->kind = WGSL_VAL_VEC;
        } else if (target->kind == WGSL_TYPE_MAT) {
            count = (uint32_t)target->width * (uint32_t)target->rows;
            out->kind = WGSL_VAL_MAT;
        } else {
            count = target->array_len;
            out->kind = WGSL_VAL_ARRAY;
            if (count == 0) {
                cev_error(cev, at, "runtime-sized array is not constructible");
                return 0;
            }
        }
        WGSLValue *elems = NULL;
        if (count > 0) {
            elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * count);
            if (!elems) {
                cev_error(cev, at, "out of memory while folding constructor");
                return 0;
            }
        }
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        for (uint32_t i = 0; i < count; i++) {
            if (!make_zero_value(cev, elem, at, &elems[i])) return 0;
        }
        out->u.agg.count = count;
        out->u.agg.elems = elems;
        return 1;
    }
    case WGSL_TYPE_STRUCT: {
        WGSLNode *sd = (WGSLNode *)target->ref;
        if (sd && (sd->flags & WGSL_FLAG_CONST_EVALING)) {
            cev_error(cev, at, "cyclic dependency detected in struct constructor");
            return 0;
        }
        if (sd) sd->flags |= WGSL_FLAG_CONST_EVALING;
        uint32_t count = struct_member_count(sd);
        WGSLValue *elems = NULL;
        if (count > 0) {
            elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * count);
            if (!elems) {
                if (sd) sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                cev_error(cev, at, "out of memory while folding struct constructor");
                return 0;
            }
        }
        uint32_t idx = 0;
        for (uint32_t i = 0; sd && i < sd->child_count; i++) {
            WGSLNode *m = sd->children[i];
            if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
            uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
            if (attrs >= m->child_count) {
                sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                return 0;
            }
            WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
            if (!mt || !make_zero_value(cev, mt, at, &elems[idx])) {
                sd->flags &= ~WGSL_FLAG_CONST_EVALING;
                return 0;
            }
            idx++;
        }
        if (sd) sd->flags &= ~WGSL_FLAG_CONST_EVALING;
        out->kind = WGSL_VAL_STRUCT;
        out->type = target;
        out->u.agg.count = count;
        out->u.agg.elems = elems;
        return 1;
    }
    default:
        cev_error(cev, at, "type is not constructible");
        return 0;
    }
}

static WGSLTypeInfo *constructor_target_type(
    const WGSLConstEvaluator *cev, const WGSLNode *callee)
{
    if (!callee) return NULL;
    WGSLSymbol *sym = callee_head_symbol(callee);
    if (callee->kind == WGSL_NODE_EXPR_IDENT) {
        if (sym && (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
                    sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS ||
                    sym->kind == WGSL_SYM_TYPE_ALIAS ||
                    sym->kind == WGSL_SYM_STRUCT))
        {
            return type_from_typespec(cev, callee);
        }
        return NULL;
    }
    if (callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        return type_from_typespec(cev, callee);
    }
    return NULL;
}

static int convert_scalar_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target,
    const WGSLNode *at);

static int convert_constructor_component(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *elem,
    const WGSLNode *at)
{
    if (!v || !elem) return 0;
    if (v->kind == WGSL_VAL_BOOL ||
        v->kind == WGSL_VAL_INT ||
        v->kind == WGSL_VAL_FLOAT)
    {
        return convert_scalar_constructor_value(cev, v, elem, at);
    }
    if (v->kind != WGSL_VAL_INT && v->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, at, "constructor requires scalar components");
        return 0;
    }
    return wgsl_consteval_materialize(cev, v, elem, at);
}

static int append_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t width,
    uint32_t *count, WGSLTypeInfo *elem, WGSLValue *src, const WGSLNode *at)
{
    if (src->kind == WGSL_VAL_VEC) {
        for (uint32_t i = 0; i < src->u.agg.count; i++) {
            if (*count >= width) {
                cev_error(cev, at, "too many vector constructor components");
                return 0;
            }
            WGSLValue c = src->u.agg.elems[i];
            if (!convert_constructor_component(cev, &c, elem, at)) return 0;
            dst[(*count)++] = c;
        }
        return 1;
    }
    if (*count >= width) {
        cev_error(cev, at, "too many vector constructor components");
        return 0;
    }
    WGSLValue c = *src;
    if (!convert_constructor_component(cev, &c, elem, at)) return 0;
    dst[(*count)++] = c;
    return 1;
}

static int common_constructor_type(
    WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count,
    const WGSLNode *at, WGSLTypeInfo **out)
{
    if (count == 0) {
        *out = cev->types->t_abstract_int;
        return 1;
    }
    WGSLTypeInfo *common = vals[0].type;
    if (!common) return 0;
    for (uint32_t i = 1; i < count; i++) {
        WGSLTypeInfo *t = vals[i].type;
        if (!t) return 0;
        if (common == t) continue;
        int ab = wgsl_type_conversion_rank(common, t);
        int ba = wgsl_type_conversion_rank(t, common);
        if (ab >= 0 && (ba < 0 || ab <= ba)) {
            common = t;
        } else if (ba >= 0) {
            /* Keep the existing common type. */
        } else {
            cev_error(cev, at, "constructor arguments have incompatible types");
            return 0;
        }
    }
    *out = common;
    return 1;
}

static int materialize_constructor_values(
    WGSLConstEvaluator *cev, WGSLValue *vals, uint32_t count,
    WGSLTypeInfo *target, const WGSLNode *at)
{
    for (uint32_t i = 0; i < count; i++) {
        if (!wgsl_consteval_materialize(cev, &vals[i], target, at))
            return 0;
    }
    return 1;
}

static int convert_scalar_constructor_value(
    WGSLConstEvaluator *cev, WGSLValue *v, WGSLTypeInfo *target,
    const WGSLNode *at)
{
    if (!v || !target) return 0;
    WGSLTypeKind tk = (WGSLTypeKind)target->kind;
    if (tk == WGSL_TYPE_BOOL) {
        bool b = false;
        if (v->kind == WGSL_VAL_BOOL) {
            b = v->u.b;
        } else if (v->kind == WGSL_VAL_INT) {
            b = v->u.i != 0;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            b = v->u.f != 0.0;
        } else {
            cev_error(cev, at, "cannot convert value to bool");
            return 0;
        }
        v->kind = WGSL_VAL_BOOL;
        v->type = target;
        v->u.b = b;
        return 1;
    }

    if (tk == WGSL_TYPE_I32 || tk == WGSL_TYPE_U32 ||
        tk == WGSL_TYPE_ABSTRACT_INT)
    {
        int64_t i = 0;
        if (v->kind == WGSL_VAL_BOOL) {
            i = v->u.b ? 1 : 0;
        } else if (v->kind == WGSL_VAL_INT) {
            i = v->u.i;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            if (!isfinite(v->u.f)) {
                cev_error(cev, at, "cannot convert non-finite value to integer");
                return 0;
            }
            double t = trunc(v->u.f);
            if (t < (double)INT64_MIN || t > (double)INT64_MAX) {
                cev_error(cev, at, "value out of range for integer constructor");
                return 0;
            }
            i = (int64_t)t;
        } else {
            cev_error(cev, at, "constructor requires scalar components");
            return 0;
        }
        if (!int_fits_target(i, tk)) {
            cev_error(cev, at, "value %lld out of range for %s",
                      (long long)i, wgsl_type_kind_name(tk));
            return 0;
        }
        v->kind = WGSL_VAL_INT;
        v->type = target;
        v->u.i = i;
        return 1;
    }

    if (tk == WGSL_TYPE_F32 || tk == WGSL_TYPE_F16 ||
        tk == WGSL_TYPE_ABSTRACT_FLOAT)
    {
        double d = 0.0;
        if (v->kind == WGSL_VAL_BOOL) {
            d = v->u.b ? 1.0 : 0.0;
        } else if (v->kind == WGSL_VAL_INT) {
            d = (double)v->u.i;
        } else if (v->kind == WGSL_VAL_FLOAT) {
            d = v->u.f;
        } else {
            cev_error(cev, at, "constructor requires scalar components");
            return 0;
        }
        if (!float_fits_target(d, tk)) {
            cev_error(cev, at, "value out of range for %s",
                      wgsl_type_kind_name(tk));
            return 0;
        }
        v->kind = WGSL_VAL_FLOAT;
        v->type = target;
        v->u.f = d;
        return 1;
    }

    return wgsl_consteval_materialize(cev, v, target, at);
}

static int eval_vector_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) return make_zero_value(cev, target, call, out);

    uint32_t width = target->width;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * width);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding vector constructor");
        return 0;
    }

    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind != WGSL_VAL_VEC) {
            if (!convert_constructor_component(cev, &v, elem, call->children[1])) {
                return 0;
            }
            for (uint32_t i = 0; i < width; i++) elems[i] = v;
            out->kind = WGSL_VAL_VEC;
            out->type = target;
            out->u.agg.count = width;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_constructor_value(
                cev, elems, width, &count, elem, &v, call->children[i]))
            return 0;
    }
    if (count != width) {
        cev_error(cev, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)width, (unsigned)count);
        return 0;
    }

    out->kind = WGSL_VAL_VEC;
    out->type = target;
    out->u.agg.count = width;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_matrix_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) return make_zero_value(cev, target, call, out);

    uint32_t total = (uint32_t)target->width * (uint32_t)target->rows;
    WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;

    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind == WGSL_VAL_MAT) {
            if (!v.type || v.type->kind != WGSL_TYPE_MAT ||
                v.type->width != target->width || v.type->rows != target->rows)
            {
                cev_error(cev, call->children[1], "matrix constructor shape mismatch");
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) {
                elems[i] = v.u.agg.elems[i];
                if (!convert_constructor_component(
                        cev, &elems[i], elem, call->children[1]))
                    return 0;
            }
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
        if (v.kind == WGSL_VAL_INT || v.kind == WGSL_VAL_FLOAT) {
            if (!convert_constructor_component(cev, &v, elem, call->children[1])) {
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) elems[i] = v;
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    if (argc > 0) {
        WGSLValue first = {0};
        if (!eval_expr(cev, call->children[1], &first)) return 0;
        if (first.kind == WGSL_VAL_VEC) {
            if (argc != target->width) {
                cev_error(cev, call,
                    "matrix constructor expected %u column vector(s), got %u",
                    (unsigned)target->width, (unsigned)argc);
                return 0;
            }
            WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *elems * total);
            if (!elems) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t c = 0; c < target->width; c++) {
                WGSLValue col = c == 0 ? first : (WGSLValue){0};
                if (c > 0 && !eval_expr(cev, call->children[1 + c], &col))
                    return 0;
                if (col.kind != WGSL_VAL_VEC ||
                    !col.type || col.type->kind != WGSL_TYPE_VEC ||
                    col.u.agg.count != target->rows)
                {
                    cev_error(cev, call->children[1 + c],
                        "matrix constructor column vector width mismatch");
                    return 0;
                }
                for (uint32_t r = 0; r < target->rows; r++) {
                    WGSLValue cell = col.u.agg.elems[r];
                    if (!convert_constructor_component(
                            cev, &cell, elem, call->children[1 + c]))
                        return 0;
                    elems[c * target->rows + r] = cell;
                }
            }
            out->kind = WGSL_VAL_MAT;
            out->type = target;
            out->u.agg.count = total;
            out->u.agg.elems = elems;
            return 1;
        }
    }

    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * total);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding matrix constructor");
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_constructor_value(
                cev, elems, total, &count, elem, &v, call->children[i]))
            return 0;
    }
    if (count != total) {
        cev_error(cev, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        return 0;
    }

    out->kind = WGSL_VAL_MAT;
    out->type = target;
    out->u.agg.count = total;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_array_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    uint32_t length = target->array_len ? target->array_len : argc;
    if (argc == 0 && target->array_len > 0) {
        return make_zero_value(cev, target, call, out);
    }
    if (length == 0) {
        cev_error(cev, call, "array constructor requires at least one element");
        return 0;
    }
    if (argc != length) {
        cev_error(cev, call,
            "array constructor expected %u element(s), got %u",
            (unsigned)length, (unsigned)argc);
        return 0;
    }
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * length);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding array constructor");
        return 0;
    }
    WGSLTypeInfo *elem_type = (WGSLTypeInfo *)target->ref;
    for (uint32_t i = 0; i < length; i++) {
        if (!eval_expr(cev, call->children[1 + i], &elems[i])) return 0;
        if (!wgsl_consteval_materialize(
                cev, &elems[i], elem_type, call->children[1 + i]))
            return 0;
    }
    out->kind = WGSL_VAL_ARRAY;
    out->type = target->array_len ? target
                                  : wgsl_type_array(cev->types, elem_type, length);
    out->u.agg.count = length;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_struct_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    WGSLNode *sd = (WGSLNode *)target->ref;
    uint32_t count = struct_member_count(sd);
    if (argc == 0) return make_zero_value(cev, target, call, out);
    if (argc != count) {
        cev_error(cev, call,
            "struct constructor expected %u member value(s), got %u",
            (unsigned)count, (unsigned)argc);
        return 0;
    }
    WGSLValue *elems = NULL;
    if (count > 0) {
        elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * count);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding struct constructor");
            return 0;
        }
    }
    uint32_t idx = 0;
    for (uint32_t i = 0; sd && i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t attrs = (uint32_t)(m->payload[1] & 0xFFFFFFFFu);
        if (attrs >= m->child_count) return 0;
        WGSLTypeInfo *mt = type_from_typespec(cev, m->children[attrs]);
        if (!mt) return 0;
        if (!eval_expr(cev, call->children[1 + idx], &elems[idx])) return 0;
        if (!wgsl_consteval_materialize(
                cev, &elems[idx], mt, call->children[1 + idx]))
            return 0;
        idx++;
    }
    out->kind = WGSL_VAL_STRUCT;
    out->type = target;
    out->u.agg.count = count;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_constructor_call(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    WGSLValue *out)
{
    if (!target) return 0;
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (target->kind == WGSL_TYPE_BOOL ||
        target->kind == WGSL_TYPE_I32 ||
        target->kind == WGSL_TYPE_U32 ||
        target->kind == WGSL_TYPE_F32 ||
        target->kind == WGSL_TYPE_F16 ||
        target->kind == WGSL_TYPE_ABSTRACT_INT ||
        target->kind == WGSL_TYPE_ABSTRACT_FLOAT)
    {
        if (argc == 0) return make_zero_value(cev, target, call, out);
        if (argc != 1) {
            cev_error(cev, call, "scalar constructor expects 0 or 1 argument");
            return 0;
        }
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (!convert_scalar_constructor_value(
                cev, &v, target, call->children[1]))
        {
            return 0;
        }
        *out = v;
        return 1;
    }

    switch ((WGSLTypeKind)target->kind) {
    case WGSL_TYPE_VEC:    return eval_vector_constructor(cev, call, target, out);
    case WGSL_TYPE_MAT:    return eval_matrix_constructor(cev, call, target, out);
    case WGSL_TYPE_ARRAY:  return eval_array_constructor (cev, call, target, out);
    case WGSL_TYPE_STRUCT: return eval_struct_constructor(cev, call, target, out);
    default:
        cev_error(cev, call, "constructor const-eval not implemented yet (Iter C)");
        return 0;
    }
}

static int append_inferred_component(
    WGSLConstEvaluator *cev, WGSLValue *dst, uint32_t cap,
    uint32_t *count, const WGSLValue *src, const WGSLNode *at)
{
    if (src->kind == WGSL_VAL_VEC) {
        for (uint32_t i = 0; i < src->u.agg.count; i++) {
            if (*count >= cap) {
                cev_error(cev, at, "too many constructor components");
                return 0;
            }
            dst[(*count)++] = src->u.agg.elems[i];
        }
        return 1;
    }
    if (src->kind != WGSL_VAL_BOOL &&
        src->kind != WGSL_VAL_INT &&
        src->kind != WGSL_VAL_FLOAT)
    {
        cev_error(cev, at, "constructor requires scalar or vector components");
        return 0;
    }
    if (*count >= cap) {
        cev_error(cev, at, "too many constructor components");
        return 0;
    }
    dst[(*count)++] = *src;
    return 1;
}

static int eval_inferred_vector_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, uint8_t width, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) {
        WGSLTypeInfo *target = wgsl_type_vec(cev->types, width, cev->types->t_abstract_int);
        return make_zero_value(cev, target, call, out);
    }

    WGSLValue comps[4] = {{0}};
    uint32_t count = 0;
    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind != WGSL_VAL_VEC) {
            WGSLTypeInfo *elem = v.type;
            if (!elem || !materialize_constructor_values(
                    cev, &v, 1, elem, call->children[1]))
                return 0;
            WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *stored * width);
            if (!stored) {
                cev_error(cev, call, "out of memory while folding vector constructor");
                return 0;
            }
            for (uint32_t i = 0; i < width; i++) stored[i] = v;
            out->kind = WGSL_VAL_VEC;
            out->type = wgsl_type_vec(cev->types, width, elem);
            out->u.agg.count = width;
            out->u.agg.elems = stored;
            return 1;
        }
        if (!append_inferred_component(
                cev, comps, width, &count, &v, call->children[1]))
            return 0;
    } else {
        for (uint32_t i = 1; i < call->child_count; i++) {
            WGSLValue v = {0};
            if (!eval_expr(cev, call->children[i], &v)) return 0;
            if (!append_inferred_component(
                    cev, comps, width, &count, &v, call->children[i]))
                return 0;
        }
    }
    if (count != width) {
        cev_error(cev, call,
            "vector constructor expected %u component(s), got %u",
            (unsigned)width, (unsigned)count);
        return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, comps, count, call, &elem)) return 0;
    if (!materialize_constructor_values(cev, comps, count, elem, call)) return 0;

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * width);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding vector constructor");
        return 0;
    }
    for (uint32_t i = 0; i < width; i++) stored[i] = comps[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(cev->types, width, elem);
    out->u.agg.count = width;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_inferred_matrix_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint8_t cols, uint8_t rows, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    uint32_t total = (uint32_t)cols * (uint32_t)rows;
    if (argc == 0) {
        WGSLTypeInfo *target = wgsl_type_mat(
            cev->types, cols, rows, cev->types->t_abstract_int);
        return make_zero_value(cev, target, call, out);
    }
    if (argc == 1) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[1], &v)) return 0;
        if (v.kind == WGSL_VAL_MAT && v.type &&
            v.type->kind == WGSL_TYPE_MAT &&
            v.type->width == cols && v.type->rows == rows)
        {
            *out = v;
            return 1;
        }
        if (v.kind == WGSL_VAL_INT || v.kind == WGSL_VAL_FLOAT) {
            WGSLTypeInfo *elem = v.type;
            if (!elem || !materialize_constructor_values(
                    cev, &v, 1, elem, call->children[1]))
                return 0;
            WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof *stored * total);
            if (!stored) {
                cev_error(cev, call, "out of memory while folding matrix constructor");
                return 0;
            }
            for (uint32_t i = 0; i < total; i++) stored[i] = v;
            out->kind = WGSL_VAL_MAT;
            out->type = wgsl_type_mat(cev->types, cols, rows, elem);
            out->u.agg.count = total;
            out->u.agg.elems = stored;
            return 1;
        }
    }

    WGSLValue comps[16] = {{0}};
    uint32_t count = 0;
    for (uint32_t i = 1; i < call->child_count; i++) {
        WGSLValue v = {0};
        if (!eval_expr(cev, call->children[i], &v)) return 0;
        if (!append_inferred_component(
                cev, comps, total, &count, &v, call->children[i]))
            return 0;
    }
    if (count != total) {
        cev_error(cev, call,
            "matrix constructor expected %u component(s), got %u",
            (unsigned)total, (unsigned)count);
        return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, comps, count, call, &elem)) return 0;
    if (elem == cev->types->t_bool) {
        cev_error(cev, call, "matrix constructor requires numeric components");
        return 0;
    }
    if (!materialize_constructor_values(cev, comps, count, elem, call)) return 0;

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * total);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding matrix constructor");
        return 0;
    }
    for (uint32_t i = 0; i < total; i++) stored[i] = comps[i];
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(cev->types, cols, rows, elem);
    out->u.agg.count = total;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_inferred_array_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out)
{
    uint32_t argc = call->child_count > 0 ? call->child_count - 1 : 0;
    if (argc == 0) {
        cev_error(cev, call, "array constructor requires at least one element");
        return 0;
    }
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * argc);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding array constructor");
        return 0;
    }
    for (uint32_t i = 0; i < argc; i++) {
        if (!eval_expr(cev, call->children[1 + i], &elems[i])) return 0;
    }
    WGSLTypeInfo *elem = NULL;
    if (!common_constructor_type(cev, elems, argc, call, &elem)) return 0;
    if (!materialize_constructor_values(cev, elems, argc, elem, call)) return 0;

    out->kind = WGSL_VAL_ARRAY;
    out->type = wgsl_type_array(cev->types, elem, argc);
    out->u.agg.count = argc;
    out->u.agg.elems = elems;
    return 1;
}

static int eval_inferred_typegen_constructor(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLValue *out)
{
    WGSLNode *callee = call->child_count > 0 ? call->children[0] : NULL;
    if (!callee || callee->kind != WGSL_NODE_EXPR_IDENT) return 0;
    const char *nm = NULL;
    uint32_t len = 0;
    if (!node_ident_text(cev, callee, &nm, &len)) return 0;
    int width = vector_width_from_name(nm, len);
    if (width) return eval_inferred_vector_constructor(
        cev, call, (uint8_t)width, out);
    uint8_t cols = 0, rows = 0;
    if (matrix_dims_from_name(nm, len, &cols, &rows)) {
        return eval_inferred_matrix_constructor(cev, call, cols, rows, out);
    }
    if (len == 5 && memcmp(nm, "array", 5) == 0) {
        return eval_inferred_array_constructor(cev, call, out);
    }
    return 0;
}

static double f16_bits_to_double(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp = (uint32_t)((h >> 10) & 0x1fu);
    uint32_t frac = (uint32_t)(h & 0x03ffu);
    double v = 0.0;
    if (exp == 0) {
        v = frac ? ldexp((double)frac, -24) : 0.0;
    } else if (exp == 31) {
        v = frac ? NAN : INFINITY;
    } else {
        v = ldexp(1.0 + (double)frac / 1024.0, (int)exp - 15);
    }
    return sign ? -v : v;
}

static uint16_t double_to_f16_bits(double x) {
    uint16_t sign = signbit(x) ? 0x8000u : 0u;
    double ax = fabs(x);
    if (ax == 0.0) return sign;
    if (!isfinite(ax)) return (uint16_t)(sign | 0x7c00u);

    if (ax < ldexp(1.0, -14)) {
        double step = ldexp(1.0, -24);
        uint32_t mant = (uint32_t)round_even(ax / step);
        if (mant > 0x3ffu) mant = 0x400u;
        if (mant == 0x400u) return (uint16_t)(sign | 0x0400u);
        return (uint16_t)(sign | mant);
    }

    int e = (int)floor(log2(ax));
    double base = ldexp(1.0, e);
    uint32_t mant = (uint32_t)round_even((ax / base - 1.0) * 1024.0);
    if (mant == 1024u) {
        mant = 0;
        e += 1;
    }
    int exp = e + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        double step = ldexp(1.0, -24);
        uint32_t sm = (uint32_t)round_even(ax / step);
        if (sm > 0x3ffu) sm = 0x3ffu;
        return (uint16_t)(sign | sm);
    }
    return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)(mant & 0x3ffu));
}

static int value_to_u32_bits(
    WGSLConstEvaluator *cev, WGSLNode *at, const WGSLValue *v, uint32_t *bits)
{
    if (!v || !v->type || !bits) return 0;
    if (v->kind == WGSL_VAL_INT) {
        if (v->type != cev->types->t_i32 && v->type != cev->types->t_u32) {
            cev_error(cev, at, "bitcast const-eval requires concrete integer source values");
            return 0;
        }
        *bits = (uint32_t)v->u.i;
        return 1;
    }
    if (v->kind == WGSL_VAL_FLOAT) {
        if (v->type == cev->types->t_f32) {
            float f = (float)v->u.f;
            memcpy(bits, &f, sizeof f);
            return 1;
        }
        cev_error(cev, at, "bitcast const-eval requires f32 float source values");
        return 0;
    }
    cev_error(cev, at, "bitcast const-eval requires numeric source values");
    return 0;
}

static int make_bitcast_component(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *target,
    uint32_t bits, WGSLValue *out)
{
    if (target == cev->types->t_i32) {
        out->kind = WGSL_VAL_INT;
        out->type = target;
        out->u.i = (int32_t)bits;
        return 1;
    }
    if (target == cev->types->t_u32) {
        out->kind = WGSL_VAL_INT;
        out->type = target;
        out->u.i = (int64_t)(uint32_t)bits;
        return 1;
    }
    if (target == cev->types->t_f32) {
        float f = 0.0f;
        memcpy(&f, &bits, sizeof f);
        if (!isfinite((double)f)) {
            cev_error(cev, call,
                "builtin 'bitcast' produced a non-finite f32 value");
            return 0;
        }
        out->kind = WGSL_VAL_FLOAT;
        out->type = target;
        out->u.f = (double)f;
        return 1;
    }
    if (target == cev->types->t_f16) {
        double f = f16_bits_to_double((uint16_t)bits);
        if (!isfinite(f)) {
            cev_error(cev, call,
                "builtin 'bitcast' produced a non-finite f16 value");
            return 0;
        }
        out->kind = WGSL_VAL_FLOAT;
        out->type = target;
        out->u.f = f;
        return 1;
    }
    cev_error(cev, call, "bitcast<T> target is not a concrete numeric scalar");
    return 0;
}

static int eval_call_bitcast_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "bitcast")) return 0;
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "bitcast<T> expects exactly one argument");
        return -1;
    }
    WGSLNode *callee = call->children[0];
    if (!callee || callee->kind != WGSL_NODE_EXPR_TEMPLATED_IDENT ||
        callee->child_count < 2)
    {
        cev_error(cev, call, "bitcast requires an explicit target type");
        return -1;
    }
    WGSLTypeInfo *target = type_from_typespec(cev, callee->children[1]);
    if (!target) {
        cev_error(cev, call, "bitcast target type could not be resolved");
        return -1;
    }

    WGSLValue src = {0};
    if (!eval_expr(cev, call->children[1], &src)) return -1;
    uint32_t src_words[4] = {0};
    uint32_t src_count = 0;
    if (src.kind == WGSL_VAL_VEC) {
        if (src.u.agg.count > 4) {
            cev_error(cev, call, "bitcast source vector is too wide");
            return -1;
        }
        src_count = src.u.agg.count;
        for (uint32_t i = 0; i < src_count; i++) {
            if (!value_to_u32_bits(
                    cev, call->children[1], &src.u.agg.elems[i], &src_words[i]))
            {
                return -1;
            }
        }
    } else {
        src_count = 1;
        if (!value_to_u32_bits(cev, call->children[1], &src, &src_words[0])) {
            return -1;
        }
    }

    if (target->kind == WGSL_TYPE_VEC) {
        WGSLTypeInfo *elem = (WGSLTypeInfo *)target->ref;
        uint32_t width = target->width;
        uint32_t needed_words = elem == cev->types->t_f16
            ? (width + 1u) / 2u : width;
        if (elem == cev->types->t_f16 && (width % 2u) != 0u) {
            cev_error(cev, call, "bitcast to vecN<f16> requires an even width");
            return -1;
        }
        if (src_count != needed_words) {
            cev_error(cev, call, "bitcast source and target bit sizes differ");
            return -1;
        }
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * width);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding bitcast");
            return -1;
        }
        for (uint32_t i = 0; i < width; i++) {
            uint32_t bits = elem == cev->types->t_f16
                ? ((src_words[i / 2u] >> ((i % 2u) * 16u)) & 0xffffu)
                : src_words[i];
            if (!make_bitcast_component(cev, call, elem, bits, &elems[i])) {
                return -1;
            }
        }
        out->kind = WGSL_VAL_VEC;
        out->type = target;
        out->u.agg.count = width;
        out->u.agg.elems = elems;
        return 1;
    }

    if (target == cev->types->t_f16) {
        cev_error(cev, call, "bitcast to scalar f16 is not a 32-bit target");
        return -1;
    }
    if (src_count != 1) {
        cev_error(cev, call, "bitcast source and target bit sizes differ");
        return -1;
    }
    return make_bitcast_component(cev, call, target, src_words[0], out) ? 1 : -1;
}

static int eval_call_unpack_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_4x8_snorm = name_eq(nm, len, "unpack4x8snorm");
    int is_4x8_unorm = name_eq(nm, len, "unpack4x8unorm");
    int is_4xi8      = name_eq(nm, len, "unpack4xI8");
    int is_4xu8      = name_eq(nm, len, "unpack4xU8");
    int is_2x16_snorm = name_eq(nm, len, "unpack2x16snorm");
    int is_2x16_unorm = name_eq(nm, len, "unpack2x16unorm");
    int is_2x16_float = name_eq(nm, len, "unpack2x16float");
    if (!is_4x8_snorm && !is_4x8_unorm && !is_4xi8 && !is_4xu8 &&
        !is_2x16_snorm && !is_2x16_unorm && !is_2x16_float)
    {
        return 0;
    }
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "builtin '%.*s' expects one argument", (int)len, nm);
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (!wgsl_consteval_materialize(cev, &arg, cev->types->t_u32, call->children[1])) {
        return -1;
    }
    uint32_t x = (uint32_t)arg.u.i;
    uint32_t width = (is_2x16_snorm || is_2x16_unorm || is_2x16_float) ? 2u : 4u;
    WGSLTypeInfo *elem_type =
        is_4xi8 ? cev->types->t_i32 :
        is_4xu8 ? cev->types->t_u32 : cev->types->t_f32;
    WGSLValue elems[4] = {{0}};

    for (uint32_t i = 0; i < width; i++) {
        if (is_4xi8) {
            int8_t v = (int8_t)((x >> (i * 8u)) & 0xffu);
            elems[i].kind = WGSL_VAL_INT;
            elems[i].type = elem_type;
            elems[i].u.i = (int64_t)v;
        } else if (is_4xu8) {
            uint8_t v = (uint8_t)((x >> (i * 8u)) & 0xffu);
            elems[i].kind = WGSL_VAL_INT;
            elems[i].type = elem_type;
            elems[i].u.i = (int64_t)v;
        } else {
            double v = 0.0;
            if (is_4x8_snorm) {
                int8_t s = (int8_t)((x >> (i * 8u)) & 0xffu);
                v = (double)s / 127.0;
                if (v < -1.0) v = -1.0;
            } else if (is_4x8_unorm) {
                uint8_t u = (uint8_t)((x >> (i * 8u)) & 0xffu);
                v = (double)u / 255.0;
            } else if (is_2x16_snorm) {
                int16_t s = (int16_t)((x >> (i * 16u)) & 0xffffu);
                v = (double)s / 32767.0;
                if (v < -1.0) v = -1.0;
            } else if (is_2x16_unorm) {
                uint16_t u = (uint16_t)((x >> (i * 16u)) & 0xffffu);
                v = (double)u / 65535.0;
            } else {
                uint16_t h = (uint16_t)((x >> (i * 16u)) & 0xffffu);
                v = f16_bits_to_double(h);
                if (!isfinite(v)) {
                    cev_error(cev, call,
                        "builtin '%.*s' produced a non-finite result",
                        (int)len, nm);
                    return -1;
                }
            }
            elems[i].kind = WGSL_VAL_FLOAT;
            elems[i].type = elem_type;
            elems[i].u.f = (double)(float)v;
        }
    }

    return make_vector_result(cev, call, width, elem_type, elems, out) ? 1 : -1;
}

static int eval_call_pack_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_4x8_snorm = name_eq(nm, len, "pack4x8snorm");
    int is_4x8_unorm = name_eq(nm, len, "pack4x8unorm");
    int is_4xi8      = name_eq(nm, len, "pack4xI8");
    int is_4xu8      = name_eq(nm, len, "pack4xU8");
    int is_4xi8c     = name_eq(nm, len, "pack4xI8Clamp");
    int is_4xu8c     = name_eq(nm, len, "pack4xU8Clamp");
    int is_2x16_snorm = name_eq(nm, len, "pack2x16snorm");
    int is_2x16_unorm = name_eq(nm, len, "pack2x16unorm");
    int is_2x16_float = name_eq(nm, len, "pack2x16float");
    if (!is_4x8_snorm && !is_4x8_unorm && !is_4xi8 && !is_4xu8 &&
        !is_4xi8c && !is_4xu8c &&
        !is_2x16_snorm && !is_2x16_unorm && !is_2x16_float)
    {
        return 0;
    }
    if (!call || call->child_count != 2) {
        cev_error(cev, call, "builtin '%.*s' expects one argument", (int)len, nm);
        return -1;
    }

    uint32_t width = (is_2x16_snorm || is_2x16_unorm || is_2x16_float) ? 2u : 4u;
    WGSLTypeInfo *elem_type =
        (is_4xi8 || is_4xi8c) ? cev->types->t_i32 :
        (is_4xu8 || is_4xu8c) ? cev->types->t_u32 : cev->types->t_f32;
    WGSLTypeInfo *target = wgsl_type_vec(cev->types, (uint8_t)width, elem_type);

    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (!wgsl_consteval_materialize(cev, &arg, target, call->children[1])) {
        return -1;
    }
    if (arg.kind != WGSL_VAL_VEC || arg.u.agg.count != width) {
        cev_error(cev, call, "builtin '%.*s' requires a vec%u argument",
                  (int)len, nm, (unsigned)width);
        return -1;
    }

    uint32_t bits = 0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (is_4xi8 || is_4xi8c) {
            int64_t v = c.u.i;
            if (is_4xi8c) {
                if (v < -128) v = -128;
                if (v > 127) v = 127;
            }
            bits |= ((uint32_t)(uint8_t)(int8_t)v) << (i * 8u);
        } else if (is_4xu8 || is_4xu8c) {
            int64_t v = c.u.i;
            if (is_4xu8c) {
                if (v < 0) v = 0;
                if (v > 255) v = 255;
            }
            bits |= ((uint32_t)(uint8_t)(uint32_t)v) << (i * 8u);
        } else if (is_4x8_snorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 127.0);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            bits |= ((uint32_t)(uint8_t)(int8_t)q) << (i * 8u);
        } else if (is_4x8_unorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 255.0);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            bits |= ((uint32_t)(uint8_t)q) << (i * 8u);
        } else if (is_2x16_snorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 32767.0);
            if (q < -32767) q = -32767;
            if (q > 32767) q = 32767;
            bits |= ((uint32_t)(uint16_t)(int16_t)q) << (i * 16u);
        } else if (is_2x16_unorm) {
            double x = c.u.f;
            if (!isfinite(x)) {
                cev_error(cev, call, "builtin '%.*s' requires finite inputs",
                          (int)len, nm);
                return -1;
            }
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            int q = (int)round_even(x * 65535.0);
            if (q < 0) q = 0;
            if (q > 65535) q = 65535;
            bits |= ((uint32_t)(uint16_t)q) << (i * 16u);
        } else {
            double x = c.u.f;
            if (!isfinite(x) || x < -65504.0 || x > 65504.0) {
                cev_error(cev, call,
                    "builtin '%.*s' input is outside the finite f16 range",
                    (int)len, nm);
                return -1;
            }
            bits |= ((uint32_t)double_to_f16_bits(x)) << (i * 16u);
        }
    }

    out->kind = WGSL_VAL_INT;
    out->type = cev->types->t_u32;
    out->u.i = (int64_t)bits;
    return 1;
}

static int eval_call_dot4_packed_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_i = name_eq(nm, len, "dot4I8Packed");
    int is_u = name_eq(nm, len, "dot4U8Packed");
    if (!is_i && !is_u) return 0;
    if (!call || call->child_count != 3) {
        cev_error(cev, call, "builtin '%.*s' expects two arguments", (int)len, nm);
        return -1;
    }
    WGSLValue a = {0}, b = {0};
    if (!eval_expr(cev, call->children[1], &a) ||
        !eval_expr(cev, call->children[2], &b))
    {
        return -1;
    }
    if (!wgsl_consteval_materialize(cev, &a, cev->types->t_u32, call->children[1]) ||
        !wgsl_consteval_materialize(cev, &b, cev->types->t_u32, call->children[2]))
    {
        return -1;
    }
    uint32_t ua = (uint32_t)a.u.i;
    uint32_t ub = (uint32_t)b.u.i;
    int64_t sum = 0;
    for (uint32_t i = 0; i < 4; i++) {
        uint8_t ba = (uint8_t)((ua >> (i * 8u)) & 0xffu);
        uint8_t bb = (uint8_t)((ub >> (i * 8u)) & 0xffu);
        if (is_i) {
            sum += (int64_t)(int8_t)ba * (int64_t)(int8_t)bb;
        } else {
            sum += (int64_t)ba * (int64_t)bb;
        }
    }
    out->kind = WGSL_VAL_INT;
    out->type = is_i ? cev->types->t_i32 : cev->types->t_u32;
    out->u.i = is_i ? (int64_t)(int32_t)sum : (int64_t)(uint32_t)sum;
    return 1;
}

static int eval_unary_float_builtin_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out)
{
    WGSLTypeInfo *result_type = arg->type;
    if (arg->kind == WGSL_VAL_INT) {
        if (arg->type != cev->types->t_abstract_int) {
            cev_error(cev, call,
                "'%.*s' requires a floating-point argument",
                (int)len, nm);
            return 0;
        }
        if (!wgsl_consteval_materialize(
                cev, arg, cev->types->t_abstract_float, call))
            return 0;
        result_type = cev->types->t_abstract_float;
    }
    if (arg->kind != WGSL_VAL_FLOAT) {
        cev_error(cev, call,
            "'%.*s' requires a floating-point argument", (int)len, nm);
        return 0;
    }

    double x = arg->u.f;
    double r = 0.0;
    const char *domain = NULL;

    if (name_eq(nm, len, "sqrt")) {
        if (x < 0.0) domain = "argument must be >= 0";
        r = sqrt(x);
    } else if (name_eq(nm, len, "inverseSqrt")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = 1.0 / sqrt(x);
    } else if (name_eq(nm, len, "log")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = log(x);
    } else if (name_eq(nm, len, "log2")) {
        if (x <= 0.0) domain = "argument must be > 0";
        r = log2(x);
    } else if (name_eq(nm, len, "asin")) {
        if (x < -1.0 || x > 1.0) domain = "argument must be in [-1, 1]";
        r = asin(x);
    } else if (name_eq(nm, len, "acos")) {
        if (x < -1.0 || x > 1.0) domain = "argument must be in [-1, 1]";
        r = acos(x);
    } else if (name_eq(nm, len, "acosh")) {
        if (x < 1.0) domain = "argument must be >= 1";
        r = acosh(x);
    } else if (name_eq(nm, len, "atanh")) {
        if (x <= -1.0 || x >= 1.0) domain = "argument must be in (-1, 1)";
        r = atanh(x);
    } else if (name_eq(nm, len, "ceil")) {
        r = ceil(x);
    } else if (name_eq(nm, len, "floor")) {
        r = floor(x);
    } else if (name_eq(nm, len, "fract")) {
        r = x - floor(x);
    } else if (name_eq(nm, len, "round")) {
        r = round_even(x);
    } else if (name_eq(nm, len, "trunc")) {
        r = trunc(x);
    } else if (name_eq(nm, len, "sin")) {
        r = sin(x);
    } else if (name_eq(nm, len, "cos")) {
        r = cos(x);
    } else if (name_eq(nm, len, "tan")) {
        r = tan(x);
    } else if (name_eq(nm, len, "sinh")) {
        r = sinh(x);
    } else if (name_eq(nm, len, "cosh")) {
        r = cosh(x);
    } else if (name_eq(nm, len, "tanh")) {
        r = tanh(x);
    } else if (name_eq(nm, len, "exp")) {
        r = exp(x);
    } else if (name_eq(nm, len, "exp2")) {
        r = exp2(x);
    } else if (name_eq(nm, len, "atan")) {
        r = atan(x);
    } else if (name_eq(nm, len, "asinh")) {
        r = asinh(x);
    } else if (name_eq(nm, len, "saturate")) {
        r = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
    } else if (name_eq(nm, len, "radians")) {
        r = x * 0.017453292519943295769;
    } else if (name_eq(nm, len, "degrees")) {
        r = x * 57.295779513082320877;
    } else if (name_eq(nm, len, "quantizeToF16")) {
        if (x < -65504.0 || x > 65504.0) {
            domain = "argument must be in the finite f16 range";
        }
        r = x;
    }

    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, result_type, r, out);
}

static int eval_call_unary_float_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "sqrt") || name_eq(nm, len, "inverseSqrt") ||
        name_eq(nm, len, "log")  || name_eq(nm, len, "log2") ||
        name_eq(nm, len, "asin") || name_eq(nm, len, "acos") ||
        name_eq(nm, len, "acosh") || name_eq(nm, len, "atanh") ||
        name_eq(nm, len, "ceil") || name_eq(nm, len, "floor") ||
        name_eq(nm, len, "fract") || name_eq(nm, len, "round") ||
        name_eq(nm, len, "trunc") || name_eq(nm, len, "sin") ||
        name_eq(nm, len, "cos") || name_eq(nm, len, "tan") ||
        name_eq(nm, len, "sinh") || name_eq(nm, len, "cosh") ||
        name_eq(nm, len, "tanh") || name_eq(nm, len, "exp") ||
        name_eq(nm, len, "exp2") || name_eq(nm, len, "atan") ||
        name_eq(nm, len, "asinh") || name_eq(nm, len, "saturate") ||
        name_eq(nm, len, "radians") || name_eq(nm, len, "degrees") ||
        name_eq(nm, len, "quantizeToF16");
    if (!supported) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }

    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;

    if (arg.kind == WGSL_VAL_VEC) {
        WGSLTypeInfo *vt = arg.type;
        if (!vt || vt->kind != WGSL_TYPE_VEC) {
            cev_error(cev, call,
                "'%.*s' requires a floating-point argument", (int)len, nm);
            return -1;
        }
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * arg.u.agg.count);
        if (!elems) {
            cev_error(cev, call, "out of memory while folding builtin '%.*s'",
                      (int)len, nm);
            return -1;
        }
        for (uint32_t i = 0; i < arg.u.agg.count; i++) {
            WGSLValue c = arg.u.agg.elems[i];
            if (!eval_unary_float_builtin_scalar(cev, call, nm, len, &c, &elems[i])) {
                return -1;
            }
        }
        out->kind = WGSL_VAL_VEC;
        out->type = vt;
        out->u.agg.count = arg.u.agg.count;
        out->u.agg.elems = elems;
        return 1;
    }

    if (!eval_unary_float_builtin_scalar(cev, call, nm, len, &arg, out)) {
        return -1;
    }
    return 1;
}

static int fold_abs_sign_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *arg, WGSLValue *out)
{
    if (name_eq(nm, len, "abs")) {
        if (arg->kind == WGSL_VAL_INT) {
            *out = *arg;
            if (arg->type == cev->types->t_u32 || arg->u.i == INT64_MIN ||
                (arg->type == cev->types->t_i32 && arg->u.i == INT32_MIN))
            {
                return 1;
            }
            if (arg->u.i < 0) out->u.i = -arg->u.i;
            return 1;
        }
        if (!require_float_scalar(cev, call, nm, len, arg)) return 0;
        return finish_float_builtin_result(
            cev, call, nm, len, arg->type, fabs(arg->u.f), out);
    }

    if (name_eq(nm, len, "sign")) {
        if (arg->kind == WGSL_VAL_INT) {
            if (arg->type == cev->types->t_u32) {
                cev_error(cev, call, "'sign' does not accept u32");
                return 0;
            }
            *out = *arg;
            out->u.i = arg->u.i > 0 ? 1 : (arg->u.i < 0 ? -1 : 0);
            return 1;
        }
        if (!require_float_scalar(cev, call, nm, len, arg)) return 0;
        double r = arg->u.f > 0.0 ? 1.0 : (arg->u.f < 0.0 ? -1.0 : 0.0);
        return finish_float_builtin_result(
            cev, call, nm, len, arg->type, r, out);
    }
    return 0;
}

static int eval_call_abs_sign_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "abs") && !name_eq(nm, len, "sign")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = value_component(&arg, i);
        if (!fold_abs_sign_scalar(cev, call, nm, len, &c, &tmp[i])) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int fold_minmax_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call)) return 0;
    if (a->kind == WGSL_VAL_INT && b->kind == WGSL_VAL_INT) {
        *out = *a;
        if (a->type == cev->types->t_u32) {
            uint32_t au = (uint32_t)a->u.i, bu = (uint32_t)b->u.i;
            uint32_t r = name_eq(nm, len, "min")
                ? (au < bu ? au : bu)
                : (au > bu ? au : bu);
            out->u.i = (int64_t)r;
        } else {
            out->u.i = name_eq(nm, len, "min")
                ? (a->u.i < b->u.i ? a->u.i : b->u.i)
                : (a->u.i > b->u.i ? a->u.i : b->u.i);
        }
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double r = name_eq(nm, len, "min")
        ? (a->u.f < b->u.f ? a->u.f : b->u.f)
        : (a->u.f > b->u.f ? a->u.f : b->u.f);
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int fold_float_binary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *a, WGSLValue *b, WGSLValue *out)
{
    if (!promote_pair(cev, a, b, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b))
    {
        return 0;
    }
    double x = a->u.f, y = b->u.f, r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "pow")) {
        if (x < 0.0) domain = "base must be >= 0";
        else if (x == 0.0 && y <= 0.0) domain = "zero base requires a positive exponent";
        r = pow(x, y);
    } else if (name_eq(nm, len, "atan2")) {
        r = atan2(x, y);
    } else if (name_eq(nm, len, "step")) {
        r = y < x ? 0.0 : 1.0;
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int eval_call_binary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "min") || name_eq(nm, len, "max") ||
        name_eq(nm, len, "pow") || name_eq(nm, len, "atan2") ||
        name_eq(nm, len, "step");
    if (!supported) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin '%.*s' expects 2 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        int ok = (name_eq(nm, len, "min") || name_eq(nm, len, "max"))
            ? fold_minmax_scalar(cev, call, nm, len, &a, &b, &tmp[i])
            : fold_float_binary_scalar(cev, call, nm, len, &a, &b, &tmp[i]);
        if (!ok) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int fold_clamp_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *x, WGSLValue *lo, WGSLValue *hi, WGSLValue *out)
{
    if (!promote_three(cev, x, lo, hi, call)) return 0;
    if (x->kind == WGSL_VAL_INT && lo->kind == WGSL_VAL_INT &&
        hi->kind == WGSL_VAL_INT)
    {
        *out = *x;
        if (x->type == cev->types->t_u32) {
            uint32_t xv = (uint32_t)x->u.i;
            uint32_t lv = (uint32_t)lo->u.i;
            uint32_t hv = (uint32_t)hi->u.i;
            if (lv > hv) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            if (xv < lv) xv = lv;
            if (xv > hv) xv = hv;
            out->u.i = (int64_t)xv;
        } else {
            if (lo->u.i > hi->u.i) {
                cev_error(cev, call,
                    "builtin 'clamp' domain error: low must be <= high");
                return 0;
            }
            int64_t r = x->u.i < lo->u.i ? lo->u.i : x->u.i;
            if (r > hi->u.i) r = hi->u.i;
            out->u.i = r;
        }
        (void)nm; (void)len;
        return 1;
    }
    if (!require_float_scalar(cev, call, nm, len, x) ||
        !require_float_scalar(cev, call, nm, len, lo) ||
        !require_float_scalar(cev, call, nm, len, hi))
    {
        return 0;
    }
    if (lo->u.f > hi->u.f) {
        cev_error(cev, call,
            "builtin 'clamp' domain error: low must be <= high");
        return 0;
    }
    double r = x->u.f < lo->u.f ? lo->u.f : x->u.f;
    if (r > hi->u.f) r = hi->u.f;
    return finish_float_builtin_result(cev, call, nm, len, x->type, r, out);
}

static int fold_float_ternary_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *a, WGSLValue *b, WGSLValue *c, WGSLValue *out)
{
    if (!promote_three(cev, a, b, c, call) ||
        !require_float_scalar(cev, call, nm, len, a) ||
        !require_float_scalar(cev, call, nm, len, b) ||
        !require_float_scalar(cev, call, nm, len, c))
    {
        return 0;
    }
    double r = 0.0;
    const char *domain = NULL;
    if (name_eq(nm, len, "mix")) {
        double one_minus_c = 0.0;
        double lhs = 0.0;
        double rhs = 0.0;
        if (!round_float_value_to_type(cev, a->type, 1.0 - c->u.f,
                                       &one_minus_c) ||
            !round_float_value_to_type(cev, a->type, a->u.f * one_minus_c,
                                       &lhs) ||
            !round_float_value_to_type(cev, a->type, b->u.f * c->u.f,
                                       &rhs) ||
            !round_float_value_to_type(cev, a->type, lhs + rhs, &r))
        {
            cev_error(cev, call,
                "builtin '%.*s' produced a non-finite result "
                "(spec §15.7.3 finite-math assumption)",
                (int)len, nm);
            return 0;
        }
    } else if (name_eq(nm, len, "fma")) {
        r = fma(a->u.f, b->u.f, c->u.f);
    } else if (name_eq(nm, len, "smoothstep")) {
        if (a->u.f == b->u.f) {
            domain = "edge0 and edge1 must differ";
        } else {
            double t = (c->u.f - a->u.f) / (b->u.f - a->u.f);
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            r = t * t * (3.0 - 2.0 * t);
        }
    }
    if (domain) {
        cev_error(cev, call,
            "builtin '%.*s' domain error: %s", (int)len, nm, domain);
        return 0;
    }
    return finish_float_builtin_result(cev, call, nm, len, a->type, r, out);
}

static int eval_call_ternary_numeric_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "clamp") || name_eq(nm, len, "mix") ||
        name_eq(nm, len, "fma") || name_eq(nm, len, "smoothstep");
    if (!supported) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin '%.*s' expects 3 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    int broadcast = name_eq(nm, len, "mix") ? 2 : -1;
    if (!componentwise_shape(cev, call, nm, len, vals, 3, broadcast,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue a = value_component(&vals[0], i);
        WGSLValue b = value_component(&vals[1], i);
        WGSLValue c = value_component(&vals[2], i);
        int ok = name_eq(nm, len, "clamp")
            ? fold_clamp_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i])
            : fold_float_ternary_scalar(cev, call, nm, len, &a, &b, &c, &tmp[i]);
        if (!ok) return -1;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_ldexp_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "ldexp")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'ldexp' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue x = value_component(&vals[0], i);
        WGSLValue e = value_component(&vals[1], i);
        if (!require_float_scalar(cev, call, nm, len, &x)) return -1;
        if (e.kind != WGSL_VAL_INT) {
            cev_error(cev, call, "'ldexp' requires an integer exponent");
            return -1;
        }
        if (e.type == cev->types->t_u32) {
            cev_error(cev, call, "'ldexp' requires a signed integer exponent");
            return -1;
        }
        if (x.type != cev->types->t_abstract_float &&
            e.type == cev->types->t_abstract_int &&
            !wgsl_consteval_materialize(cev, &e, cev->types->t_i32, call))
        {
            return -1;
        }
        int bias = x.type == cev->types->t_f16 ? 15 :
                   x.type == cev->types->t_f32 ? 127 : 1023;
        if (e.u.i > bias + 1) {
            cev_error(cev, call,
                "builtin 'ldexp' domain error: exponent too large");
            return -1;
        }
        double r = e.u.i < -1075 ? 0.0 : ldexp(x.u.f, (int)e.u.i);
        if (!finish_float_builtin_result(cev, call, nm, len, x.type, r, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_length_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "length")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'length' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    WGSLValue first = value_component(&arg, 0);
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    uint32_t width = value_width(&arg);
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue c = value_component(&arg, i);
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        sum += c.u.f * c.u.f;
    }
    double r = width == 1 ? fabs(first.u.f) : sqrt(sum);
    return finish_float_builtin_result(cev, call, nm, len, first.type, r, out) ? 1 : -1;
}

static int eval_call_distance_dot_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_distance = name_eq(nm, len, "distance");
    int is_dot = name_eq(nm, len, "dot");
    if (!is_distance && !is_dot) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin '%.*s' expects 2 arguments; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (is_distance && vals[0].kind != WGSL_VAL_VEC &&
        vals[1].kind != WGSL_VAL_VEC)
    {
        WGSLValue a = vals[0];
        WGSLValue b = vals[1];
        if (!promote_pair(cev, &a, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        return finish_float_builtin_result(
            cev, call, nm, len, a.type, fabs(a.u.f - b.u.f), out) ? 1 : -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != vals[1].u.agg.count)
    {
        cev_error(cev, call,
            "builtin '%.*s' requires same-width vector arguments",
            (int)len, nm);
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call)) return -1;
    if (is_distance) {
        if (!require_float_scalar(cev, call, nm, len, &a0) ||
            !require_float_scalar(cev, call, nm, len, &b0))
        {
            return -1;
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call) ||
                !promote_pair(cev, &a0, &b, call) ||
                !require_float_scalar(cev, call, nm, len, &a) ||
                !require_float_scalar(cev, call, nm, len, &b))
            {
                return -1;
            }
            double d = a.u.f - b.u.f;
            sum += d * d;
        }
        return finish_float_builtin_result(
            cev, call, nm, len, a0.type, sqrt(sum), out) ? 1 : -1;
    }

    if (a0.kind == WGSL_VAL_INT && b0.kind == WGSL_VAL_INT) {
        int64_t sum = 0;
        int is_u = a0.type == cev->types->t_u32;
        int is_abstract = a0.type == cev->types->t_abstract_int;
        for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
            WGSLValue a = vals[0].u.agg.elems[i];
            WGSLValue b = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &a, &b, call) ||
                !promote_pair(cev, &a0, &a, call))
            {
                return -1;
            }
            if (is_u) {
                uint64_t prod = (uint64_t)(uint32_t)a.u.i * (uint64_t)(uint32_t)b.u.i;
                sum = (int64_t)(uint32_t)((uint32_t)sum + (uint32_t)prod);
            } else if (is_abstract) {
                int64_t prod = 0;
                if (!checked_abstract_int_result(
                        cev, call, (__int128)a.u.i * (__int128)b.u.i,
                        "dot", &prod) ||
                    !checked_abstract_int_result(
                        cev, call, (__int128)sum + (__int128)prod,
                        "dot", &sum))
                {
                    return -1;
                }
            } else {
                sum += a.u.i * b.u.i;
                if (a0.type == cev->types->t_i32) sum = (int32_t)sum;
            }
        }
        out->kind = WGSL_VAL_INT;
        out->type = a0.type;
        out->u.i = sum;
        return 1;
    }

    if (!require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < vals[0].u.agg.count; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        sum += a.u.f * b.u.f;
    }
    return finish_float_builtin_result(cev, call, nm, len, a0.type, sum, out) ? 1 : -1;
}

static int eval_call_cross_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "cross")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'cross' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    if (vals[0].kind != WGSL_VAL_VEC || vals[1].kind != WGSL_VAL_VEC ||
        vals[0].u.agg.count != 3 || vals[1].u.agg.count != 3)
    {
        cev_error(cev, call, "builtin 'cross' requires vec3 arguments");
        return -1;
    }
    WGSLValue a0 = vals[0].u.agg.elems[0];
    WGSLValue b0 = vals[1].u.agg.elems[0];
    if (!promote_pair(cev, &a0, &b0, call) ||
        !require_float_scalar(cev, call, nm, len, &a0) ||
        !require_float_scalar(cev, call, nm, len, &b0))
    {
        return -1;
    }
    double av[3], bv[3];
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue a = vals[0].u.agg.elems[i];
        WGSLValue b = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &a, &b, call) ||
            !promote_pair(cev, &a0, &a, call) ||
            !promote_pair(cev, &a0, &b, call) ||
            !require_float_scalar(cev, call, nm, len, &a) ||
            !require_float_scalar(cev, call, nm, len, &b))
        {
            return -1;
        }
        av[i] = a.u.f;
        bv[i] = b.u.f;
    }
    WGSLValue tmp[3] = {{0}};
    static const uint8_t idx[3][4] = {
        { 1, 2, 2, 1 },
        { 2, 0, 0, 2 },
        { 0, 1, 1, 0 },
    };
    for (uint32_t i = 0; i < 3; i++) {
        WGSLValue lhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][0]],
        };
        WGSLValue lhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][1]],
        };
        WGSLValue rhs_a = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = av[idx[i][2]],
        };
        WGSLValue rhs_b = {
            .kind = WGSL_VAL_FLOAT, .type = a0.type, .u.f = bv[idx[i][3]],
        };
        WGSLValue lhs = {0}, rhs = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &lhs_a, &lhs_b, &lhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_STAR, &rhs_a, &rhs_b, &rhs) ||
            !eval_binary_float(cev, call, WGSL_TOK_MINUS, &lhs, &rhs, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, 3, a0.type, tmp, out) ? 1 : -1;
}

static int eval_call_normalize_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "normalize")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'normalize' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_VEC || arg.u.agg.count == 0) {
        cev_error(cev, call, "builtin 'normalize' requires a vector argument");
        return -1;
    }
    WGSLValue first = arg.u.agg.elems[0];
    if (!require_float_scalar(cev, call, nm, len, &first)) return -1;
    WGSLValue sumv = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = 0.0,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        WGSLValue prod = {0};
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &cv, &cv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sumv, &prod, &next))
        {
            return -1;
        }
        sumv = next;
    }
    double lenv = 0.0;
    if (!round_float_value_to_type(cev, first.type, sqrt(sumv.u.f), &lenv)) {
        cev_error(cev, call,
            "builtin 'normalize' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)");
        return -1;
    }
    if (lenv == 0.0) {
        cev_error(cev, call,
            "builtin 'normalize' domain error: argument must not be the zero vector");
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    WGSLValue denom = {
        .kind = WGSL_VAL_FLOAT,
        .type = first.type,
        .u.f = lenv,
    };
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (!promote_pair(cev, &first, &c, call) ||
            !require_float_scalar(cev, call, nm, len, &c))
        {
            return -1;
        }
        WGSLValue cv = {
            .kind = WGSL_VAL_FLOAT,
            .type = first.type,
            .u.f = c.u.f,
        };
        if (!eval_binary_float(cev, call, WGSL_TOK_SLASH, &cv, &denom, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(
        cev, call, arg.u.agg.count, arg.type, tmp, out) ? 1 : -1;
}

static int fold_float_dot(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    const WGSLValue *a, const WGSLValue *b,
    uint32_t width, WGSLTypeInfo **out_type, double *out_dot)
{
    WGSLValue at = a->u.agg.elems[0];
    WGSLValue bt = b->u.agg.elems[0];
    if (!promote_pair(cev, &at, &bt, call) ||
        !require_float_scalar(cev, call, nm, len, &at) ||
        !require_float_scalar(cev, call, nm, len, &bt))
    {
        return 0;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue ai = a->u.agg.elems[i];
        WGSLValue bi = b->u.agg.elems[i];
        if (!promote_pair(cev, &ai, &bi, call) ||
            !promote_pair(cev, &at, &ai, call) ||
            !promote_pair(cev, &at, &bi, call) ||
            !require_float_scalar(cev, call, nm, len, &ai) ||
            !require_float_scalar(cev, call, nm, len, &bi))
        {
            return 0;
        }
        WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = ai.u.f };
        WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = bi.u.f };
        WGSLValue prod = {0};
        WGSLValue sv = { .kind = WGSL_VAL_FLOAT, .type = at.type, .u.f = sum };
        WGSLValue next = {0};
        if (!eval_binary_float(cev, call, WGSL_TOK_STAR, &av, &bv, &prod) ||
            !eval_binary_float(cev, call, WGSL_TOK_PLUS, &sv, &prod, &next)) {
            return 0;
        }
        sum = next.u.f;
    }
    if (!isfinite(sum)) {
        cev_error(cev, call,
            "builtin '%.*s' produced a non-finite result "
            "(spec §15.7.3 finite-math assumption)",
            (int)len, nm);
        return 0;
    }
    *out_type = at.type;
    *out_dot = sum;
    return 1;
}

static int fold_float_inherited_binary(
    WGSLConstEvaluator *cev, WGSLNode *call, WGSLTypeInfo *type,
    WGSLTokenKind op, double a, double b, double *out)
{
    WGSLValue av = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = a };
    WGSLValue bv = { .kind = WGSL_VAL_FLOAT, .type = type, .u.f = b };
    WGSLValue rv = {0};
    if (!eval_binary_float(cev, call, op, &av, &bv, &rv)) return 0;
    *out = rv.u.f;
    return 1;
}

static int eval_call_reflect_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "reflect")) return 0;
    if (call->child_count != 3) {
        cev_error(cev, call,
            "builtin 'reflect' expects 2 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue I = vals[0].u.agg.elems[i];
        WGSLValue N = vals[1].u.agg.elems[i];
        if (!promote_pair(cev, &I, &N, call) ||
            !wgsl_consteval_materialize(cev, &I, rt, call) ||
            !wgsl_consteval_materialize(cev, &N, rt, call))
        {
            return -1;
        }
        double r = I.u.f - 2.0 * d * N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_face_forward_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "faceForward")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'faceForward' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 3, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[2], &vals[1],
                        width, &rt, &d))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    int keep = d < 0.0;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue N = vals[0].u.agg.elems[i];
        if (!wgsl_consteval_materialize(cev, &N, rt, call)) return -1;
        double r = keep ? N.u.f : -N.u.f;
        if (!finish_float_builtin_result(cev, call, nm, len, rt, r, &tmp[i])) {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_refract_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "refract")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'refract' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[2] = {{0}};
    WGSLValue eta = {0};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &eta))
    {
        return -1;
    }
    uint32_t width = 0;
    if (!require_same_width_vectors(cev, call, nm, len, vals, 2, &width)) {
        return -1;
    }
    WGSLTypeInfo *rt = NULL;
    double d = 0.0;
    if (!fold_float_dot(cev, call, nm, len, &vals[1], &vals[0],
                        width, &rt, &d) ||
        !wgsl_consteval_materialize(cev, &eta, rt, call) ||
        !require_float_scalar(cev, call, nm, len, &eta))
    {
        return -1;
    }

    double eta2 = 0.0;
    double d2 = 0.0;
    double one_minus_d2 = 0.0;
    double term = 0.0;
    double k = 0.0;
    if (!fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta.u.f, eta.u.f, &eta2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, d, d, &d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, d2, &one_minus_d2) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_STAR, eta2, one_minus_d2, &term) ||
        !fold_float_inherited_binary(
            cev, call, rt, WGSL_TOK_MINUS, 1.0, term, &k))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        double r = 0.0;
        if (k >= 0.0) {
            WGSLValue I = vals[0].u.agg.elems[i];
            WGSLValue N = vals[1].u.agg.elems[i];
            if (!promote_pair(cev, &I, &N, call) ||
                !wgsl_consteval_materialize(cev, &I, rt, call) ||
                !wgsl_consteval_materialize(cev, &N, rt, call))
            {
                return -1;
            }
            double eta_i = 0.0;
            double eta_d = 0.0;
            double sum = 0.0;
            double scaled_n = 0.0;
            WGSLValue sqrt_v = {0};
            if (!fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, I.u.f, &eta_i) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, eta.u.f, d, &eta_d) ||
                !finish_float_builtin_result(
                    cev, call, nm, len, rt, sqrt(k), &sqrt_v) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_PLUS, eta_d, sqrt_v.u.f, &sum) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_STAR, sum, N.u.f, &scaled_n) ||
                !fold_float_inherited_binary(
                    cev, call, rt, WGSL_TOK_MINUS, eta_i, scaled_n, &r))
            {
                return -1;
            }
        }
        if (!finish_float_builtin_result_relaxed_range(
                cev, call, nm, len, rt, r, &tmp[i]))
        {
            return -1;
        }
    }
    return make_vector_result(cev, call, width, rt, tmp, out) ? 1 : -1;
}

static int eval_call_transpose_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "transpose")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'transpose' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_MAT || !arg.type ||
        arg.type->kind != WGSL_TYPE_MAT)
    {
        cev_error(cev, call, "builtin 'transpose' requires a matrix argument");
        return -1;
    }
    uint32_t cols = arg.type->width;
    uint32_t rows = arg.type->rows;
    uint32_t count = cols * rows;
    WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *elems * count);
    if (!elems) {
        cev_error(cev, call, "out of memory while folding transpose");
        return -1;
    }
    for (uint32_t out_col = 0; out_col < rows; out_col++) {
        for (uint32_t out_row = 0; out_row < cols; out_row++) {
            uint32_t src_col = out_row;
            uint32_t src_row = out_col;
            elems[out_col * cols + out_row] =
                arg.u.agg.elems[src_col * rows + src_row];
        }
    }
    out->kind = WGSL_VAL_MAT;
    out->type = wgsl_type_mat(
        cev->types, (uint8_t)rows, (uint8_t)cols,
        (WGSLTypeInfo *)arg.type->ref);
    out->u.agg.count = count;
    out->u.agg.elems = elems;
    return 1;
}

static int value_numeric_to_double(const WGSLValue *v, double *out) {
    if (!v || !out) return 0;
    if (v->kind == WGSL_VAL_INT) {
        *out = (double)v->u.i;
        return 1;
    }
    if (v->kind == WGSL_VAL_FLOAT) {
        *out = v->u.f;
        return 1;
    }
    return 0;
}

static double det_small(double m[4][4], int n) {
    if (n == 1) return m[0][0];
    if (n == 2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    double sum = 0.0;
    for (int col = 0; col < n; col++) {
        double sub[4][4] = {{0.0}};
        for (int r = 1; r < n; r++) {
            int dc = 0;
            for (int c = 0; c < n; c++) {
                if (c == col) continue;
                sub[r - 1][dc++] = m[r][c];
            }
        }
        double term = m[0][col] * det_small(sub, n - 1);
        sum += (col & 1) ? -term : term;
    }
    return sum;
}

static int eval_call_determinant_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "determinant")) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin 'determinant' expects 1 argument; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind != WGSL_VAL_MAT || !arg.type ||
        arg.type->kind != WGSL_TYPE_MAT ||
        arg.type->width != arg.type->rows)
    {
        cev_error(cev, call,
            "builtin 'determinant' requires a square matrix argument");
        return -1;
    }
    int n = (int)arg.type->width;
    if (n < 2 || n > 4) {
        cev_error(cev, call,
            "builtin 'determinant' supports mat2x2, mat3x3, or mat4x4");
        return -1;
    }

    double m[4][4] = {{0.0}};
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < n; r++) {
            WGSLValue cell = arg.u.agg.elems[(uint32_t)c * arg.type->rows + (uint32_t)r];
            if (!value_numeric_to_double(&cell, &m[r][c])) {
                cev_error(cev, call,
                    "builtin 'determinant' requires numeric matrix components");
                return -1;
            }
        }
    }

    WGSLTypeInfo *elem = (WGSLTypeInfo *)arg.type->ref;
    WGSLTypeInfo *rt = elem;
    if (rt == cev->types->t_abstract_int ||
        rt == cev->types->t_i32 ||
        rt == cev->types->t_u32)
    {
        rt = cev->types->t_abstract_float;
    }
    double d = det_small(m, n);
    return finish_float_builtin_result(cev, call, nm, len, rt, d, out) ? 1 : -1;
}

static int read_u32_const_arg(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, const WGSLValue *v, uint32_t *out)
{
    WGSLValue x = *v;
    if (x.kind != WGSL_VAL_INT) {
        cev_error(cev, call,
            "builtin '%.*s' requires u32 offset/count arguments",
            (int)len, nm);
        return 0;
    }
    if (!wgsl_consteval_materialize(cev, &x, cev->types->t_u32, call)) {
        return 0;
    }
    *out = (uint32_t)x.u.i;
    return 1;
}

static int bitfield_shape(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len,
    WGSLValue *value, WGSLValue *newbits,
    uint32_t *out_width, WGSLTypeInfo **out_vec_type)
{
    WGSLValue vals[2] = {*value, {0}};
    int argc = 1;
    if (newbits) {
        vals[1] = *newbits;
        argc = 2;
    }
    if (!componentwise_shape(cev, call, nm, len, vals, argc, -1,
                             out_width, out_vec_type))
    {
        return 0;
    }
    if (newbits) {
        *value = vals[0];
        *newbits = vals[1];
    } else {
        *value = vals[0];
    }
    return 1;
}

static int fold_extract_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    WGSLValue *value, uint32_t offset, uint32_t count, WGSLValue *out)
{
    if (value->kind != WGSL_VAL_INT) {
        cev_error(cev, call, "extractBits value must be integer");
        return 0;
    }
    if (offset + count > 32) {
        cev_error(cev, call,
            "builtin 'extractBits' domain error: offset + count must be <= 32");
        return 0;
    }
    uint32_t bits = (uint32_t)value->u.i;
    uint32_t r = 0;
    if (count != 0) {
        uint32_t mask = count == 32 ? UINT32_MAX : ((1u << count) - 1u);
        r = (bits >> offset) & mask;
        if (value->type == cev->types->t_i32 && count < 32 &&
            (r & (1u << (count - 1u))))
        {
            r |= ~mask;
        }
    }
    out->kind = WGSL_VAL_INT;
    out->type = value->type;
    out->u.i = value->type == cev->types->t_u32
        ? (int64_t)r : (int64_t)(int32_t)r;
    return 1;
}

static int fold_insert_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    WGSLValue *value, WGSLValue *newbits,
    uint32_t offset, uint32_t count, WGSLValue *out)
{
    if (value->kind != WGSL_VAL_INT || newbits->kind != WGSL_VAL_INT) {
        cev_error(cev, call, "insertBits values must be integers");
        return 0;
    }
    if (!promote_pair(cev, value, newbits, call)) return 0;
    if (offset + count > 32) {
        cev_error(cev, call,
            "builtin 'insertBits' domain error: offset + count must be <= 32");
        return 0;
    }
    uint32_t v = (uint32_t)value->u.i;
    uint32_t nb = (uint32_t)newbits->u.i;
    uint32_t r = v;
    if (count != 0) {
        uint32_t low = count == 32 ? UINT32_MAX : ((1u << count) - 1u);
        uint32_t mask = low << offset;
        r = (v & ~mask) | ((nb << offset) & mask);
    }
    out->kind = WGSL_VAL_INT;
    out->type = value->type;
    out->u.i = value->type == cev->types->t_u32
        ? (int64_t)r : (int64_t)(int32_t)r;
    return 1;
}

static int eval_call_bitfield_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_insert = name_eq(nm, len, "insertBits");
    int is_extract = name_eq(nm, len, "extractBits");
    if (!is_insert && !is_extract) return 0;
    uint32_t expected = is_insert ? 5 : 4;
    if (call->child_count != expected) {
        cev_error(cev, call,
            "builtin '%.*s' expects %u arguments; got %u",
            (int)len, nm, (unsigned)(expected - 1),
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue value = {0}, newbits = {0}, offset_v = {0}, count_v = {0};
    if (!eval_expr(cev, call->children[1], &value)) return -1;
    if (is_insert && !eval_expr(cev, call->children[2], &newbits)) return -1;
    uint32_t offset_idx = is_insert ? 3u : 2u;
    if (!eval_expr(cev, call->children[offset_idx], &offset_v) ||
        !eval_expr(cev, call->children[offset_idx + 1u], &count_v))
    {
        return -1;
    }
    uint32_t offset = 0, count = 0;
    if (!read_u32_const_arg(cev, call, nm, len, &offset_v, &offset) ||
        !read_u32_const_arg(cev, call, nm, len, &count_v, &count))
    {
        return -1;
    }

    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!bitfield_shape(cev, call, nm, len, &value,
                        is_insert ? &newbits : NULL,
                        &width, &vec_type))
    {
        return -1;
    }

    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue v = value_component(&value, i);
        if (is_insert) {
            WGSLValue nb = value_component(&newbits, i);
            if (!fold_insert_bits_scalar(
                    cev, call, &v, &nb, offset, count, &tmp[i]))
            {
                return -1;
            }
        } else if (!fold_extract_bits_scalar(
                       cev, call, &v, offset, count, &tmp[i]))
        {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static uint32_t reverse32(uint32_t x) {
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x >> 4) & 0x0f0f0f0fu);
    x = ((x & 0x00ff00ffu) << 8) | ((x >> 8) & 0x00ff00ffu);
    return (x << 16) | (x >> 16);
}

static uint32_t popcount32(uint32_t x) {
    uint32_t c = 0;
    while (x) {
        c += x & 1u;
        x >>= 1;
    }
    return c;
}

static uint32_t clz32(uint32_t x) {
    if (x == 0) return 32;
    uint32_t n = 0;
    for (int i = 31; i >= 0 && ((x & (1u << i)) == 0); i--) n++;
    return n;
}

static uint32_t ctz32(uint32_t x) {
    if (x == 0) return 32;
    uint32_t n = 0;
    while ((x & 1u) == 0) {
        n++;
        x >>= 1;
    }
    return n;
}

static int fold_integer_bits_scalar(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *v, WGSLValue *out)
{
    if (v->kind != WGSL_VAL_INT) {
        cev_error(cev, call,
            "builtin '%.*s' requires an integer argument", (int)len, nm);
        return 0;
    }
    if (v->type == cev->types->t_abstract_int &&
        !wgsl_consteval_materialize(cev, v, cev->types->t_i32, call))
    {
        return 0;
    }
    uint32_t x = (uint32_t)v->u.i;
    uint32_t r = 0;
    if (name_eq(nm, len, "countOneBits")) {
        r = popcount32(x);
    } else if (name_eq(nm, len, "countLeadingZeros")) {
        r = clz32(x);
    } else if (name_eq(nm, len, "countTrailingZeros")) {
        r = ctz32(x);
    } else if (name_eq(nm, len, "reverseBits")) {
        r = reverse32(x);
    } else if (name_eq(nm, len, "firstTrailingBit")) {
        r = x == 0 ? UINT32_MAX : ctz32(x);
    } else if (name_eq(nm, len, "firstLeadingBit")) {
        uint32_t y = (v->type == cev->types->t_i32 && (int32_t)x < 0)
            ? ~x : x;
        r = y == 0 ? UINT32_MAX : (31u - clz32(y));
    } else {
        return 0;
    }

    out->kind = WGSL_VAL_INT;
    out->type = v->type;
    out->u.i = v->type == cev->types->t_u32
        ? (int64_t)r : (int64_t)(int32_t)r;
    return 1;
}

static int eval_call_integer_bits_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int supported =
        name_eq(nm, len, "countOneBits") ||
        name_eq(nm, len, "countLeadingZeros") ||
        name_eq(nm, len, "countTrailingZeros") ||
        name_eq(nm, len, "reverseBits") ||
        name_eq(nm, len, "firstLeadingBit") ||
        name_eq(nm, len, "firstTrailingBit");
    if (!supported) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue v = value_component(&arg, i);
        if (!fold_integer_bits_scalar(cev, call, nm, len, &v, &tmp[i])) {
            return -1;
        }
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_select_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (!name_eq(nm, len, "select")) return 0;
    if (call->child_count != 4) {
        cev_error(cev, call,
            "builtin 'select' expects 3 arguments; got %u",
            (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue vals[3] = {{0}};
    if (!eval_expr(cev, call->children[1], &vals[0]) ||
        !eval_expr(cev, call->children[2], &vals[1]) ||
        !eval_expr(cev, call->children[3], &vals[2]))
    {
        return -1;
    }
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, vals, 2, -1,
                             &width, &vec_type))
    {
        return -1;
    }
    int cond_vec = vals[2].kind == WGSL_VAL_VEC;
    if (cond_vec && vals[2].u.agg.count != width) {
        cev_error(cev, call, "builtin 'select' condition shape mismatch");
        return -1;
    }
    if (!cond_vec && vals[2].kind != WGSL_VAL_BOOL) {
        cev_error(cev, call, "builtin 'select' requires bool condition");
        return -1;
    }
    WGSLValue tmp[4] = {{0}};
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue f = value_component(&vals[0], i);
        WGSLValue t = value_component(&vals[1], i);
        if (!promote_pair(cev, &f, &t, call)) return -1;
        WGSLValue c = cond_vec ? vals[2].u.agg.elems[i] : vals[2];
        if (c.kind != WGSL_VAL_BOOL) {
            cev_error(cev, call, "builtin 'select' requires bool condition");
            return -1;
        }
        tmp[i] = c.u.b ? t : f;
    }
    return make_componentwise_result(cev, call, width, vec_type, tmp, out) ? 1 : -1;
}

static int eval_call_decompose_float_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_frexp = name_eq(nm, len, "frexp");
    int is_modf = name_eq(nm, len, "modf");
    if (!is_frexp && !is_modf) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    uint32_t width = 1;
    WGSLTypeInfo *vec_type = NULL;
    if (!componentwise_shape(cev, call, nm, len, &arg, 1, -1,
                             &width, &vec_type))
    {
        return -1;
    }

    WGSLValue fract_tmp[4] = {{0}};
    WGSLValue aux_tmp[4] = {{0}};
    WGSLTypeInfo *float_type = NULL;
    for (uint32_t i = 0; i < width; i++) {
        WGSLValue x = value_component(&arg, i);
        if (!require_float_scalar(cev, call, nm, len, &x)) return -1;
        if (!float_type) float_type = x.type;
        if (!wgsl_consteval_materialize(cev, &x, float_type, call)) return -1;

        if (is_frexp) {
            int exp = 0;
            double frac = frexp(x.u.f, &exp);
            if (!finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, frac, &fract_tmp[i]))
            {
                return -1;
            }
            aux_tmp[i].kind = WGSL_VAL_INT;
            aux_tmp[i].type = cev->types->t_i32;
            aux_tmp[i].u.i = exp;
        } else {
            double whole = trunc(x.u.f);
            double frac = x.u.f - whole;
            if (!finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, frac, &fract_tmp[i]) ||
                !finish_float_builtin_result_relaxed_range(
                    cev, call, nm, len, float_type, whole, &aux_tmp[i]))
            {
                return -1;
            }
        }
    }

    WGSLValue fields[2] = {{0}};
    WGSLTypeInfo *result_shape = float_type;
    if (vec_type) {
        if (!make_vector_result(cev, call, width, float_type, fract_tmp, &fields[0])) {
            return -1;
        }
        WGSLTypeInfo *aux_elem = is_frexp ? cev->types->t_i32 : float_type;
        if (!make_vector_result(cev, call, width, aux_elem, aux_tmp, &fields[1])) {
            return -1;
        }
        result_shape = fields[0].type;
    } else {
        fields[0] = fract_tmp[0];
        fields[1] = aux_tmp[0];
    }

    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * 2);
    if (!stored) {
        cev_error(cev, call, "out of memory while folding builtin '%.*s'",
                  (int)len, nm);
        return -1;
    }
    stored[0] = fields[0];
    stored[1] = fields[1];
    out->kind = WGSL_VAL_STRUCT;
    out->type = is_frexp
        ? wgsl_type_frexp_result(cev->types, result_shape)
        : wgsl_type_modf_result(cev->types, result_shape);
    out->u.agg.count = 2;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_call_any_all_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int is_any = name_eq(nm, len, "any");
    int is_all = name_eq(nm, len, "all");
    if (!is_any && !is_all) return 0;
    if (call->child_count != 2) {
        cev_error(cev, call,
            "builtin '%.*s' expects 1 argument; got %u",
            (int)len, nm, (unsigned)(call->child_count - 1));
        return -1;
    }
    WGSLValue arg = {0};
    if (!eval_expr(cev, call->children[1], &arg)) return -1;
    if (arg.kind == WGSL_VAL_BOOL) {
        out->kind = WGSL_VAL_BOOL;
        out->type = cev->types->t_bool;
        out->u.b = arg.u.b;
        return 1;
    }
    if (arg.kind != WGSL_VAL_VEC || !arg.type ||
        arg.type->kind != WGSL_TYPE_VEC ||
        (WGSLTypeInfo *)arg.type->ref != cev->types->t_bool)
    {
        cev_error(cev, call,
            "builtin '%.*s' requires a vecN<bool> argument", (int)len, nm);
        return -1;
    }
    bool r = is_all ? true : false;
    for (uint32_t i = 0; i < arg.u.agg.count; i++) {
        WGSLValue c = arg.u.agg.elems[i];
        if (c.kind != WGSL_VAL_BOOL) {
            cev_error(cev, call,
                "builtin '%.*s' requires a vecN<bool> argument", (int)len, nm);
            return -1;
        }
        if (is_all) r = r && c.u.b;
        else        r = r || c.u.b;
    }
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    out->u.b = r;
    return 1;
}

static int eval_call_const_builtin(
    WGSLConstEvaluator *cev, WGSLNode *call,
    const char *nm, uint32_t len, WGSLValue *out)
{
    int folded = eval_call_bitcast_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_pack_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_unpack_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_dot4_packed_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_any_all_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_unary_float_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_abs_sign_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_binary_numeric_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_ternary_numeric_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_ldexp_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_length_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_distance_dot_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_cross_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_normalize_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_reflect_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_face_forward_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_refract_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_transpose_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_determinant_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_bitfield_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_integer_bits_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    folded = eval_call_select_builtin(cev, call, nm, len, out);
    if (folded) return folded;
    return eval_call_decompose_float_builtin(cev, call, nm, len, out);
}

static int eval_call(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (!n || n->child_count == 0) return 0;
    WGSLNode *callee = n->children[0];
    const char *nm = NULL;
    uint32_t len = 0;
    WGSLSymbol *sym = callee_head_symbol(callee);
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_FN) {
        WGSLNode *name_node =
            callee->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT &&
            callee->child_count >= 1 ? callee->children[0] : callee;
        if (!node_ident_text(cev, name_node, &nm, &len)) {
            cev_error(cev, n, "malformed builtin call");
            return 0;
        }
        int folded = eval_call_const_builtin(cev, n, nm, len, out);
        if (folded > 0) return 1;
        if (folded < 0) return 0;
        cev_error(cev, n,
            "builtin '%.*s' const-eval not implemented yet (Iter C)",
            (int)len, nm);
        return 0;
    }
    WGSLTypeInfo *target = constructor_target_type(cev, callee);
    if (target) return eval_constructor_call(cev, n, target, out);
    if (sym && sym->kind == WGSL_SYM_PREDECLARED_TYPEGEN &&
        callee->kind == WGSL_NODE_EXPR_IDENT)
    {
        if (eval_inferred_typegen_constructor(cev, n, out)) return 1;
        if (!cev->had_error) {
            cev_error(cev, n, "constructor const-eval not implemented yet (Iter C)");
        }
        return 0;
    }
    cev_error(cev, n, "constructor const-eval not implemented yet (Iter C)");
    return 0;
}

static int eval_index(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed index expression");
        return 0;
    }
    WGSLValue base = {0}, idx = {0};
    if (!eval_expr(cev, n->children[0], &base)) return 0;
    if (!eval_expr(cev, n->children[1], &idx)) return 0;
    if (idx.kind != WGSL_VAL_INT) {
        cev_error(cev, n->children[1], "constant index must be an integer");
        return 0;
    }
    if (idx.u.i < 0 || idx.u.i > UINT32_MAX) {
        cev_error(cev, n->children[1], "constant index is out of range");
        return 0;
    }
    uint32_t i = (uint32_t)idx.u.i;
    if (base.kind == WGSL_VAL_VEC || base.kind == WGSL_VAL_ARRAY) {
        if (i >= base.u.agg.count) {
            cev_error(cev, n->children[1], "constant index is out of bounds");
            return 0;
        }
        *out = base.u.agg.elems[i];
        return 1;
    }
    if (base.kind == WGSL_VAL_MAT) {
        if (!base.type || base.type->kind != WGSL_TYPE_MAT) return 0;
        if (i >= base.type->width) {
            cev_error(cev, n->children[1], "constant index is out of bounds");
            return 0;
        }
        uint32_t rows = base.type->rows;
        WGSLValue *elems = (WGSLValue *)wgsl_arena_alloc(
            cev->arena, sizeof *elems * rows);
        if (!elems) {
            cev_error(cev, n, "out of memory while folding matrix index");
            return 0;
        }
        for (uint32_t r = 0; r < rows; r++) {
            elems[r] = base.u.agg.elems[i * rows + r];
        }
        out->kind = WGSL_VAL_VEC;
        out->type = wgsl_type_vec(
            cev->types, (uint8_t)rows, (WGSLTypeInfo *)base.type->ref);
        out->u.agg.count = rows;
        out->u.agg.elems = elems;
        return 1;
    }
    cev_error(cev, n, "constant index requires vector, matrix, or array");
    return 0;
}

static int swizzle_index(char c, int *set) {
    switch (c) {
    case 'x': *set = 0; return 0;
    case 'y': *set = 0; return 1;
    case 'z': *set = 0; return 2;
    case 'w': *set = 0; return 3;
    case 'r': *set = 1; return 0;
    case 'g': *set = 1; return 1;
    case 'b': *set = 1; return 2;
    case 'a': *set = 1; return 3;
    default: return -1;
    }
}

static int eval_vector_member(
    WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (len == 0 || len > 4) {
        cev_error(cev, n, "invalid vector swizzle");
        return 0;
    }
    int set0 = -1;
    WGSLValue elems[4] = {{0}};
    for (uint32_t i = 0; i < len; i++) {
        int set = -1;
        int idx = swizzle_index(nm[i], &set);
        if (idx < 0 || (uint32_t)idx >= base->u.agg.count) {
            cev_error(cev, n, "invalid vector swizzle");
            return 0;
        }
        if (set0 < 0) set0 = set;
        if (set != set0) {
            cev_error(cev, n, "vector swizzle must not mix component sets");
            return 0;
        }
        elems[i] = base->u.agg.elems[idx];
    }
    if (len == 1) {
        *out = elems[0];
        return 1;
    }
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(
        cev->arena, sizeof *stored * len);
    if (!stored) {
        cev_error(cev, n, "out of memory while folding vector swizzle");
        return 0;
    }
    for (uint32_t i = 0; i < len; i++) stored[i] = elems[i];
    out->kind = WGSL_VAL_VEC;
    out->type = wgsl_type_vec(
        cev->types, (uint8_t)len, (WGSLTypeInfo *)base->type->ref);
    out->u.agg.count = len;
    out->u.agg.elems = stored;
    return 1;
}

static int eval_struct_member(
    WGSLConstEvaluator *cev, WGSLNode *n, const WGSLValue *base,
    const char *nm, uint32_t len, WGSLValue *out)
{
    if (base->type && base->type->kind == WGSL_TYPE_FREXP_RESULT) {
        if (len == 5 && memcmp(nm, "fract", 5) == 0 &&
            base->u.agg.count >= 1)
        {
            *out = base->u.agg.elems[0];
            return 1;
        }
        if (len == 3 && memcmp(nm, "exp", 3) == 0 &&
            base->u.agg.count >= 2)
        {
            *out = base->u.agg.elems[1];
            return 1;
        }
        cev_error(cev, n, "__frexp_result has no such member");
        return 0;
    }
    if (base->type && base->type->kind == WGSL_TYPE_MODF_RESULT) {
        if (len == 5 && memcmp(nm, "fract", 5) == 0 &&
            base->u.agg.count >= 1)
        {
            *out = base->u.agg.elems[0];
            return 1;
        }
        if (len == 5 && memcmp(nm, "whole", 5) == 0 &&
            base->u.agg.count >= 2)
        {
            *out = base->u.agg.elems[1];
            return 1;
        }
        cev_error(cev, n, "__modf_result has no such member");
        return 0;
    }
    if (!base->type || base->type->kind != WGSL_TYPE_STRUCT) return 0;
    WGSLNode *sd = (WGSLNode *)base->type->ref;
    uint32_t idx = 0;
    for (uint32_t i = 0; sd && i < sd->child_count; i++) {
        WGSLNode *m = sd->children[i];
        if (!m || m->kind != WGSL_NODE_DECL_STRUCT_MEMBER) continue;
        uint32_t off = (uint32_t)(m->payload[0] & 0xFFFFFFFFu);
        uint32_t ml = (uint32_t)(m->payload[0] >> 32);
        if (ml == len && cev->src && off + ml <= cev->src->length &&
            memcmp(cev->src->bytes + off, nm, len) == 0)
        {
            if (idx >= base->u.agg.count) return 0;
            *out = base->u.agg.elems[idx];
            return 1;
        }
        idx++;
    }
    cev_error(cev, n, "unknown struct member in constant expression");
    return 0;
}

static int eval_member(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out) {
    if (n->child_count != 2 || !n->children[0] || !n->children[1]) {
        cev_error(cev, n, "malformed member expression");
        return 0;
    }
    WGSLValue base = {0};
    if (!eval_expr(cev, n->children[0], &base)) return 0;
    const char *nm = NULL;
    uint32_t len = 0;
    if (!node_ident_text(cev, n->children[1], &nm, &len)) {
        cev_error(cev, n->children[1], "malformed member name");
        return 0;
    }
    if (base.kind == WGSL_VAL_VEC) {
        return eval_vector_member(cev, n, &base, nm, len, out);
    }
    if (base.kind == WGSL_VAL_STRUCT) {
        return eval_struct_member(cev, n, &base, nm, len, out);
    }
    cev_error(cev, n, "constant member access requires vector or struct");
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
            /* Lazy evaluation for module-scope consts */
            if (sym->ast && sym->ast->kind == WGSL_NODE_DECL_CONST) {
                if (sym->ast->flags & WGSL_FLAG_CONST_EVALING) {
                    cev_error(cev, n, "cyclic dependency detected for '%.*s'",
                              (int)sym->name_len, sym->name);
                    return 0;
                }
                sym->ast->flags |= WGSL_FLAG_CONST_EVALING;
                int ok = eval_decl_const(cev, sym->ast);
                sym->ast->flags &= ~WGSL_FLAG_CONST_EVALING;
                if (!ok) return 0;
                bound = wgsl_consteval_value_of(cev, sym);
            }
        }
        if (!bound) {
            cev_error(cev, n,
                "'%.*s' is not a constant",
                (int)sym->name_len, sym->name);
            return 0;
        }
        *out = *bound;
        return 1;
    }

    case WGSL_NODE_EXPR_TEMPLATED_IDENT:
        cev_error(cev, n,
            "constructor const-eval not implemented yet (Iter C)");
        return 0;
    case WGSL_NODE_EXPR_CALL:
        return eval_call(cev, n, out);
    case WGSL_NODE_EXPR_INDEX:
        return eval_index(cev, n, out);
    case WGSL_NODE_EXPR_MEMBER:
        return eval_member(cev, n, out);

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

/* Resolve the constructible type-specifier shapes consteval needs before
 * the type-checker pre-pass has populated every user alias / struct type. */
static WGSLTypeInfo *type_from_typespec(
    const WGSLConstEvaluator *cev, const WGSLNode *tnode)
{
    if (!tnode) return NULL;
    if (tnode->kind == WGSL_NODE_EXPR_TEMPLATED_IDENT) {
        if (tnode->child_count == 0 || !tnode->children[0]) return NULL;
        const WGSLNode *head = tnode->children[0];
        if (head->kind != WGSL_NODE_EXPR_IDENT) return NULL;
        WGSLSymbol *sym = wgsl_node_resolved_symbol((WGSLNode *)head);
        if (!sym) return NULL;
        if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
            sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
        {
            return sym->type;
        }
        if (sym->kind != WGSL_SYM_PREDECLARED_TYPEGEN) return NULL;
        const char *nm = sym->name;
        uint32_t len = sym->name_len;
        int width = vector_width_from_name(nm, len);
        if (width) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            return elem ? wgsl_type_vec(cev->types, (uint8_t)width, elem) : NULL;
        }
        uint8_t cols = 0, rows = 0;
        if (matrix_dims_from_name(nm, len, &cols, &rows)) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            return elem ? wgsl_type_mat(cev->types, cols, rows, elem) : NULL;
        }
        if (len == 5 && memcmp(nm, "array", 5) == 0) {
            if (tnode->child_count < 2) return NULL;
            WGSLTypeInfo *elem = type_from_typespec(cev, tnode->children[1]);
            if (!elem) return NULL;
            uint32_t length = 0;
            if (tnode->child_count >= 3) {
                WGSLValue v = {0};
                if (!eval_expr((WGSLConstEvaluator *)cev,
                               tnode->children[2], &v) ||
                    v.kind != WGSL_VAL_INT || v.u.i <= 0)
                {
                    return NULL;
                }
                length = (uint32_t)v.u.i;
            }
            return wgsl_type_array(cev->types, elem, length);
        }
        return NULL;
    }
    if (tnode->kind != WGSL_NODE_EXPR_IDENT) return NULL;
    WGSLSymbol *sym = wgsl_node_resolved_symbol((WGSLNode *)tnode);
    if (!sym) return NULL;
    if (sym->kind == WGSL_SYM_PREDECLARED_TYPE ||
        sym->kind == WGSL_SYM_PREDECLARED_TYPE_ALIAS)
    {
        return sym->type;
    }
    if (sym->kind == WGSL_SYM_TYPE_ALIAS) {
        if (sym->type) return sym->type;
        if (!sym->ast || sym->ast->kind != WGSL_NODE_DECL_ALIAS ||
            sym->ast->child_count == 0)
        {
            return NULL;
        }
        if (sym->ast->flags & WGSL_FLAG_TYPING) return NULL;
        sym->ast->flags |= WGSL_FLAG_TYPING;
        WGSLTypeInfo *t = type_from_typespec(cev, sym->ast->children[0]);
        sym->ast->flags &= ~WGSL_FLAG_TYPING;
        if (t) sym->type = t;
        return t;
    }
    if (sym->kind == WGSL_SYM_STRUCT) {
        if (sym->type) return sym->type;
        if (!sym->ast || sym->ast->kind != WGSL_NODE_DECL_STRUCT) return NULL;
        WGSLTypeInfo *st = (WGSLTypeInfo *)wgsl_arena_calloc(
            cev->types->arena, 1, sizeof *st);
        if (!st) return NULL;
        st->kind = (uint16_t)WGSL_TYPE_STRUCT;
        st->ref = sym->ast;
        sym->type = st;
        return st;
    }
    return NULL;
}

/* DECL_CONST children: [type?][init].  payload[1] low bit = has_type.
 *
 * An untyped `const` keeps the initializer's raw type.  In particular,
 * `const C = 1;` binds `C` as AbstractInt, not i32; materialization is
 * a use-site concern.  A typed `const` still materializes to its declared
 * type so value-side range and conversion diagnostics fire here. */
static int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n) {
    int has_type = (int)(n->payload[1] & 1u);
    WGSLSymbol *sym = find_decl_symbol(cev, n);
    if (sym && wgsl_consteval_value_of(cev, sym)) return 1;

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
    if (target && !wgsl_consteval_materialize(cev, &v, target, init)) {
        return 0;
    }

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

/* const_assert evaluation is deferred to the validator (Phase 8) so the
 * type checker (Phase 7) can enforce the boolean type constraint first. */

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
        /* Deferred to Phase 8 */
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
        case WGSL_NODE_DECL_CONST_ASSERT: /* Deferred to Phase 8 */ break;
        case WGSL_NODE_DECL_FUNCTION:     walk_function     (out, n); break;
        default: break;
        }
    }
    return out->had_error ? 0 : 1;
}
