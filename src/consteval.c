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


#include "consteval/materialize_ops.inc"
#include "consteval/builtin_helpers.inc"
#include "consteval/constructors.inc"
#include "consteval/bitcast_pack.inc"
#include "consteval/numeric_builtins.inc"
#include "consteval/bits_select_dispatch.inc"
#include "consteval/eval_dispatch.inc"
