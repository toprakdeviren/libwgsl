/**
 * @file consteval.c — WGSL const-expression evaluator.
 *
 * Implements scalar/composite folding, materialization, builtin domain checks,
 * pure user-function evaluation, and the symbol-value cache used by later
 * frontend passes.
 */
#include "internal/consteval.h"
#include "internal/consteval_priv.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/token.h"

/* Diagnostics. */

void cev_note_status(WGSLConstEvaluator *cev, WGSLCevStatus kind) {
    if (!cev || kind == WGSL_CEV_OK) return;
    /* Escalate: VALUE_ERROR wins; otherwise keep first non-OK noise kind. */
    if (cev->store.last_status == WGSL_CEV_VALUE_ERROR) return;
    if (kind == WGSL_CEV_VALUE_ERROR || cev->store.last_status == WGSL_CEV_OK)
        cev->store.last_status = (int)kind;
}

void cev_error_kind(
    WGSLConstEvaluator *cev, WGSLCevStatus kind,
    const WGSLNode *at, const char *fmt, ...)
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
    cev->store.had_error = 1;
    cev_note_status(cev, kind);
}

void cev_error(
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
    cev->store.had_error = 1;
    cev_note_status(cev, WGSL_CEV_VALUE_ERROR);
}

/* Init / destroy. */

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
    for (size_t i = 0; i < cev->store.binding_count; i++)
        free(cev->store.bindings[i].lane_values);   /* per-lane pointer arrays */
    free(cev->store.bindings);
    free(cev->store.binding_htab);
    free(cev->store.fn_memo);                       /* arg/result bodies live in arena */
    free(cev->store.fn_memo_htab);
    memset(cev, 0, sizeof *cev);
}

/* Bindings table. */

int wgsl_consteval_sym_is_per_lane(const WGSLSymbol *sym) {
    if (!sym) return 0;
    switch (sym->kind) {
    case WGSL_SYM_PARAM:               /* function parameters                */
    case WGSL_SYM_LET:                 /* `let` — always function-scope      */
        return 1;
    case WGSL_SYM_VAR:                 /* function-local var, or var<private>;
                                        * var<storage/uniform/workgroup> shared */
        return sym->as == WGSL_AS_FUNCTION || sym->as == WGSL_AS_PRIVATE;
    default:                           /* const/override/function: uniform → shared */
        return 0;
    }
}

/* A binding is stored per-lane only in interpreter mode (nlanes >= 1) and only
 * for per-lane symbols; the const-eval path (nlanes == 0) keeps all shared. */
int binding_per_lane(const WGSLConstEvaluator *cev, const WGSLSymbol *sym) {
    return cev->simt.nlanes >= 1 && wgsl_consteval_sym_is_per_lane(sym);
}

static uint64_t binding_sym_hash(const WGSLSymbol *sym) {
    uint64_t h = (uint64_t)(uintptr_t)sym;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h;
}

static void binding_htab_put_index(WGSLConstEvaluator *cev, size_t idx) {
    if (!cev->store.binding_htab || !cev->store.binding_htab_cap ||
        idx >= cev->store.binding_count || !cev->store.bindings[idx].sym)
    {
        return;
    }
    const WGSLSymbol *sym = cev->store.bindings[idx].sym;
    size_t mask = cev->store.binding_htab_cap - 1;
    size_t slot = (size_t)binding_sym_hash(sym) & mask;
    while (cev->store.binding_htab[slot] != (size_t)-1) {
        size_t old = cev->store.binding_htab[slot];
        if (old < cev->store.binding_count && cev->store.bindings[old].sym == sym) {
            cev->store.binding_htab[slot] = idx;
            return;
        }
        slot = (slot + 1) & mask;
    }
    cev->store.binding_htab[slot] = idx;
}

static int binding_htab_reserve(WGSLConstEvaluator *cev, size_t want_count) {
    if (want_count == 0) return 1;
    if (want_count > SIZE_MAX / 10) return 0;
    size_t cap = cev->store.binding_htab_cap ? cev->store.binding_htab_cap : 32;
    while (want_count * 10 >= cap * 7) {
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    if (cap == cev->store.binding_htab_cap) return 1;

    size_t *nt = (size_t *)malloc(cap * sizeof *nt);
    if (!nt) return 0;
    for (size_t i = 0; i < cap; i++) nt[i] = (size_t)-1;
    size_t *old = cev->store.binding_htab;
    cev->store.binding_htab = nt;
    cev->store.binding_htab_cap = cap;
    for (size_t i = 0; i < cev->store.binding_count; i++)
        binding_htab_put_index(cev, i);
    free(old);
    return 1;
}

static int binding_index_of(
    const WGSLConstEvaluator *cev, const WGSLSymbol *sym)
{
    if (!cev || !sym || !cev->store.binding_htab || !cev->store.binding_htab_cap)
        return -1;
    size_t mask = cev->store.binding_htab_cap - 1;
    size_t slot = (size_t)binding_sym_hash(sym) & mask;
    for (size_t n = 0; n < cev->store.binding_htab_cap; n++) {
        size_t idx = cev->store.binding_htab[slot];
        if (idx == (size_t)-1) break;
        if (idx < cev->store.binding_count && cev->store.bindings[idx].sym == sym)
            return idx <= (size_t)INT_MAX ? (int)idx : -1;
        slot = (slot + 1) & mask;
    }
    return -1;
}

const WGSLValue *wgsl_consteval_value_of(
    const WGSLConstEvaluator *cev, const WGSLSymbol *sym)
{
    int idx = binding_index_of(cev, sym);
    if (idx < 0) return NULL;
    const WGSLConstBinding *b = &cev->store.bindings[idx];
    if (b->per_lane)
        return b->lane_values ? b->lane_values[cev->simt.cur_lane] : NULL;
    return b->value;
}

/* Deep-copy an aggregate into fresh arena storage so the binding OWNS its
 * element array.  Without this a `var local = buf;` binding would alias `buf`'s
 * elements (shallow struct copy), and a later `local[i] = …` would clobber the
 * source — and, worse, every lane's per-lane local would alias one shared array.
 * Scalars are copied trivially.  Old values stay in the arena (freed on reset). */
WGSLValue deep_copy_value(WGSLConstEvaluator *cev, const WGSLValue *v) {
    WGSLValue r = *v;
    switch (v->kind) {
    case WGSL_VAL_VEC: case WGSL_VAL_MAT:
    case WGSL_VAL_ARRAY: case WGSL_VAL_STRUCT:
        if (v->u.agg.elems && v->u.agg.count) {
            WGSLValue *e = (WGSLValue *)wgsl_arena_alloc(
                cev->arena, sizeof(WGSLValue) * v->u.agg.count);
            if (e) {
                for (uint32_t i = 0; i < v->u.agg.count; i++)
                    e[i] = deep_copy_value(cev, &v->u.agg.elems[i]);
                r.u.agg.elems = e;
            }
        }
        break;
    case WGSL_VAL_PTR:
        /* POD address (root + path + lane) — shallow copy is correct. */
        break;
    default: break;
    }
    return r;
}

/* Write `v` into binding slot `b` for the current lane (per-lane) or the
 * shared slot.  Lazily allocates the per-lane pointer array. */
int binding_write(WGSLConstEvaluator *cev, WGSLConstBinding *b,
                         const WGSLValue *v) {
    WGSLValue *stored = (WGSLValue *)wgsl_arena_alloc(cev->arena, sizeof *stored);
    if (!stored) return 0;
    *stored = deep_copy_value(cev, v);
    if (b->per_lane) {
        if (!b->lane_values) {
            b->lane_values = (WGSLValue **)calloc(cev->simt.nlanes, sizeof(WGSLValue *));
            if (!b->lane_values) return 0;
        }
        b->lane_values[cev->simt.cur_lane] = stored;
    } else {
        b->value = stored;
    }
    return 1;
}

WGSLConstBinding *append_binding(
    WGSLConstEvaluator *cev, const WGSLSymbol *sym)
{
    if (!binding_htab_reserve(cev, cev->store.binding_count + 1)) return NULL;
    if (cev->store.binding_count == cev->store.binding_capacity) {
        if (cev->store.binding_capacity > SIZE_MAX / 2) return NULL;
        size_t cap = cev->store.binding_capacity ? cev->store.binding_capacity * 2 : 16;
        if (cap > SIZE_MAX / sizeof *cev->store.bindings) return NULL;
        WGSLConstBinding *g = (WGSLConstBinding *)realloc(
            cev->store.bindings, cap * sizeof(*g));
        if (!g) return NULL;
        cev->store.bindings = g;
        cev->store.binding_capacity = cap;
    }
    WGSLConstBinding *b = &cev->store.bindings[cev->store.binding_count++];
    b->sym         = sym;
    b->per_lane    = binding_per_lane(cev, sym);
    b->value       = NULL;
    b->lane_values = NULL;
    binding_htab_put_index(cev, (size_t)(b - cev->store.bindings));
    return b;
}

int store_binding(
    WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v)
{
    WGSLConstBinding *b = append_binding(cev, sym);
    return b && binding_write(cev, b, v);
}

int wgsl_consteval_set_value(
    WGSLConstEvaluator *cev, const WGSLSymbol *sym, const WGSLValue *v)
{
    /* Mutable set: overwrite an existing binding in place, else append.
     * (The const path's `store_binding` never overwrites; the interpreter
     * needs re-binding for var/let/assignment.) */
    int idx = binding_index_of(cev, sym);
    if (idx >= 0)
        return binding_write(cev, &cev->store.bindings[idx], v);
    return store_binding(cev, sym, v);
}

/* AST -> bound symbol reverse lookup. */

WGSLSymbol *find_decl_symbol(
    const WGSLConstEvaluator *cev, const WGSLNode *decl)
{
    if (!cev || !cev->res) return NULL;
    return wgsl_resolver_symbol_for_decl(cev->res, decl);
}

/* Literal parsing. */

/* Slice the source bytes for `n`, copy into a null-terminated stack
 * buffer.  Returns the local length; truncates if too large.  WGSL
 * literal lengths are tiny (< 64 chars in practice). */
int slice_text(
    const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap)
{
    if (!src || !n) { if (cap) buf[0] = '\0'; return 0; }
    uint32_t len = n->span_length;
    if (len + 1 > cap) len = (uint32_t)(cap - 1);
    memcpy(buf, src->bytes + n->span_offset, len);
    buf[len] = '\0';
    return (int)len;
}

WGSLTypeInfo *type_for_int_suffix(
    WGSLConstEvaluator *cev, uint32_t suffix_bits)
{
    uint32_t base = suffix_bits & 0x0fu;
    switch (base) {
    case WGSL_NUM_SUFFIX_I: return cev->types->t_i32;
    case WGSL_NUM_SUFFIX_U: return cev->types->t_u32;
    default:                return cev->types->t_abstract_int;
    }
}

WGSLTypeInfo *type_for_float_suffix(
    WGSLConstEvaluator *cev, uint32_t suffix_bits)
{
    uint32_t base = suffix_bits & 0x0fu;
    switch (base) {
    case WGSL_NUM_SUFFIX_F: return cev->types->t_f32;
    case WGSL_NUM_SUFFIX_H: return cev->types->t_f16;
    default:                return cev->types->t_abstract_float;
    }
}

int parse_int_literal(
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
    out->type = type_for_int_suffix(cev, (uint32_t)wgsl_lit_raw(n));
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
void truncate_significand(char *buf, int len, int max_sig, int is_hex) {
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

int parse_float_literal(
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
    uint32_t suffix = (uint32_t)wgsl_lit_raw(n) & 0x0fu;
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
    out->type = type_for_float_suffix(cev, (uint32_t)wgsl_lit_raw(n));
    out->u.f  = v;
    return 1;
}

int eval_literal_bool(
    WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out)
{
    out->kind = WGSL_VAL_BOOL;
    out->type = cev->types->t_bool;
    out->u.b  = wgsl_lit_bool(n);
    return 1;
}

/* Predicates. */

int val_is_int(const WGSLValue *v) {
    return v->kind == WGSL_VAL_INT;
}
int val_is_float(const WGSLValue *v) {
    return v->kind == WGSL_VAL_FLOAT;
}
int val_is_numeric(const WGSLValue *v) {
    return val_is_int(v) || val_is_float(v);
}
int val_is_bool(const WGSLValue *v) {
    return v->kind == WGSL_VAL_BOOL;
}

/* Forward decls needed by recursive aggregate materialization. */
WGSLTypeInfo *type_from_typespec(
    const WGSLConstEvaluator *cev, const WGSLNode *tnode);
int eval_expr(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_expr_inner(WGSLConstEvaluator *cev, WGSLNode *n, WGSLValue *out);
int eval_decl_const(WGSLConstEvaluator *cev, WGSLNode *n);
uint32_t value_width(const WGSLValue *v);
WGSLValue value_component(const WGSLValue *v, uint32_t i);
int make_componentwise_result(
    WGSLConstEvaluator *cev, WGSLNode *call,
    uint32_t width, WGSLTypeInfo *vec_type,
    WGSLValue *elems, WGSLValue *out);

/* Materialisation. */

/* Modules under consteval modules */
