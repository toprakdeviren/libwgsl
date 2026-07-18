/**
 * @file helpers.c — SB/JSON buffer, source helpers, trace recording
 */
#include "internal/interp_priv.h"

/* growable string buffer (JSON assembly). */

int sb_reserve(SB *s, size_t extra) {
    if (s->len + extra + 1 <= s->cap) return 1;
    size_t cap = s->cap ? s->cap : 256;
    while (s->len + extra + 1 > cap) {
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    char *g = (char *)realloc(s->buf, cap);
    if (!g) return 0;
    s->buf = g; s->cap = cap;
    return 1;
}
void sb_putc(SB *s, char c) {
    if (sb_reserve(s, 1)) { s->buf[s->len++] = c; s->buf[s->len] = 0; }
}
void sb_write(SB *s, const char *p, size_t n) {
    if (sb_reserve(s, n)) { memcpy(s->buf + s->len, p, n); s->len += n; s->buf[s->len] = 0; }
}
void sb_puts(SB *s, const char *p) { sb_write(s, p, strlen(p)); }
void sb_u(SB *s, unsigned long v) {
    char t[24]; int n = snprintf(t, sizeof t, "%lu", v);
    if (n > 0) sb_write(s, t, (size_t)n);
}
/* Format a double with `fmt` and append it, CLAMPING to the buffer — snprintf
 * returns the length it *would* have written, which for a large %.0f can exceed
 * the buffer; writing that many bytes would over-read the stack. */
void sb_dbl(SB *s, const char *fmt, double v) {
    char t[340];   /* worst case %.0f of ~DBL_MAX is ~309 digits */
    int n = snprintf(t, sizeof t, fmt, v);
    if (n < 0) return;
    if ((size_t)n >= sizeof t) n = (int)sizeof t - 1;
    sb_write(s, t, (size_t)n);
}
/* Append `p` as the *contents* of a JSON string (caller writes the quotes). */
void sb_json(SB *s, const char *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"':  sb_puts(s, "\\\""); break;
        case '\\': sb_puts(s, "\\\\"); break;
        case '\n': sb_puts(s, "\\n");  break;
        case '\r': sb_puts(s, "\\r");  break;
        case '\t': sb_puts(s, "\\t");  break;
        default:
            if (c < 0x20) { char e[8]; snprintf(e, sizeof e, "\\u%04x", c); sb_puts(s, e); }
            else sb_putc(s, (char)c);
        }
    }
}

/* source helpers. */

uint32_t line_of(const WGSLSource *src, const WGSLNode *n) {
    uint32_t line = 1, off = n->span_offset;
    if (off > src->length) off = (uint32_t)src->length;
    for (uint32_t i = 0; i < off; i++) if (src->bytes[i] == '\n') line++;
    return line;
}
void span_text(const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap) {
    uint32_t off = n->span_offset, len = n->span_length;
    if (len == 0 || (size_t)off + len > src->length || len >= cap) { snprintf(buf, cap, "?"); return; }
    memcpy(buf, src->bytes + off, len); buf[len] = 0;
}
void name_text(const WGSLSource *src, const WGSLNode *n, char *buf, size_t cap) {
    uint32_t off = wgsl_node_name_span(n).offset;
    uint32_t len = wgsl_node_name_span(n).length;
    if (len == 0 || (size_t)off + len > src->length || len >= cap) { snprintf(buf, cap, "?"); return; }
    memcpy(buf, src->bytes + off, len); buf[len] = 0;
}

/* Format a value textually into `s` (no surrounding quotes).  NaN/Inf are
 * spelled out because they are not valid JSON numbers. */
void fmt_value(SB *s, const WGSLValue *v) {
    char t[32];
    switch (v->kind) {
    case WGSL_VAL_BOOL: sb_puts(s, v->u.b ? "true" : "false"); break;
    case WGSL_VAL_INT:  { int n = snprintf(t, sizeof t, "%lld", (long long)v->u.i); if (n > 0) sb_write(s, t, (size_t)n); } break;
    case WGSL_VAL_FLOAT: {
        double f = v->u.f;
        if (isnan(f)) sb_puts(s, "NaN");
        else if (isinf(f)) sb_puts(s, f < 0 ? "-Inf" : "Inf");
        else { int n = snprintf(t, sizeof t, "%g", f); if (n > 0) sb_write(s, t, (size_t)n); }
        break;
    }
    case WGSL_VAL_VEC:
    case WGSL_VAL_MAT:
    case WGSL_VAL_ARRAY:
    case WGSL_VAL_STRUCT: {
        const char *tag = v->kind == WGSL_VAL_VEC ? "vec"
                        : v->kind == WGSL_VAL_MAT ? "mat"
                        : v->kind == WGSL_VAL_ARRAY ? "array" : "struct";
        sb_puts(s, tag); sb_putc(s, '(');
        uint32_t cap = v->u.agg.count > 16 ? 16 : v->u.agg.count;
        for (uint32_t i = 0; i < cap; i++) { if (i) sb_puts(s, ", "); fmt_value(s, &v->u.agg.elems[i]); }
        if (v->u.agg.count > cap) sb_puts(s, ", …");
        sb_putc(s, ')');
        break;
    }
    case WGSL_VAL_PTR: {
        sb_puts(s, "ptr(");
        if (v->u.ptr.root && v->u.ptr.root->name)
            sb_write(s, v->u.ptr.root->name, v->u.ptr.root->name_len);
        else
            sb_puts(s, "?");
        if (v->u.ptr.has_index) {
            char t2[32];
            int n = snprintf(t2, sizeof t2, "[%d]", (int)v->u.ptr.index);
            if (n > 0) sb_write(s, t2, (size_t)n);
        }
        sb_putc(s, ')');
        break;
    }
    default: sb_putc(s, '?');
    }
}

int is_true(const WGSLValue *v) {
    return (v->kind == WGSL_VAL_BOOL) ? v->u.b
         : (v->kind == WGSL_VAL_INT)  ? (v->u.i != 0)
         : (v->kind == WGSL_VAL_FLOAT)? (v->u.f != 0.0) : 0;
}

int swizzle_index_h(char c) {
    switch (c) {
    case 'x': case 'r': return 0;
    case 'y': case 'g': return 1;
    case 'z': case 'b': return 2;
    case 'w': case 'a': return 3;
    default: return -1;
    }
}

/* trace recording. */

/* Append one step object.  `anomaly` is NULL or a short tag ("Inf"/"NaN"/…). */
void record_step(Interp *ip, uint32_t line, const char *op,
                        const char *name, const WGSLValue *val, const char *anomaly) {
    if (!ip->recording) return;   /* only the focus lane's trace is serialised */
    SB *s = &ip->trace;
    if (ip->nsteps++) sb_putc(s, ',');
    sb_puts(s, "{\"line\":"); sb_u(s, line);
    sb_puts(s, ",\"op\":\""); sb_puts(s, op); sb_putc(s, '"');
    sb_puts(s, ",\"name\":\""); sb_json(s, name, strlen(name)); sb_putc(s, '"');
    if (val && val->kind != WGSL_VAL_INVALID) {
        ip->scratch.len = 0;
        fmt_value(&ip->scratch, val);
        sb_puts(s, ",\"value\":\"");
        sb_json(s, ip->scratch.buf ? ip->scratch.buf : "", ip->scratch.len);
        sb_putc(s, '"');
    } else {
        sb_puts(s, ",\"value\":null");
    }
    if (anomaly) { sb_puts(s, ",\"anomaly\":\""); sb_puts(s, anomaly); sb_putc(s, '"'); }
    else sb_puts(s, ",\"anomaly\":null");
    /* Reconvergence timeline: how many of the N lanes are executing in lockstep
     * with the focus lane at this step (dips = divergence, rises = reconverge). */
    sb_puts(s, ",\"active\":"); sb_u(s, ip->cur_active);
    /* Timeline marker: the barrier epoch this step belongs to, so the studio
     * can scrub the per-lane timeline segmented by barrier. */
    sb_puts(s, ",\"epoch\":"); sb_u(s, ip->epoch);
    sb_putc(s, '}');
}

/* Return a non-NULL anomaly tag if `v` carries a non-finite float, else NULL.
 * The nan/inf counters count only the focus invocation's anomalies (the one
 * being inspected), gated on `ip->recording`.  Recurses into vectors. */
const char *anomaly_tag(Interp *ip, const WGSLValue *v) {
    const char *tag = NULL;
    if (v->kind == WGSL_VAL_FLOAT) {
        if (isnan(v->u.f)) { if (ip->recording) ip->nan_hits++; tag = "NaN"; }
        else if (isinf(v->u.f)) { if (ip->recording) ip->inf_hits++; tag = v->u.f < 0 ? "-Inf" : "Inf"; }
    } else if (v->kind == WGSL_VAL_VEC || v->kind == WGSL_VAL_ARRAY) {
        for (uint32_t i = 0; i < v->u.agg.count; i++) {
            const char *t = anomaly_tag(ip, &v->u.agg.elems[i]);
            if (t && !tag) tag = t;
        }
    }
    return tag;
}
