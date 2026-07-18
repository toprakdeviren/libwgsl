/**
 * @file json_buf.c — JSON buffer + type/attr helpers for module reflection
 */
#include "internal/module_json_priv.h"

void jb_grow(JsonBuf *j, size_t need) {
    if (j->oom) return;
    if (need > SIZE_MAX - j->len || j->len + need == SIZE_MAX) {
        j->oom = 1;
        return;
    }
    size_t want = j->len + need + 1;
    if (want <= j->cap) return;
    size_t cap = j->cap ? j->cap : 256;
    while (cap < want) {
        if (cap > SIZE_MAX / 2) {
            cap = want;
            break;
        }
        cap *= 2;
    }
    char *g = (char *)realloc(j->buf, cap);
    if (!g) { j->oom = 1; return; }
    j->buf = g;
    j->cap = cap;
}

void jb_putc(JsonBuf *j, char c) {
    jb_grow(j, 1);
    if (j->oom) return;
    j->buf[j->len++] = c;
}

void jb_putn(JsonBuf *j, const char *s, size_t n) {
    jb_grow(j, n);
    if (j->oom) return;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
}

void jb_puts(JsonBuf *j, const char *s) {
    jb_putn(j, s, strlen(s));
}

void jb_put_int(JsonBuf *j, long long v) {
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%lld", v);
    if (n > 0) jb_putn(j, tmp, (size_t)n);
}

/* JSON-escape a byte range. */
void jb_put_jstr(JsonBuf *j, const char *s, size_t n) {
    jb_putc(j, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  jb_putn(j, "\\\"", 2); break;
        case '\\': jb_putn(j, "\\\\", 2); break;
        case '\b': jb_putn(j, "\\b", 2);  break;
        case '\f': jb_putn(j, "\\f", 2);  break;
        case '\n': jb_putn(j, "\\n", 2);  break;
        case '\r': jb_putn(j, "\\r", 2);  break;
        case '\t': jb_putn(j, "\\t", 2);  break;
        default:
            if (c < 0x20) {
                char tmp[8];
                int k = snprintf(tmp, sizeof tmp, "\\u%04x", c);
                if (k > 0) jb_putn(j, tmp, (size_t)k);
            } else {
                jb_putc(j, (char)c);
            }
        }
    }
    jb_putc(j, '"');
}

/* Convenience: emit a JSON string from a (offset, length) source span. */
void jb_put_span(JsonBuf *j, const WGSLSource *src,
                        uint32_t off, uint32_t len)
{
    if (!src || !src->bytes ||
        len > UINT32_MAX - off ||
        (size_t)off + (size_t)len > src->length)
    {
        jb_puts(j, "\"\"");
        return;
    }
    jb_put_jstr(j, src->bytes + off, len);
}

/* Format a TypeInfo into the JSON buffer as a quoted string. */
void jb_put_type(JsonBuf *j, WGSLTypeInfo *t) {
    if (!t) { jb_puts(j, "\"\""); return; }
    char tmp[128];
    size_t n = wgsl_type_format(t, tmp, sizeof tmp);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    jb_put_jstr(j, tmp, n);
}

/* Resource-facing type name.  For a struct-typed binding, emit the
 * declared struct name (e.g. "Uniforms") instead of the generic
 * "struct" tag `jb_put_type` would produce — embedders key their
 * uniform/storage layout off the "structs" table by name, so the
 * binding must name its struct.  For a struct type, `ref` is the
 * DECL_STRUCT AST node and `payload[0]` holds its name span. */
void jb_put_resource_type(WGSLResult *r, JsonBuf *j, WGSLTypeInfo *t) {
    if (t && t->kind == (uint16_t)WGSL_TYPE_STRUCT && t->ref) {
        WGSLNode *sn = (WGSLNode *)t->ref;
        uint32_t no = wgsl_node_name_span(sn).offset;
        uint32_t nl = wgsl_node_name_span(sn).length;
        jb_put_span(j, &r->source, no, nl);
        return;
    }
    jb_put_type(j, t);
}

/* Resolve an attribute argument (an EXPR_*) to an integer.  Returns 0
 * if the arg isn't a const-eval'd integer; uses the result's existing
 * `WGSLConstEvaluator` (no second pass). */
static int wgsl_attr_int_arg_maybe(WGSLResult *r, WGSLNode *arg, long long *out) {
    if (!r || !arg || !out) return 0;
    size_t diag_count = r->diag.count;
    int diag_errors = r->diag.error_count;
    int cev_error = r->cev.store.had_error;
    WGSLValue v = {0};
    if (!wgsl_consteval_expr(&r->cev, arg, &v) || v.kind != WGSL_VAL_INT) {
        r->diag.count = diag_count;
        r->diag.error_count = diag_errors;
        r->cev.store.had_error = cev_error;
        return 0;
    }
    *out = (long long)v.u.i;
    return 1;
}

long long wgsl_attr_int_arg(WGSLResult *r, WGSLNode *arg) {
    long long v = 0;
    (void)wgsl_attr_int_arg_maybe(r, arg, &v);
    return v;
}

void jb_put_wgs_dim(WGSLResult *r, JsonBuf *j, WGSLNode *arg) {
    long long v = 0;
    if (!arg) {
        jb_put_int(j, 1);
    } else if (wgsl_attr_int_arg_maybe(r, arg, &v)) {
        jb_put_int(j, v);
    } else {
        jb_put_span(j, &r->source, arg->span_offset, arg->span_length);
    }
}

/* Look up an attribute on `[parent->children + first .. + first+count)`
 * by name.  Returns the ATTRIBUTE node or NULL. */
WGSLNode *json_find_attr(
    WGSLResult *r, const WGSLNode *parent,
    uint32_t first, uint32_t count, const char *want)
{
    size_t wl = strlen(want);
    for (uint32_t i = 0; i < count && first + i < parent->child_count; i++) {
        WGSLNode *a = parent->children[first + i];
        if (!a || a->kind != WGSL_NODE_ATTRIBUTE) continue;
        if (a->child_count == 0) continue;
        WGSLNode *nm = a->children[0];
        if (!nm) continue;
        uint32_t off = wgsl_node_name_span(nm).offset;
        uint32_t len = wgsl_node_name_span(nm).length;
        if (len == wl &&
            memcmp(r->source.bytes + off, want, wl) == 0)
        {
            return a;
        }
    }
    return NULL;
}

const char *json_stage_for(WGSLResult *r, WGSLNode *fn,
                                  uint32_t fn_attr_count)
{
    if (json_find_attr(r, fn, 0, fn_attr_count, "vertex"))   return "vertex";
    if (json_find_attr(r, fn, 0, fn_attr_count, "fragment")) return "fragment";
    if (json_find_attr(r, fn, 0, fn_attr_count, "compute"))  return "compute";
    return NULL;
}
