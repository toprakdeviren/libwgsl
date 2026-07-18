/**
 * @file wgsl_lsp_core.c — shared storage, JSON, wire and diagnostic helpers.
 */
#include "wgsl_lsp_priv.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── growable string ─────────────────────────────────────────────── */

void lsp_sb_clear(SB *s) { s->n = 0; if (s->b) s->b[0] = 0; }

int lsp_sb_grow(SB *s, size_t need) {
    if (s->n + need + 1 <= s->cap) return 1;
    size_t nc = s->cap ? s->cap : 256;
    while (nc < s->n + need + 1) nc *= 2;
    char *g = (char *)realloc(s->b, nc);
    if (!g) return 0;
    s->b = g; s->cap = nc;
    return 1;
}

void lsp_sb_putn(SB *s, const char *p, size_t n) {
    if (!lsp_sb_grow(s, n)) return;
    memcpy(s->b + s->n, p, n);
    s->n += n;
    s->b[s->n] = 0;
}

void lsp_sb_puts(SB *s, const char *p) { if (p) lsp_sb_putn(s, p, strlen(p)); }
void lsp_sb_putc(SB *s, char c) { lsp_sb_putn(s, &c, 1); }

void lsp_sb_put_u(SB *s, unsigned long v) {
    char t[32];
    int n = snprintf(t, sizeof t, "%lu", v);
    if (n > 0) lsp_sb_putn(s, t, (size_t)n);
}

void lsp_sb_put_i(SB *s, long v) {
    char t[32];
    int n = snprintf(t, sizeof t, "%ld", v);
    if (n > 0) lsp_sb_putn(s, t, (size_t)n);
}

void lsp_sb_json_str(SB *s, const char *p) {
    lsp_sb_putc(s, '"');
    if (!p) { lsp_sb_putc(s, '"'); return; }
    for (; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':  lsp_sb_puts(s, "\\\""); break;
        case '\\': lsp_sb_puts(s, "\\\\"); break;
        case '\n': lsp_sb_puts(s, "\\n"); break;
        case '\r': lsp_sb_puts(s, "\\r"); break;
        case '\t': lsp_sb_puts(s, "\\t"); break;
        default:
            if (c < 0x20) {
                char t[8];
                snprintf(t, sizeof t, "\\u%04x", c);
                lsp_sb_puts(s, t);
            } else lsp_sb_putc(s, (char)c);
        }
    }
    lsp_sb_putc(s, '"');
}

static char *lsp_tool_strdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

/* ── document store ──────────────────────────────────────────────── */

Doc *doc_find(const char *uri) {
    for (int i = 0; i < ndocs; i++)
        if (docs[i].uri && uri && strcmp(docs[i].uri, uri) == 0)
            return &docs[i];
    return NULL;
}

Doc *doc_ensure(const char *uri) {
    Doc *d = doc_find(uri);
    if (d) return d;
    if (ndocs + 1 > cdocs) {
        int nc = cdocs ? cdocs * 2 : 4;
        Doc *g = (Doc *)realloc(docs, (size_t)nc * sizeof *g);
        if (!g) return NULL;
        docs = g; cdocs = nc;
    }
    Doc *n = &docs[ndocs++];
    memset(n, 0, sizeof *n);
    n->uri = lsp_tool_strdup(uri ? uri : "");
    return n;
}

const WGSLResult *doc_result(Doc *d) {
    if (!d || !d->session) return NULL;
    return wgsl_session_check(d->session, d->text ? d->text : "");
}

void doc_set_text(Doc *d, const char *text) {
    if (!d) return;
    free(d->text);
    d->text = lsp_tool_strdup(text ? text : "");
    d->text_len = d->text ? strlen(d->text) : 0;
    if (!d->session) d->session = wgsl_session_new();
    /* Session check runs on demand via doc_result / publish. */
    (void)doc_result(d);
}

int doc_apply_change(Doc *d, uint32_t start, uint32_t end,
                            const char *text)
{
    if (!d) return 0;
    if (!d->text) doc_set_text(d, "");
    if (!d->text) return 0;
    if (start > d->text_len) start = (uint32_t)d->text_len;
    if (end > d->text_len) end = (uint32_t)d->text_len;
    if (end < start) end = start;
    size_t ins = text ? strlen(text) : 0;
    size_t prefix = start;
    size_t suffix = d->text_len - end;
    if (prefix > SIZE_MAX - ins || prefix + ins > SIZE_MAX - suffix)
        return 0;
    char *next = (char *)malloc(prefix + ins + suffix + 1);
    if (!next) return 0;
    if (prefix) memcpy(next, d->text, prefix);
    if (ins) memcpy(next + prefix, text, ins);
    if (suffix) memcpy(next + prefix + ins, d->text + end, suffix);
    next[prefix + ins + suffix] = 0;
    free(d->text);
    d->text = next;
    d->text_len = prefix + ins + suffix;
    if (!d->session) d->session = wgsl_session_new();
    (void)doc_result(d);
    return 1;
}

void doc_close(const char *uri) {
    for (int i = 0; i < ndocs; i++) {
        if (!docs[i].uri || !uri || strcmp(docs[i].uri, uri) != 0) continue;
        free(docs[i].uri);
        free(docs[i].text);
        if (docs[i].session) wgsl_session_destroy(docs[i].session);
        docs[i] = docs[ndocs - 1];
        ndocs--;
        return;
    }
}

/* ── position helpers (LSP UTF-16 ↔ byte offset; ASCII-fast path) ── */

void offset_to_pos(const char *text, size_t len, uint32_t off,
                          unsigned *line, unsigned *ch)
{
    *line = 0; *ch = 0;
    if (!text) return;
    if (off > len) off = (uint32_t)len;
    unsigned ln = 0, col = 0;
    for (uint32_t i = 0; i < off; i++) {
        if (text[i] == '\n') { ln++; col = 0; }
        else col++; /* byte≈UTF-16 for BMP-only WGSL sources */
    }
    *line = ln; *ch = col;
}

uint32_t pos_to_offset(const char *text, size_t len,
                              unsigned line, unsigned ch)
{
    if (!text) return 0;
    unsigned ln = 0;
    size_t i = 0;
    while (i < len && ln < line) {
        if (text[i] == '\n') ln++;
        i++;
    }
    size_t start = i;
    while (i < len && text[i] != '\n' && (unsigned)(i - start) < ch) i++;
    return (uint32_t)i;
}

/* ── wire framing ────────────────────────────────────────────────── */

void send_msg(const char *body) {
    size_t n = strlen(body);
    fprintf(stdout, "Content-Length: %zu\r\n\r\n", n);
    fwrite(body, 1, n, stdout);
    fflush(stdout);
}

void reply_raw(const char *id_json, const char *result_json) {
    lsp_sb_clear(&outb);
    lsp_sb_puts(&outb, "{\"jsonrpc\":\"2.0\",\"id\":");
    lsp_sb_puts(&outb, id_json ? id_json : "null");
    lsp_sb_puts(&outb, ",\"result\":");
    lsp_sb_puts(&outb, result_json ? result_json : "null");
    lsp_sb_putc(&outb, '}');
    send_msg(outb.b ? outb.b : "{}");
}

void reply_null(const char *id_json) { reply_raw(id_json, "null"); }

void notify(const char *method, const char *params_json) {
    lsp_sb_clear(&outb);
    lsp_sb_puts(&outb, "{\"jsonrpc\":\"2.0\",\"method\":");
    lsp_sb_json_str(&outb, method);
    lsp_sb_puts(&outb, ",\"params\":");
    lsp_sb_puts(&outb, params_json ? params_json : "{}");
    lsp_sb_putc(&outb, '}');
    send_msg(outb.b ? outb.b : "{}");
}

/* ── tiny JSON helpers (subset; good enough for LSP requests) ───── */

const char *json_skip_ws(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p;
}

/* Find `"key"` then `:` and return pointer to value start. */
const char *json_val(const char *json, const char *key) {
    if (!json || !key) return NULL;
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = json;
    for (;;) {
        p = strstr(p, pat);
        if (!p) return NULL;
        /* crude: ensure previous non-ws is not a backslash-ish letter;
         * good enough for our generated/client messages. */
        const char *v = p + strlen(pat);
        v = json_skip_ws(v);
        if (*v == ':') return json_skip_ws(v + 1);
        p += strlen(pat);
    }
}

int json_get_string(const char *json, const char *key, char *out, size_t cap)
{
    const char *v = json_val(json, key);
    if (!v || *v != '"' || !out || cap == 0) return 0;
    v++;
    size_t n = 0;
    while (*v && *v != '"' && n + 1 < cap) {
        if (*v == '\\' && v[1]) {
            v++;
            char c = *v++;
            if (c == 'n') out[n++] = '\n';
            else if (c == 't') out[n++] = '\t';
            else if (c == 'r') out[n++] = '\r';
            else if (c == '"' || c == '\\' || c == '/') out[n++] = c;
            else if (c == 'u' && v[0] && v[1] && v[2] && v[3]) {
                /* skip 4 hex — emit '?' for non-ASCII; WGSL sources are ASCII */
                v += 4;
                out[n++] = '?';
            } else out[n++] = c;
        } else out[n++] = *v++;
    }
    out[n] = 0;
    return 1;
}

long json_get_int(const char *json, const char *key, long def) {
    const char *v = json_val(json, key);
    if (!v) return def;
    return strtol(v, NULL, 10);
}

/* For string ids we need the quoted JSON form in replies. */
void json_get_id_raw(const char *json, char *out, size_t cap) {
    const char *v = json_val(json, "id");
    if (!v || !out || cap < 2) { snprintf(out, cap, "null"); return; }
    if (*v == '"') {
        size_t n = 0;
        out[n++] = '"';
        v++;
        while (*v && n + 2 < cap) {
            if (*v == '\\' && v[1]) {
                out[n++] = *v++;
                out[n++] = *v++;
            } else if (*v == '"') {
                out[n++] = '"';
                break;
            } else out[n++] = *v++;
        }
        out[n] = 0;
        return;
    }
    if (strncmp(v, "null", 4) == 0) { snprintf(out, cap, "null"); return; }
    size_t n = 0;
    while ((isdigit((unsigned char)*v) || *v == '-') && n + 1 < cap)
        out[n++] = *v++;
    out[n] = 0;
    if (n == 0) snprintf(out, cap, "null");
}

/* ── diagnostics publish ─────────────────────────────────────────── */

static int sev_to_lsp(WGSLDiagSeverity s) {
    switch (s) {
    case WGSL_DIAG_ERROR:   return 1;
    case WGSL_DIAG_WARNING: return 2;
    case WGSL_DIAG_INFO:    return 3;
    case WGSL_DIAG_HINT:    return 4;
    }
    return 1;
}

static void append_lsp_diag(SB *p, int *first, const char *text, size_t text_len,
                            uint32_t off, uint32_t len, int severity,
                            const char *source, const char *message,
                            const char *code)
{
    if (!p || !first) return;
    if (off > text_len) off = (uint32_t)text_len;
    if ((size_t)off + (size_t)len > text_len)
        len = (uint32_t)(text_len - off);
    unsigned sl, sc, el, ec;
    offset_to_pos(text, text_len, off, &sl, &sc);
    offset_to_pos(text, text_len, off + len, &el, &ec);
    if (!*first) lsp_sb_putc(p, ',');
    *first = 0;
    lsp_sb_puts(p, "{\"range\":{\"start\":{\"line\":");
    lsp_sb_put_u(p, sl);
    lsp_sb_puts(p, ",\"character\":");
    lsp_sb_put_u(p, sc);
    lsp_sb_puts(p, "},\"end\":{\"line\":");
    lsp_sb_put_u(p, el);
    lsp_sb_puts(p, ",\"character\":");
    lsp_sb_put_u(p, ec);
    lsp_sb_puts(p, "}},\"severity\":");
    lsp_sb_put_i(p, severity);
    lsp_sb_puts(p, ",\"source\":");
    lsp_sb_json_str(p, source ? source : "libwgsl");
    lsp_sb_puts(p, ",\"message\":");
    lsp_sb_json_str(p, message ? message : "");
    if (code && code[0]) {
        lsp_sb_puts(p, ",\"code\":");
        lsp_sb_json_str(p, code);
    }
    lsp_sb_putc(p, '}');
}

static void append_opt_hints(Doc *d, const WGSLResult *res, SB *p, int *first) {
    if (!d || !res || !wgsl_ok(res)) return;
    char *json = wgsl_optimize_json(res);
    if (!json) return;
    const char *findings = json_val(json, "findings");
    const char *obj = findings ? strchr(findings, '{') : NULL;
    int emitted = 0;
    while (obj && emitted < 8) {
        char msg[384];
        if (!json_get_string(obj, "message", msg, sizeof msg)) break;
        long off = json_get_int(obj, "offset", 0);
        long len = json_get_int(obj, "length", 1);
        if (len < 0) len = 0;
        append_lsp_diag(p, first, d->text, d->text_len,
                        off < 0 ? 0u : (uint32_t)off,
                        (uint32_t)len, 4,
                        "libwgsl.optimize", msg, "optimization");
        emitted++;
        obj = strchr(obj + 1, '{');
    }
    wgsl_free_string(json);
}

static void append_ml_hints(Doc *d, const WGSLResult *res, SB *p, int *first) {
    if (!d || !res || !wgsl_ok(res)) return;
    char *json = wgsl_ml_analyze_json(res);
    if (!json) return;
    const char *advice = json_val(json, "advice");
    const char *q = advice ? json_skip_ws(advice + 1) : NULL;
    if (advice && *advice == '[' && q && *q != ']') {
        char *decoded = (*q == '"') ? json_decode_string_at(q) : NULL;
        const char *msg = decoded && decoded[0]
            ? decoded
            : "ML kernel analysis produced tuning advice";
        append_lsp_diag(p, first, d->text, d->text_len, 0, 0, 4,
                        "libwgsl.ml", msg, "ml.hint");
        free(decoded);
    }
    wgsl_free_string(json);
}

void publish_diags(Doc *d) {
    if (!d || !d->uri) return;
    const WGSLResult *res = doc_result(d);
    SB p; memset(&p, 0, sizeof p);
    lsp_sb_puts(&p, "{\"uri\":");
    lsp_sb_json_str(&p, d->uri);
    lsp_sb_puts(&p, ",\"diagnostics\":[");
    int n = res ? wgsl_diagnostic_count(res) : 0;
    int first = 1;
    for (int i = 0; i < n; i++) {
        const WGSLDiagnostic *dg = wgsl_diagnostic(res, i);
        if (!dg) continue;
        if (!first) lsp_sb_putc(&p, ',');
        first = 0;
        lsp_sb_puts(&p, "{\"range\":{\"start\":{\"line\":");
        lsp_sb_put_u(&p, dg->line ? dg->line - 1 : 0);
        lsp_sb_puts(&p, ",\"character\":");
        lsp_sb_put_u(&p, dg->column ? dg->column - 1 : 0);
        lsp_sb_puts(&p, "},\"end\":{\"line\":");
        lsp_sb_put_u(&p, dg->end_line ? dg->end_line - 1 : 0);
        lsp_sb_puts(&p, ",\"character\":");
        lsp_sb_put_u(&p, dg->end_column ? dg->end_column - 1 : 0);
        lsp_sb_puts(&p, "}},\"severity\":");
        lsp_sb_put_i(&p, sev_to_lsp(dg->severity));
        lsp_sb_puts(&p, ",\"source\":\"libwgsl\",\"message\":");
        lsp_sb_json_str(&p, dg->message ? dg->message : "");
        if (dg->rule && dg->rule[0]) {
            lsp_sb_puts(&p, ",\"code\":");
            lsp_sb_json_str(&p, dg->rule);
        }
        lsp_sb_putc(&p, '}');
    }
    append_opt_hints(d, res, &p, &first);
    append_ml_hints(d, res, &p, &first);
    lsp_sb_puts(&p, "]}");
    notify("textDocument/publishDiagnostics", p.b ? p.b : "{}");
    free(p.b);
}
