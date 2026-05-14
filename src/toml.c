/**
 * @file toml.c — minimal TOML subset parser.
 *
 * Hand-rolled recursive descent.  Tokens come from a forward
 * iterator over the source bytes; the parser allocates entries and
 * strings on the heap and frees the lot via `wgsl_toml_free`.
 *
 * Grammar lives in `internal/toml.h`.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/toml.h"

/* Local strdup — POSIX `strdup` requires `_POSIX_C_SOURCE` and that
 * macro doesn't propagate cleanly across the project's translation
 * units (especially emscripten's musl variant).  Three lines is
 * cheaper than the macro-juggling alternative. */
static char *tdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    if (n == SIZE_MAX) return NULL;
    char *o = (char *)malloc(n + 1);
    if (!o) return NULL;
    memcpy(o, s, n + 1);
    return o;
}

typedef enum {
    WGSL_TOML_STRING,
    WGSL_TOML_STRING_ARRAY,
} WGSLTomlKind;

typedef struct WGSLTomlEntry {
    char         *section;        /* "" for root */
    char         *key;
    WGSLTomlKind  kind;
    char         *str;            /* WGSL_TOML_STRING */
    char        **items;          /* WGSL_TOML_STRING_ARRAY (NULL-terminated for cheap len) */
    int           item_count;
    struct WGSLTomlEntry *next;
} WGSLTomlEntry;

struct WGSLToml {
    WGSLTomlEntry *head;
    /* Bookkeeping for cleanup: a separate "all string slabs" list
     * could be added later if we need to dedupe; for now each entry
     * owns its strings outright. */
};

/* ── Parser state ─────────────────────────────────────────────────── */

typedef struct {
    const char *src;
    size_t      len;
    size_t      pos;
    int         line;
    int         col;
    char       *err;
    size_t      err_len;
    int         had_error;
} P;

static void p_error(P *p, const char *fmt, ...) {
    if (p->had_error) return;            /* keep first error */
    p->had_error = 1;
    if (p->err && p->err_len > 0) {
        char buf[256];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        snprintf(p->err, p->err_len, "line %d col %d: %s", p->line, p->col, buf);
    }
}

static int p_eof(const P *p) { return p->pos >= p->len; }
static char p_peek(const P *p) { return p_eof(p) ? '\0' : p->src[p->pos]; }

static void p_advance(P *p) {
    if (p_eof(p)) return;
    char c = p->src[p->pos++];
    if (c == '\n') { p->line++; p->col = 1; }
    else p->col++;
}

/* Skip horizontal whitespace (space, tab) only — newlines remain
 * significant as statement terminators. */
static void p_skip_hws(P *p) {
    while (!p_eof(p)) {
        char c = p_peek(p);
        if (c == ' ' || c == '\t') p_advance(p);
        else break;
    }
}

/* Skip whitespace + comments + blank lines. */
static void p_skip_trivia(P *p) {
    while (!p_eof(p)) {
        char c = p_peek(p);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            p_advance(p);
        } else if (c == '#') {
            while (!p_eof(p) && p_peek(p) != '\n') p_advance(p);
        } else {
            break;
        }
    }
}

/* Skip to end of current line (for error recovery). */
static void p_skip_to_eol(P *p) {
    while (!p_eof(p) && p_peek(p) != '\n') p_advance(p);
}

/* Bare key: [A-Za-z0-9_-]+. */
static char *p_parse_key(P *p) {
    size_t start = p->pos;
    while (!p_eof(p)) {
        char c = p_peek(p);
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) break;
        p_advance(p);
    }
    if (p->pos == start) {
        p_error(p, "expected key");
        return NULL;
    }
    size_t n = p->pos - start;
    if (n == SIZE_MAX) {
        p_error(p, "out of memory");
        return NULL;
    }
    char *out = (char *)malloc(n + 1);
    if (!out) { p_error(p, "out of memory"); return NULL; }
    memcpy(out, p->src + start, n);
    out[n] = '\0';
    return out;
}

/* Quoted string: "…" with backslash escapes \\ \" \n \t \r \0. */
static char *p_parse_string(P *p) {
    if (p_peek(p) != '"') { p_error(p, "expected '\"'"); return NULL; }
    p_advance(p);
    /* Two-pass: count length, then allocate. */
    size_t scan_pos = p->pos;
    size_t out_len  = 0;
    while (scan_pos < p->len && p->src[scan_pos] != '"') {
        if (p->src[scan_pos] == '\n') {
            p_error(p, "newline in string literal");
            return NULL;
        }
        if (p->src[scan_pos] == '\\') {
            if (scan_pos + 1 >= p->len) {
                p_error(p, "unterminated escape");
                return NULL;
            }
            scan_pos += 2;
        } else {
            scan_pos += 1;
        }
        if (out_len == SIZE_MAX) {
            p_error(p, "out of memory");
            return NULL;
        }
        out_len += 1;
    }
    if (scan_pos >= p->len) {
        p_error(p, "unterminated string");
        return NULL;
    }
    if (out_len == SIZE_MAX) {
        p_error(p, "out of memory");
        return NULL;
    }
    char *out = (char *)malloc(out_len + 1);
    if (!out) { p_error(p, "out of memory"); return NULL; }
    size_t oi = 0;
    while (p_peek(p) != '"') {
        char c = p_peek(p);
        if (c == '\\') {
            p_advance(p);
            char e = p_peek(p);
            switch (e) {
            case '\\': out[oi++] = '\\'; break;
            case '"':  out[oi++] = '"';  break;
            case 'n':  out[oi++] = '\n'; break;
            case 't':  out[oi++] = '\t'; break;
            case 'r':  out[oi++] = '\r'; break;
            case '0':  out[oi++] = '\0'; break;
            default:
                p_error(p, "unknown escape '\\%c'", e);
                free(out);
                return NULL;
            }
            p_advance(p);
        } else {
            out[oi++] = c;
            p_advance(p);
        }
    }
    p_advance(p);   /* consume closing `"` */
    out[oi] = '\0';
    return out;
}

/* Parse a string array `["a", "b", …]`. */
static char **p_parse_string_array(P *p, int *out_count) {
    if (p_peek(p) != '[') { p_error(p, "expected '['"); return NULL; }
    p_advance(p);

    size_t cap = 4;
    size_t n   = 0;
    char **items = (char **)malloc(cap * sizeof(char *));
    if (!items) { p_error(p, "out of memory"); return NULL; }

    p_skip_trivia(p);
    if (p_peek(p) == ']') {
        p_advance(p);
        *out_count = 0;
        return items;
    }
    while (1) {
        p_skip_trivia(p);
        char *s = p_parse_string(p);
        if (!s) {
            for (size_t i = 0; i < n; i++) free(items[i]);
            free(items);
            return NULL;
        }
        if (n == cap) {
            if (cap > SIZE_MAX / 2) {
                p_error(p, "out of memory");
                free(s);
                for (size_t i = 0; i < n; i++) free(items[i]);
                free(items);
                return NULL;
            }
            size_t new_cap = cap * 2;
            if (new_cap > SIZE_MAX / sizeof(char *)) {
                p_error(p, "out of memory");
                free(s);
                for (size_t i = 0; i < n; i++) free(items[i]);
                free(items);
                return NULL;
            }
            char **grown = (char **)realloc(items, new_cap * sizeof(char *));
            if (!grown) { p_error(p, "out of memory"); free(s); for (size_t i = 0; i < n; i++) free(items[i]); free(items); return NULL; }
            cap = new_cap;
            items = grown;
        }
        items[n++] = s;
        p_skip_trivia(p);
        if (p_peek(p) == ',') {
            p_advance(p);
            p_skip_trivia(p);
            if (p_peek(p) == ']') break;     /* trailing comma */
            continue;
        }
        if (p_peek(p) == ']') break;
        p_error(p, "expected ',' or ']' in array");
        for (size_t i = 0; i < n; i++) free(items[i]);
        free(items);
        return NULL;
    }
    p_advance(p);           /* consume `]` */
    if (n > (size_t)INT32_MAX) {
        p_error(p, "too many array items");
        for (size_t i = 0; i < n; i++) free(items[i]);
        free(items);
        return NULL;
    }
    *out_count = (int)n;
    return items;
}

/* Top-level parse loop. */
static WGSLToml *p_parse(P *p) {
    WGSLToml *t = (WGSLToml *)calloc(1, sizeof *t);
    if (!t) { p_error(p, "out of memory"); return NULL; }

    char *cur_section = tdup("");
    if (!cur_section) {
        p_error(p, "out of memory");
        wgsl_toml_free(t);
        return NULL;
    }

    WGSLTomlEntry *tail = NULL;

    while (!p_eof(p)) {
        p_skip_trivia(p);
        if (p_eof(p)) break;

        if (p_peek(p) == '[') {
            p_advance(p);
            /* Disallow `[[…]]` array-of-tables for now. */
            if (p_peek(p) == '[') {
                p_error(p, "array-of-tables `[[…]]` not supported");
                goto fail;
            }
            p_skip_hws(p);
            free(cur_section);
            cur_section = p_parse_key(p);
            if (!cur_section) goto fail;
            /* Allow `[a.b]` keys but treat the dotted form as a flat
             * section name — we don't model nested tables. */
            while (p_peek(p) == '.') {
                p_advance(p);
                size_t flat_len = strlen(cur_section);
                char *more = p_parse_key(p);
                if (!more) goto fail;
                size_t add_len = strlen(more);
                if (flat_len > SIZE_MAX - 1 ||
                    flat_len + 1 > SIZE_MAX - add_len ||
                    flat_len + 1 + add_len == SIZE_MAX)
                {
                    free(more);
                    p_error(p, "out of memory");
                    goto fail;
                }
                char *grown = (char *)malloc(flat_len + 1 + add_len + 1);
                if (!grown) { free(more); p_error(p, "out of memory"); goto fail; }
                memcpy(grown, cur_section, flat_len);
                grown[flat_len] = '.';
                memcpy(grown + flat_len + 1, more, add_len + 1);
                free(more);
                free(cur_section);
                cur_section = grown;
            }
            p_skip_hws(p);
            if (p_peek(p) != ']') { p_error(p, "expected ']'"); goto fail; }
            p_advance(p);
            p_skip_hws(p);
            if (!p_eof(p) && p_peek(p) != '\n' && p_peek(p) != '#') {
                p_error(p, "trailing junk after section header");
                goto fail;
            }
            p_skip_to_eol(p);
            continue;
        }

        /* Assignment: key = value */
        char *key = p_parse_key(p);
        if (!key) goto fail;
        p_skip_hws(p);
        if (p_peek(p) != '=') {
            p_error(p, "expected '=' after key '%s'", key);
            free(key);
            goto fail;
        }
        p_advance(p);
        p_skip_hws(p);

        WGSLTomlEntry *e = (WGSLTomlEntry *)calloc(1, sizeof *e);
        if (!e) { p_error(p, "out of memory"); free(key); goto fail; }
        e->section = tdup(cur_section);
        e->key     = key;
        if (!e->section) {
            p_error(p, "out of memory");
            free(e->key);
            free(e);
            goto fail;
        }

        if (p_peek(p) == '"') {
            char *v = p_parse_string(p);
            if (!v) { free(e->section); free(e->key); free(e); goto fail; }
            e->kind = WGSL_TOML_STRING;
            e->str  = v;
        } else if (p_peek(p) == '[') {
            int n = 0;
            char **arr = p_parse_string_array(p, &n);
            if (!arr) { free(e->section); free(e->key); free(e); goto fail; }
            e->kind       = WGSL_TOML_STRING_ARRAY;
            e->items      = arr;
            e->item_count = n;
        } else {
            p_error(p, "expected string or array");
            free(e->section); free(e->key); free(e);
            goto fail;
        }

        if (tail) tail->next = e; else t->head = e;
        tail = e;

        p_skip_hws(p);
        if (!p_eof(p) && p_peek(p) != '\n' && p_peek(p) != '#') {
            p_error(p, "trailing junk after value");
            goto fail;
        }
        p_skip_to_eol(p);
    }

    free(cur_section);
    return t;

fail:
    free(cur_section);
    wgsl_toml_free(t);
    return NULL;
}

/* ── Public API ───────────────────────────────────────────────────── */

WGSLToml *wgsl_toml_parse(const char *src, size_t len, char *err, size_t err_len) {
    if (err && err_len > 0) err[0] = '\0';
    P p = { src, len, 0, 1, 1, err, err_len, 0 };
    WGSLToml *t = p_parse(&p);
    if (p.had_error) return NULL;
    return t;
}

void wgsl_toml_free(WGSLToml *t) {
    if (!t) return;
    WGSLTomlEntry *e = t->head;
    while (e) {
        WGSLTomlEntry *nx = e->next;
        free(e->section);
        free(e->key);
        if (e->kind == WGSL_TOML_STRING) {
            free(e->str);
        } else {
            for (int i = 0; i < e->item_count; i++) free(e->items[i]);
            free(e->items);
        }
        free(e);
        e = nx;
    }
    free(t);
}

static const WGSLTomlEntry *toml_find(const WGSLToml *t, const char *section, const char *key) {
    if (!t) return NULL;
    if (!section) section = "";
    for (const WGSLTomlEntry *e = t->head; e; e = e->next) {
        if (strcmp(e->section, section) == 0 && strcmp(e->key, key) == 0) return e;
    }
    return NULL;
}

const char *wgsl_toml_get_string(const WGSLToml *t, const char *section, const char *key) {
    const WGSLTomlEntry *e = toml_find(t, section, key);
    if (!e || e->kind != WGSL_TOML_STRING) return NULL;
    return e->str;
}

const char *const *wgsl_toml_get_string_array(
    const WGSLToml *t, const char *section, const char *key, int *out_count)
{
    const WGSLTomlEntry *e = toml_find(t, section, key);
    if (!e || e->kind != WGSL_TOML_STRING_ARRAY) return NULL;
    if (out_count) *out_count = e->item_count;
    return (const char *const *)e->items;
}
