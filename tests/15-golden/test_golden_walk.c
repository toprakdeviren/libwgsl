/**
 * File-walk golden corpus harness.
 *
 * Walks tests/15-golden/cases/ for each .wgsl + sibling .expect.json,
 * and verifies interpreter storage buffers against hand-reasoned
 * expectations (exact needles via wgsl_golden_run, or numeric
 * tolerance when expect.json marks known imprecision).
 *
 * Adding a case = drop NAME.wgsl + NAME.expect.json (no C edit).
 * Not part of default `make test` — run via `make test-golden`.
 */
#include "wgsl.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CASES_DIR
#define CASES_DIR "tests/15-golden/cases"
#endif

#define MAX_VALUES  64
#define MAX_BUF     8
#define MAX_NEEDLES 16

typedef struct {
    char name[64];
    int  nvals;
    char vals[MAX_VALUES][64];
} ExpBuf;

typedef struct {
    char   entry[64];
    unsigned len;
    int    expect_ok;       /* 1 true, 0 false, -1 ignore */
    int    known_numeric;   /* whitelist: allow abs_tol compare */
    double abs_tol;         /* 0 → exact needle match */
    int    nbuf;
    ExpBuf bufs[MAX_BUF];
} Expect;

static int fail_count;
static int pass_count;
static int skip_count;

/* file helpers. */

static char *slurp(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > (1 << 20)) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

static int has_suffix(const char *name, const char *suf) {
    size_t nl = strlen(name), sl = strlen(suf);
    return nl >= sl && memcmp(name + nl - sl, suf, sl) == 0;
}

/* minimal expect.json scanner (shape is fixed / small). */

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string starting at opening quote; write into out[out_sz]. */
static const char *parse_string(const char *p, char *out, size_t out_sz) {
    p = skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\' && *p) {
            c = *p++;
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == 'r') c = '\r';
            /* else keep escaped char as-is */
        }
        if (i + 1 < out_sz) out[i++] = c;
    }
    if (*p != '"') return NULL;
    out[i] = '\0';
    return p + 1;
}

static int json_get_string_field(const char *json, const char *key,
                                 char *out, size_t out_sz) {
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    p = skip_ws(p);
    if (*p != ':') return 0;
    p = skip_ws(p + 1);
    return parse_string(p, out, out_sz) != NULL;
}

static int json_get_number_field(const char *json, const char *key,
                                 double *out) {
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    p = skip_ws(p);
    if (*p != ':') return 0;
    p = skip_ws(p + 1);
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static int json_get_bool_field(const char *json, const char *key, int *out) {
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    p = skip_ws(p);
    if (*p != ':') return 0;
    p = skip_ws(p + 1);
    if (strncmp(p, "true", 4) == 0) { *out = 1; return 1; }
    if (strncmp(p, "false", 5) == 0) { *out = 0; return 1; }
    return 0;
}

/* Parse one values array into ExpBuf.vals. */
static const char *parse_values_array(const char *p, ExpBuf *b) {
    p = skip_ws(p);
    if (*p != '[') return NULL;
    p++;
    b->nvals = 0;
    for (;;) {
        p = skip_ws(p);
        if (*p == ']') return p + 1;
        if (b->nvals >= MAX_VALUES) return NULL;
        if (*p == '"') {
            const char *n = parse_string(p, b->vals[b->nvals],
                                         sizeof b->vals[0]);
            if (!n) return NULL;
            p = n;
            b->nvals++;
        } else {
            /* bare number → stringify */
            char *end = NULL;
            (void)strtod(p, &end);
            if (end == p) return NULL;
            size_t n = (size_t)(end - p);
            if (n >= sizeof b->vals[0]) n = sizeof b->vals[0] - 1;
            memcpy(b->vals[b->nvals], p, n);
            b->vals[b->nvals][n] = '\0';
            b->nvals++;
            p = end;
        }
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        if (*p == ']') return p + 1;
        return NULL;
    }
}

static int parse_expect(const char *json, Expect *e) {
    memset(e, 0, sizeof *e);
    strcpy(e->entry, "main");
    e->len = 1;
    e->expect_ok = 1;
    e->abs_tol = 0.0;
    e->known_numeric = 0;

    char s[64];
    if (json_get_string_field(json, "entry", s, sizeof s))
        snprintf(e->entry, sizeof e->entry, "%s", s);
    double d;
    if (json_get_number_field(json, "len", &d) && d >= 0)
        e->len = (unsigned)d;
    int b;
    if (json_get_bool_field(json, "ok", &b))
        e->expect_ok = b ? 1 : 0;
    if (json_get_bool_field(json, "known_numeric_difference", &b) && b)
        e->known_numeric = 1;
    if (json_get_number_field(json, "abs_tol", &d) && d > 0.0) {
        e->abs_tol = d;
        e->known_numeric = 1;
    }
    if (json_get_number_field(json, "tolerance", &d) && d > 0.0) {
        e->abs_tol = d;
        e->known_numeric = 1;
    }
    /* Default tolerance when flagged without explicit abs_tol. */
    if (e->known_numeric && e->abs_tol <= 0.0)
        e->abs_tol = 1e-3;

    /* Walk every "name" : "..." near a "values" array inside buffers. */
    const char *p = strstr(json, "\"buffers\"");
    if (!p) return 1; /* empty buffers ok for some failure cases */
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    while (e->nbuf < MAX_BUF) {
        const char *name_key = strstr(p, "\"name\"");
        const char *vals_key = strstr(p, "\"values\"");
        if (!name_key || !vals_key) break;
        /* Stay inside this buffer object: name before values. */
        if (name_key > vals_key) {
            p = vals_key + 8;
            continue;
        }
        const char *q = name_key + 6;
        q = skip_ws(q);
        if (*q != ':') break;
        q = skip_ws(q + 1);
        ExpBuf *eb = &e->bufs[e->nbuf];
        if (!parse_string(q, eb->name, sizeof eb->name)) break;
        q = vals_key + 8;
        q = skip_ws(q);
        if (*q != ':') break;
        q = skip_ws(q + 1);
        q = parse_values_array(q, eb);
        if (!q) break;
        e->nbuf++;
        p = q;
        /* Next object or end of buffers array. */
        const char *next_obj = strstr(p, "{");
        const char *arr_end = strchr(p, ']');
        if (!next_obj || (arr_end && arr_end < next_obj)) break;
        p = next_obj;
    }
    return 1;
}

/* Build exact needle: "values":["a","b",...] */
static void build_values_needle(const ExpBuf *b, char *out, size_t out_sz) {
    size_t n = 0;
    n += (size_t)snprintf(out + n, out_sz - n, "\"values\":[");
    for (int i = 0; i < b->nvals && n + 4 < out_sz; i++) {
        if (i) out[n++] = ',';
        n += (size_t)snprintf(out + n, out_sz - n, "\"%s\"", b->vals[i]);
    }
    if (n + 2 < out_sz) {
        out[n++] = ']';
        out[n] = '\0';
    }
}

/* Extract "values":[...] after "name":"<bufname>" in interp JSON. */
static int extract_buffer_values(const char *json, const char *bufname,
                                 ExpBuf *out) {
    char name_pat[96];
    snprintf(name_pat, sizeof name_pat, "\"name\":\"%s\"", bufname);
    const char *p = strstr(json, name_pat);
    if (!p) return 0;
    const char *v = strstr(p, "\"values\"");
    if (!v) return 0;
    v = strchr(v, '[');
    if (!v) return 0;
    return parse_values_array(v, out) != NULL;
}

static int values_match_tol(const ExpBuf *want, const ExpBuf *got,
                            double abs_tol) {
    if (want->nvals != got->nvals) return 0;
    for (int i = 0; i < want->nvals; i++) {
        char *ew = NULL, *eg = NULL;
        double dw = strtod(want->vals[i], &ew);
        double dg = strtod(got->vals[i], &eg);
        /* If both parse as numbers, compare with tolerance. */
        if (ew != want->vals[i] && eg != got->vals[i] &&
            *ew == '\0' && *eg == '\0') {
            double ad = dw - dg;
            if (ad < 0.0) ad = -ad;
            if (ad > abs_tol) return 0;
        } else if (strcmp(want->vals[i], got->vals[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

static int run_case(const char *label, const char *src, const Expect *exp) {
    /* Numeric / whitelist path: full buffer compare with abs_tol. */
    if (exp->known_numeric) {
        char *j1 = wgsl_interp(src, exp->entry, 0, 0, 0, exp->len, NULL);
        if (!j1) {
            if (label) fprintf(stderr, "FAIL  %s: interp NULL\n", label);
            return 0;
        }
        int ok = 1;
        if (exp->expect_ok == 1 && !strstr(j1, "\"ok\":true")) {
            if (label) fprintf(stderr, "FAIL  %s: expected ok:true\n", label);
            ok = 0;
        }
        if (exp->expect_ok == 0 && strstr(j1, "\"ok\":true")) {
            if (label) fprintf(stderr, "FAIL  %s: expected ok:false\n", label);
            ok = 0;
        }
        for (int i = 0; i < exp->nbuf; i++) {
            ExpBuf got;
            memset(&got, 0, sizeof got);
            if (!extract_buffer_values(j1, exp->bufs[i].name, &got)) {
                if (label)
                    fprintf(stderr, "FAIL  %s: buffer '%s' missing\n",
                            label, exp->bufs[i].name);
                ok = 0;
                continue;
            }
            if (!values_match_tol(&exp->bufs[i], &got, exp->abs_tol)) {
                if (label) {
                    fprintf(stderr,
                            "FAIL  %s: buffer '%s' numeric mismatch "
                            "(tol=%g)\n",
                            label, exp->bufs[i].name, exp->abs_tol);
                    fprintf(stderr, "  want:");
                    for (int k = 0; k < exp->bufs[i].nvals; k++)
                        fprintf(stderr, " %s", exp->bufs[i].vals[k]);
                    fprintf(stderr, "\n  got: ");
                    for (int k = 0; k < got.nvals; k++)
                        fprintf(stderr, " %s", got.vals[k]);
                    fprintf(stderr, "\n");
                }
                ok = 0;
            }
        }
        /* Determinism: second run buffers slice. */
        char *j2 = wgsl_interp(src, exp->entry, 0, 0, 0, exp->len, NULL);
        if (!j2) {
            if (label) fprintf(stderr, "FAIL  %s: second interp NULL\n", label);
            ok = 0;
        } else {
            const char *b1 = strstr(j1, "\"buffers\"");
            const char *b2 = strstr(j2, "\"buffers\"");
            if (!b1 || !b2) {
                if (label) fprintf(stderr, "FAIL  %s: missing buffers\n", label);
                ok = 0;
            } else {
                const char *e1 = strstr(b1, "\"anomalies\"");
                const char *e2 = strstr(b2, "\"anomalies\"");
                size_t l1 = e1 ? (size_t)(e1 - b1) : strlen(b1);
                size_t l2 = e2 ? (size_t)(e2 - b2) : strlen(b2);
                if (l1 != l2 || memcmp(b1, b2, l1) != 0) {
                    if (label)
                        fprintf(stderr,
                                "FAIL  %s: non-deterministic buffers\n",
                                label);
                    ok = 0;
                }
            }
            wgsl_free_string(j2);
        }
        wgsl_free_string(j1);
        return ok;
    }

    /* Exact path: needles + wgsl_golden_run. */
    char needle_store[MAX_BUF][512];
    const char *needles[MAX_NEEDLES];
    int ni = 0;
    for (int i = 0; i < exp->nbuf && ni + 1 < MAX_NEEDLES; i++) {
        build_values_needle(&exp->bufs[i], needle_store[ni],
                            sizeof needle_store[0]);
        needles[ni] = needle_store[ni];
        ni++;
    }
    needles[ni] = NULL;
    return wgsl_golden_run(label, src, exp->entry, exp->len,
                           exp->expect_ok, ni ? needles : NULL);
}

/* qsort comparator for case basenames. */
static int cmp_strptr(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/**
 * Negative-test invariant: a deliberately wrong expect MUST fail.
 * Keeps the harness from becoming a vacuous always-green oracle.
 * Returns number of invariant failures (0 = OK).
 */
static int run_negative_invariants(void) {
    static const char *src =
        "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() { o[0] = 42; }\n";
    int inv_fail = 0;

    /* Exact path: wrong value must not pass. */
    {
        Expect bad;
        memset(&bad, 0, sizeof bad);
        strcpy(bad.entry, "main");
        bad.len = 1;
        bad.expect_ok = 1;
        bad.nbuf = 1;
        strcpy(bad.bufs[0].name, "o");
        bad.bufs[0].nvals = 1;
        strcpy(bad.bufs[0].vals[0], "999"); /* actual is 42 */
        /* label=NULL: suppress run_case diagnostics for expected rejection */
        if (run_case(NULL, src, &bad)) {
            fprintf(stderr,
                    "FAIL  invariant: wrong exact expect should FAIL\n");
            inv_fail++;
        } else {
            fprintf(stderr, "PASS  neg:exact_wrong_value (correctly rejected)\n");
        }
    }

    /* Numeric path: outside abs_tol must not pass. */
    {
        static const char *fsrc =
            "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
            "@compute @workgroup_size(1)\n"
            "fn main() { o[0] = 1.0; }\n";
        Expect bad;
        memset(&bad, 0, sizeof bad);
        strcpy(bad.entry, "main");
        bad.len = 1;
        bad.expect_ok = 1;
        bad.known_numeric = 1;
        bad.abs_tol = 0.01;
        bad.nbuf = 1;
        strcpy(bad.bufs[0].name, "o");
        bad.bufs[0].nvals = 1;
        strcpy(bad.bufs[0].vals[0], "2.0"); /* |2-1| >> 0.01 */
        if (run_case(NULL, fsrc, &bad)) {
            fprintf(stderr,
                    "FAIL  invariant: outside-tol numeric expect should FAIL\n");
            inv_fail++;
        } else {
            fprintf(stderr,
                    "PASS  neg:numeric_outside_tol (correctly rejected)\n");
        }
    }

    /* Numeric path: inside abs_tol must pass (sanity of whitelist). */
    {
        static const char *fsrc =
            "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
            "@compute @workgroup_size(1)\n"
            "fn main() { o[0] = 1.0; }\n";
        Expect good;
        memset(&good, 0, sizeof good);
        strcpy(good.entry, "main");
        good.len = 1;
        good.expect_ok = 1;
        good.known_numeric = 1;
        good.abs_tol = 0.01;
        good.nbuf = 1;
        strcpy(good.bufs[0].name, "o");
        good.bufs[0].nvals = 1;
        strcpy(good.bufs[0].vals[0], "1.005"); /* within 0.01 of 1.0 */
        if (!run_case("neg:numeric_inside_tol", fsrc, &good)) {
            fprintf(stderr,
                    "FAIL  invariant: inside-tol numeric expect should PASS\n");
            inv_fail++;
        } else {
            fprintf(stderr, "PASS  neg:numeric_inside_tol\n");
        }
    }

    return inv_fail;
}

int main(void) {
    /* Harness self-check before walking disk cases. */
    {
        int inv = run_negative_invariants();
        if (inv) {
            fprintf(stderr, "FAIL  test_golden_walk  (%d negative invariant%s)\n",
                    inv, inv == 1 ? "" : "s");
            return 1;
        }
    }

    DIR *d = opendir(CASES_DIR);
    if (!d) {
        fprintf(stderr,
                "FAIL  test_golden_walk: cannot opendir %s "
                "(run from repo root?)\n",
                CASES_DIR);
        return 1;
    }

    /* Collect *.wgsl basenames (no path). */
    char *names[512];
    int nnames = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!has_suffix(ent->d_name, ".wgsl")) continue;
        if (nnames >= 512) {
            fprintf(stderr, "FAIL  too many cases\n");
            closedir(d);
            return 1;
        }
        names[nnames] = strdup(ent->d_name);
        if (!names[nnames]) {
            closedir(d);
            return 1;
        }
        nnames++;
    }
    closedir(d);
    qsort(names, (size_t)nnames, sizeof names[0], cmp_strptr);

    for (int i = 0; i < nnames; i++) {
        const char *wgsl_name = names[i];
        size_t base_len = strlen(wgsl_name) - 5; /* strip .wgsl */
        char base[256];
        if (base_len >= sizeof base) base_len = sizeof base - 1;
        memcpy(base, wgsl_name, base_len);
        base[base_len] = '\0';

        char wgsl_path[512], exp_path[512];
        snprintf(wgsl_path, sizeof wgsl_path, "%s/%s", CASES_DIR, wgsl_name);
        snprintf(exp_path, sizeof exp_path, "%s/%s.expect.json",
                 CASES_DIR, base);

        size_t src_len = 0;
        char *src = slurp(wgsl_path, &src_len);
        if (!src) {
            fprintf(stderr, "FAIL  %s: cannot read shader\n", base);
            fail_count++;
            continue;
        }
        char *exp_json = slurp(exp_path, NULL);
        if (!exp_json) {
            fprintf(stderr, "SKIP  %s (no .expect.json)\n", base);
            skip_count++;
            free(src);
            continue;
        }

        Expect exp;
        if (!parse_expect(exp_json, &exp)) {
            fprintf(stderr, "FAIL  %s: bad expect.json\n", base);
            fail_count++;
            free(src);
            free(exp_json);
            continue;
        }
        free(exp_json);

        char label[288];
        snprintf(label, sizeof label, "golden:%s", base);
        if (run_case(label, src, &exp)) {
            fprintf(stderr, "PASS  %s%s\n", base,
                    exp.known_numeric ? " (numeric tol)" : "");
            pass_count++;
        } else {
            fail_count++;
        }
        free(src);
    }

    for (int i = 0; i < nnames; i++) free(names[i]);

    fprintf(stderr,
            "\n%s  test_golden_walk  (%d pass, %d fail, %d skip; "
            "%d file%s)\n",
            fail_count ? "FAIL" : "PASS", pass_count, fail_count,
            skip_count, nnames, nnames == 1 ? "" : "s");

    if (pass_count < 50 && fail_count == 0) {
        fprintf(stderr,
                "FAIL  expected ≥50 golden cases, got %d pass "
                "(%d skip)\n",
                pass_count, skip_count);
        return 1;
    }
    return fail_count ? 1 : 0;
}
