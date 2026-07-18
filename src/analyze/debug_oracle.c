/**
 * @file debug_oracle.c — CI-friendly N-lane correctness oracle.
 *
 * Deterministic CPU SIMT execution as a fail/pass oracle plus a compact
 * explanation layer.  This is intentionally separate from step debugging.
 */
#include "wgsl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *buf;
    size_t len, cap;
    int    oom;
} DbgJson;

static void dj_grow(DbgJson *j, size_t n) {
    if (j->oom) return;
    if (j->len + n + 1 <= j->cap) return;
    size_t nc = j->cap ? j->cap : 512;
    while (nc < j->len + n + 1) {
        if (nc > SIZE_MAX / 2) { j->oom = 1; return; }
        nc *= 2;
    }
    char *g = (char *)realloc(j->buf, nc);
    if (!g) { j->oom = 1; return; }
    j->buf = g;
    j->cap = nc;
}

static void dj_putn(DbgJson *j, const char *s, size_t n) {
    if (!s || !n || j->oom) return;
    dj_grow(j, n);
    if (j->oom) return;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
    j->buf[j->len] = '\0';
}

static void dj_puts(DbgJson *j, const char *s) {
    if (s) dj_putn(j, s, strlen(s));
}

static void dj_u(DbgJson *j, unsigned long v) {
    char b[32];
    int n = snprintf(b, sizeof b, "%lu", v);
    if (n > 0) dj_putn(j, b, (size_t)n);
}

static void dj_json(DbgJson *j, const char *s) {
    dj_puts(j, "\"");
    if (s) {
        for (; *s; s++) {
            unsigned char c = (unsigned char)*s;
            if (c == '"' || c == '\\') {
                char b[2] = { '\\', (char)c };
                dj_putn(j, b, 2);
            } else if (c == '\n') {
                dj_puts(j, "\\n");
            } else if (c == '\r') {
                dj_puts(j, "\\r");
            } else if (c == '\t') {
                dj_puts(j, "\\t");
            } else if (c < 0x20) {
                char b[7];
                int n = snprintf(b, sizeof b, "\\u%04x", c);
                if (n > 0) dj_putn(j, b, (size_t)n);
            } else {
                dj_putn(j, (const char *)&c, 1);
            }
        }
    }
    dj_puts(j, "\"");
}

static unsigned json_u_after(const char *json, const char *needle) {
    const char *p = json ? strstr(json, needle) : NULL;
    if (!p) return 0;
    p += strlen(needle);
    return (unsigned)strtoul(p, NULL, 10);
}

static int json_has_true_after(const char *json, const char *object, const char *field) {
    const char *p = object ? strstr(json, object) : json;
    if (!p) return 0;
    const char *f = strstr(p, field);
    return f && strstr(f, "true") == f + strlen(field);
}

static void add_issue_prefix(DbgJson *j, int *first, const char *kind) {
    if (!*first) dj_puts(j, ",");
    *first = 0;
    dj_puts(j, "{\"kind\":");
    dj_json(j, kind);
    dj_puts(j, ",\"severity\":\"error\"");
}

static void add_issue_suffix(DbgJson *j, const char *summary, const char *why) {
    dj_puts(j, ",\"summary\":");
    dj_json(j, summary);
    dj_puts(j, ",\"why\":");
    dj_json(j, why);
    dj_puts(j, ",\"ci\":\"fail\"}");
}

static int append_race_issues(DbgJson *out, const char *json, int *first_issue) {
    const char *races = strstr(json, "\"races\"");
    const char *samples = races ? strstr(races, "\"samples\":[") : NULL;
    if (!samples) return 0;
    const char *arr_end = strchr(samples, ']');
    if (!arr_end) return 0;
    const char *p = samples;
    int nissues = 0;
    while (nissues < 8) {
        const char *linep = strstr(p, "{\"line\":");
        if (!linep || linep > arr_end) break;
        const char *end = strchr(linep, '}');
        if (!end || end > arr_end) break;
        unsigned line = (unsigned)strtoul(linep + 8, NULL, 10);
        const char *kindp = strstr(linep, "\"kind\":\"");
        char kind[40] = "data race";
        if (kindp && kindp < end) {
            kindp += 8;
            const char *ke = strchr(kindp, '"');
            size_t kl = ke && ke < end ? (size_t)(ke - kindp) : 0;
            if (kl >= sizeof kind) kl = sizeof kind - 1;
            if (kl) { memcpy(kind, kindp, kl); kind[kl] = '\0'; }
        }
        add_issue_prefix(out, first_issue, "data_race");
        dj_puts(out, ",\"line\":");
        dj_u(out, line);
        dj_puts(out, ",\"race_kind\":");
        dj_json(out, kind);
        add_issue_suffix(out,
            "shared/storage cell race detected",
            "Two or more lanes touched the same shared/storage cell in one barrier epoch, and at least one touch was a write. GPU scheduling may pick a different winning value, so CI should reject this shader until a barrier, atomic, or per-lane address removes the race.");
        nissues++;
        p = end + 1;
    }
    return nissues;
}

char *wgsl_debug_oracle(const char *src, const char *entry,
                        unsigned gx, unsigned gy, unsigned gz,
                        unsigned default_len, const char *seeds)
{
    const char *ep = entry ? entry : "main";
    char *trace = wgsl_interp(src, ep, gx, gy, gz, default_len, seeds);
    if (!trace) return NULL;

    unsigned races = json_u_after(trace, "\"races\":{\"count\":");
    unsigned nan = json_u_after(trace, "\"anomalies\":{\"nan\":");
    unsigned inf = json_u_after(trace, "\"inf\":");
    int interp_ok = strstr(trace, "\"ok\":true") != NULL;
    int truncated = strstr(trace, "\"truncated\":true") != NULL;
    int race_truncated = json_has_true_after(trace, "\"races\"", "\"truncated\":");
    int warnings = strstr(trace, "\"warnings\":[{") != NULL;

    DbgJson out = {0};
    int first_issue = 1;
    int issue_count = 0;
    int fail = (!interp_ok || races || nan || inf || truncated ||
                race_truncated || warnings);

    dj_puts(&out, "{\"ok\":");
    dj_puts(&out, fail ? "false" : "true");
    dj_puts(&out, ",\"verdict\":\"");
    dj_puts(&out, fail ? "fail" : "pass");
    dj_puts(&out, "\",\"wedge\":\"correctness-oracle+explainer\"");
    dj_puts(&out, ",\"entry\":");
    dj_json(&out, ep);
    dj_puts(&out, ",\"gid\":[");
    dj_u(&out, gx); dj_puts(&out, ","); dj_u(&out, gy); dj_puts(&out, ","); dj_u(&out, gz);
    dj_puts(&out, "],\"issues\":[");

    int added = races ? append_race_issues(&out, trace, &first_issue) : 0;
    issue_count += added;
    if (races && added == 0) {
        add_issue_prefix(&out, &first_issue, "data_race");
        add_issue_suffix(&out, "shared/storage cell race detected",
            "The N-lane interpreter found at least one same-epoch shared/storage race, but the sample list was empty or truncated.");
        issue_count++;
    }
    if (race_truncated) {
        add_issue_prefix(&out, &first_issue, "race_analysis_truncated");
        add_issue_suffix(&out, "race-cell analysis hit its cap",
            "The oracle could not enumerate every shared/storage cell, so CI should fail closed instead of accepting incomplete race evidence.");
        issue_count++;
    }
    if (nan || inf) {
        add_issue_prefix(&out, &first_issue, "numeric_anomaly");
        dj_puts(&out, ",\"nan\":"); dj_u(&out, nan);
        dj_puts(&out, ",\"inf\":"); dj_u(&out, inf);
        add_issue_suffix(&out, "runtime NaN/Inf produced",
            "The focus lane produced a non-finite value during deterministic execution; this is usually an unstable numerical path worth blocking or explicitly whitelisting.");
        issue_count++;
    }
    if (truncated) {
        add_issue_prefix(&out, &first_issue, "execution_truncated");
        add_issue_suffix(&out, "interpreter step cap reached",
            "The shader did not finish inside the deterministic execution budget, so the oracle cannot prove the run safe.");
        issue_count++;
    }
    if (warnings) {
        add_issue_prefix(&out, &first_issue, "unsupported_runtime_surface");
        add_issue_suffix(&out, "interpreter emitted warnings",
            "The execution used a construct the CPU model reports as approximate or unsupported. CI should fail closed for correctness-oracle mode.");
        issue_count++;
    }
    if (!interp_ok && issue_count == 0) {
        add_issue_prefix(&out, &first_issue, "frontend_or_runtime_error");
        add_issue_suffix(&out, "interpreter returned ok:false",
            "The frontend or runtime rejected the shader. Inspect wgsl_interp for the detailed error trace.");
        issue_count++;
    }

    dj_puts(&out, "],\"summary\":{\"issue_count\":");
    dj_u(&out, (unsigned long)issue_count);
    dj_puts(&out, ",\"race_count\":"); dj_u(&out, races);
    dj_puts(&out, ",\"nan\":"); dj_u(&out, nan);
    dj_puts(&out, ",\"inf\":"); dj_u(&out, inf);
    dj_puts(&out, ",\"truncated\":"); dj_puts(&out, truncated ? "true" : "false");
    dj_puts(&out, ",\"race_truncated\":"); dj_puts(&out, race_truncated ? "true" : "false");
    dj_puts(&out, "},\"why\":\"");
    dj_puts(&out, "Deterministic N-lane CPU execution complements GPU debuggers by making cross-lane races, non-finite numerics, and incomplete simulations visible as CI JSON.");
    dj_puts(&out, "\",\"source\":\"wgsl_interp\"}");

    wgsl_free_string(trace);
    if (out.oom) {
        free(out.buf);
        return NULL;
    }
    return out.buf;
}
