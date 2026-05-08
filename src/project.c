/**
 * @file project.c — `wgslconfig.toml` loader implementation.
 *
 * Snapshot-only.  Load once, query many times via
 * `wgsl_project_match`; the project owns the parsed strings and
 * frees them in `wgsl_project_close`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal/project.h"
#include "internal/toml.h"
#include "internal/glob.h"

struct WGSLProject {
    char **preamble_files;
    int    preamble_count;
    char **inject_globs;
    int    glob_count;
};

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *o = (char *)malloc(n + 1);
    if (!o) return NULL;
    memcpy(o, s, n + 1);
    return o;
}

static int xpush(char ***arr, int *n, const char *s) {
    char *dup = xstrdup(s);
    if (!dup) return 0;
    char **grown = (char **)realloc(*arr, (size_t)(*n + 1) * sizeof(char *));
    if (!grown) { free(dup); return 0; }
    grown[*n] = dup;
    *arr = grown;
    *n  += 1;
    return 1;
}

WGSLProject *wgsl_project_open_from_string(
    const char *toml_text, size_t toml_len,
    char *err, size_t err_len)
{
    char toml_err[256] = {0};
    WGSLToml *t = wgsl_toml_parse(toml_text, toml_len, toml_err, sizeof toml_err);
    if (!t) {
        if (err && err_len > 0) snprintf(err, err_len, "TOML: %s", toml_err);
        return NULL;
    }

    WGSLProject *p = (WGSLProject *)calloc(1, sizeof *p);
    if (!p) {
        if (err && err_len > 0) snprintf(err, err_len, "out of memory");
        wgsl_toml_free(t);
        return NULL;
    }

    /* `[preamble] file = "…"` — single (kept for the original note's shape). */
    const char *single = wgsl_toml_get_string(t, "preamble", "file");
    if (single && !xpush(&p->preamble_files, &p->preamble_count, single)) goto oom;

    /* `[preamble] files = ["a.wgsl", "b.wgsl"]` — list. */
    int files_n = 0;
    const char *const *files = wgsl_toml_get_string_array(t, "preamble", "files", &files_n);
    for (int i = 0; i < files_n; i++) {
        if (!xpush(&p->preamble_files, &p->preamble_count, files[i])) goto oom;
    }

    /* `[preamble] auto_inject_into = ["doublestar-slash glob (any depth, any *.wgsl)"]` — path globs. */
    int inj_n = 0;
    const char *const *inj = wgsl_toml_get_string_array(t, "preamble", "auto_inject_into", &inj_n);
    for (int i = 0; i < inj_n; i++) {
        if (!xpush(&p->inject_globs, &p->glob_count, inj[i])) goto oom;
    }

    wgsl_toml_free(t);
    return p;

oom:
    if (err && err_len > 0) snprintf(err, err_len, "out of memory");
    wgsl_project_close(p);
    wgsl_toml_free(t);
    return NULL;
}

void wgsl_project_close(WGSLProject *p) {
    if (!p) return;
    for (int i = 0; i < p->preamble_count; i++) free(p->preamble_files[i]);
    free(p->preamble_files);
    for (int i = 0; i < p->glob_count; i++) free(p->inject_globs[i]);
    free(p->inject_globs);
    free(p);
}

const char *const *wgsl_project_match(
    const WGSLProject *p, const char *file_path, int *out_count)
{
    if (out_count) *out_count = 0;
    if (!p || !file_path) return NULL;
    if (p->preamble_count == 0) return NULL;
    /* No globs declared → never inject (treat as opt-in). */
    if (p->glob_count == 0) return NULL;

    /* Self-preamble exclusion: a preamble file matching its own
     * inject globs would otherwise compile under itself, which never
     * makes sense.  Skip when `file_path` is one of the preamble
     * files. */
    for (int i = 0; i < p->preamble_count; i++) {
        if (strcmp(p->preamble_files[i], file_path) == 0) return NULL;
    }

    for (int i = 0; i < p->glob_count; i++) {
        if (wgsl_glob_match(p->inject_globs[i], file_path)) {
            if (out_count) *out_count = p->preamble_count;
            return (const char *const *)p->preamble_files;
        }
    }
    return NULL;
}

const char *const *wgsl_project_preamble_files(
    const WGSLProject *p, int *out_count)
{
    if (out_count) *out_count = 0;
    if (!p) return NULL;
    if (out_count) *out_count = p->preamble_count;
    return (const char *const *)p->preamble_files;
}
