/**
 * @file session.c — public API module (was wgsl .inc fragment) (session.inc).
 * Public API implementation; see include/wgsl.h.
 */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "wgsl.h"
#include "internal/result.h"
#include "internal/check.h"
#include "internal/source.h"
#include "internal/diag.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

/* Session cache and partial typecheck.
 *
 * Byte-identical source → return cached WGSLResult (no pipeline).
 * Between checks: per-decl content hashes + module interface hash.
 * When the interface is stable, function bodies whose content hash is
 * unchanged and which carried no prior diagnostics skip the body
 * typecheck walk (signature typing still runs).
 */

static uint64_t sess_fnv_init(void) { return 14695981039346656037ull; }

static uint64_t sess_fnv_update(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t sess_fnv_bytes(const void *p, size_t n) {
    return sess_fnv_update(sess_fnv_init(), p, n);
}

static uint64_t sess_now_us(void) {
#if defined(__EMSCRIPTEN__)
    double ms = emscripten_get_now();
    return ms > 0.0 ? (uint64_t)(ms * 1000.0) : 0;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
#endif
}

static uint32_t sess_elapsed_ms(uint64_t start_us, uint64_t end_us) {
    if (!start_us || end_us < start_us) return 0;
    uint64_t us = end_us - start_us;
    uint64_t ms = (us + 999ull) / 1000ull; /* visible integer telemetry */
    return ms > UINT32_MAX ? UINT32_MAX : (uint32_t)ms;
}

typedef struct {
    char               *name;
    uint32_t            name_len;
    WGSLSessionDeclKind kind;
    uint64_t            content_hash;
    uint64_t            sig_hash;     /* interface contribution */
    uint32_t            span_off;     /* full decl source span   */
    uint32_t            span_len;
    int                 changed;
} SessDecl;

struct WGSLSession {
    char           *src;
    size_t          src_len;
    uint64_t        src_hash;
    uint64_t        iface_hash;
    WGSLResult     *result;
    SessDecl       *decls;
    int             ndecls;
    SessDecl       *prev;
    int             nprev;
    int            *changed_idx;
    int             nchanged;
    WGSLSessionInfo info;
};

static WGSLSessionDeclKind sess_kind(const WGSLNode *n) {
    if (!n) return WGSL_SESSION_DECL_OTHER;
    switch (n->kind) {
    case WGSL_NODE_DECL_FUNCTION: return WGSL_SESSION_DECL_FUNCTION;
    case WGSL_NODE_DECL_STRUCT:   return WGSL_SESSION_DECL_STRUCT;
    case WGSL_NODE_DECL_VAR:      return WGSL_SESSION_DECL_VAR;
    case WGSL_NODE_DECL_CONST:    return WGSL_SESSION_DECL_CONST;
    case WGSL_NODE_DECL_OVERRIDE: return WGSL_SESSION_DECL_OVERRIDE;
    case WGSL_NODE_DECL_ALIAS:    return WGSL_SESSION_DECL_ALIAS;
    default:                      return WGSL_SESSION_DECL_OTHER;
    }
}

static void sess_free_decls(SessDecl *d, int n) {
    if (!d) return;
    for (int i = 0; i < n; i++) free(d[i].name);
    free(d);
}

static const WGSLNode *sess_fn_body(const WGSLNode *fn) {
    if (!fn) return NULL;
    for (uint32_t i = 0; i < fn->child_count; i++) {
        const WGSLNode *c = fn->children[i];
        if (c && c->kind == WGSL_NODE_STMT_COMPOUND) return c;
    }
    return NULL;
}

static char *sess_strndup(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    if (n) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* Collect top-level decl fingerprints + module interface hash. */
static int sess_fingerprint(WGSLResult *r, SessDecl **out, int *out_n,
                            uint64_t *iface_hash)
{
    *out = NULL;
    *out_n = 0;
    uint64_t ih = sess_fnv_init();
    if (!r || !r->ast.root || !r->src_copy) {
        if (iface_hash) *iface_hash = ih;
        return 1;
    }
    const char *src = r->src_copy;
    size_t src_len = r->src_len;
    const WGSLNode *tu = r->ast.root;

    SessDecl *decls = NULL;
    int n = 0, cap = 0;

    for (uint32_t i = 0; i < tu->child_count; i++) {
        const WGSLNode *node = tu->children[i];
        if (!node) continue;

        /* Directives contribute to the interface only. */
        if (node->kind == WGSL_NODE_DIR_ENABLE ||
            node->kind == WGSL_NODE_DIR_REQUIRES ||
            node->kind == WGSL_NODE_DIR_DIAGNOSTIC)
        {
            uint32_t o = node->span_offset, l = node->span_length;
            if ((size_t)o + (size_t)l <= src_len)
                ih = sess_fnv_update(ih, src + o, l);
            continue;
        }

        WGSLSessionDeclKind k = sess_kind(node);
        if (k == WGSL_SESSION_DECL_OTHER) continue;

        uint32_t no = wgsl_node_name_span(node).offset;
        uint32_t nl = wgsl_node_name_span(node).length;
        uint32_t so = node->span_offset, sl = node->span_length;
        if ((size_t)so + (size_t)sl > src_len) continue;

        uint64_t ch = sess_fnv_bytes(src + so, sl);
        uint64_t sh = ch;
        if (k == WGSL_SESSION_DECL_FUNCTION) {
            const WGSLNode *body = sess_fn_body(node);
            if (body && body->span_offset >= so &&
                body->span_offset <= so + sl)
            {
                /* Signature = text from decl start up to body start. */
                uint32_t sig_len = body->span_offset - so;
                sh = sess_fnv_bytes(src + so, sig_len);
            }
        }
        /* Interface material: signature strip (or full for non-fn). */
        ih = sess_fnv_update(ih, &sh, sizeof sh);
        if (nl && (size_t)no + (size_t)nl <= src_len)
            ih = sess_fnv_update(ih, src + no, nl);

        if (n + 1 > cap) {
            int nc = cap ? cap * 2 : 8;
            SessDecl *g = (SessDecl *)realloc(decls, (size_t)nc * sizeof *g);
            if (!g) { sess_free_decls(decls, n); return 0; }
            decls = g; cap = nc;
        }
        SessDecl *d = &decls[n];
        memset(d, 0, sizeof *d);
        d->kind = k;
        d->content_hash = ch;
        d->sig_hash = sh;
        d->span_off = so;
        d->span_len = sl;
        d->name_len = nl;
        if (nl && (size_t)no + (size_t)nl <= src_len)
            d->name = sess_strndup(src + no, nl);
        else
            d->name = sess_strndup("", 0);
        if (!d->name) { sess_free_decls(decls, n); return 0; }
        n++;
    }

    *out = decls;
    *out_n = n;
    if (iface_hash) *iface_hash = ih;
    return 1;
}

static const SessDecl *sess_find_prev(const SessDecl *prev, int nprev,
                                      const SessDecl *d)
{
    for (int i = 0; i < nprev; i++) {
        if (prev[i].kind != d->kind) continue;
        if (prev[i].name_len != d->name_len) continue;
        if (d->name_len == 0 ||
            (prev[i].name && d->name &&
             memcmp(prev[i].name, d->name, d->name_len) == 0))
            return &prev[i];
    }
    return NULL;
}

static void sess_diff(WGSLSession *s) {
    s->nchanged = 0;
    free(s->changed_idx);
    s->changed_idx = NULL;
    s->info.decls_total = s->ndecls;
    s->info.decls_changed = 0;
    s->info.decls_removed = 0;

    if (s->ndecls > 0) {
        s->changed_idx = (int *)malloc((size_t)s->ndecls * sizeof(int));
        /* OOM → just skip enumeration */
    }

    for (int i = 0; i < s->ndecls; i++) {
        const SessDecl *p = sess_find_prev(s->prev, s->nprev, &s->decls[i]);
        int ch = 0;
        if (!p) ch = 1;
        else if (p->content_hash != s->decls[i].content_hash) ch = 1;
        s->decls[i].changed = ch;
        if (ch) {
            s->info.decls_changed++;
            if (s->changed_idx)
                s->changed_idx[s->nchanged++] = i;
        }
    }
    /* removed = prev names not in current */
    for (int i = 0; i < s->nprev; i++) {
        int found = 0;
        for (int j = 0; j < s->ndecls; j++) {
            if (s->decls[j].kind == s->prev[i].kind &&
                s->decls[j].name_len == s->prev[i].name_len &&
                (s->prev[i].name_len == 0 ||
                 (s->decls[j].name && s->prev[i].name &&
                  memcmp(s->decls[j].name, s->prev[i].name,
                         s->prev[i].name_len) == 0)))
            {
                found = 1; break;
            }
        }
        if (!found) s->info.decls_removed++;
    }
}

WGSLSession *wgsl_session_new(void) {
    WGSLSession *s = (WGSLSession *)calloc(1, sizeof *s);
    return s;
}

void wgsl_session_destroy(WGSLSession *s) {
    if (!s) return;
    free(s->src);
    if (s->result) wgsl_free(s->result);
    sess_free_decls(s->decls, s->ndecls);
    sess_free_decls(s->prev, s->nprev);
    free(s->changed_idx);
    free(s);
}

const WGSLSessionInfo *wgsl_session_info(const WGSLSession *s) {
    static const WGSLSessionInfo empty;
    return s ? &s->info : &empty;
}

int wgsl_session_changed_decl(const WGSLSession *s, int index,
                              const char **name, uint32_t *name_len,
                              WGSLSessionDeclKind *kind)
{
    if (!s || index < 0 || index >= s->nchanged || !s->changed_idx) return 0;
    int di = s->changed_idx[index];
    if (di < 0 || di >= s->ndecls) return 0;
    const SessDecl *d = &s->decls[di];
    if (name) *name = d->name ? d->name : "";
    if (name_len) *name_len = d->name_len;
    if (kind) *kind = d->kind;
    return 1;
}

const WGSLResult *wgsl_session_check(WGSLSession *s, const char *src) {
    return wgsl_session_check_n(s, src, src ? strlen(src) : 0);
}

/* True if any diagnostic in `prev` falls on a line covered by the
 * previous decl span (conservative: may over-count boundary lines). */
static int sess_prev_diag_in_span(const WGSLResult *prev,
                                  uint32_t span_off, uint32_t span_len)
{
    if (!prev || !prev->diag.count) return 0;
    uint32_t start_l = 1, start_c = 1, end_l = 1, end_c = 1;
    wgsl_source_offset_to_line_col_utf16(
        &prev->source, span_off, &start_l, &start_c);
    uint32_t end_off = span_off;
    if (span_len > 0) {
        if (span_len > UINT32_MAX - span_off) end_off = UINT32_MAX;
        else end_off = span_off + span_len;
    }
    if ((size_t)end_off > prev->src_len) end_off = (uint32_t)prev->src_len;
    wgsl_source_offset_to_line_col_utf16(
        &prev->source, end_off, &end_l, &end_c);
    for (size_t i = 0; i < prev->diag.count; i++) {
        const WGSLDiagnostic *d = &prev->diag.items[i];
        if (!d) continue;
        if (d->line >= start_l && d->line <= end_l) return 1;
    }
    return 0;
}

/* Build skip-body name list for functions that are content-stable under
 * a stable interface and carried no diagnostics last check. */
static int sess_build_skip_bodies(
    WGSLSession *s, const SessDecl *new_decls, int nnew,
    const char ***out_names, uint32_t **out_lens, int *out_n)
{
    *out_names = NULL;
    *out_lens = NULL;
    *out_n = 0;
    if (!s || !s->result || nnew <= 0) return 1;

    const char **names = NULL;
    uint32_t *lens = NULL;
    int n = 0, cap = 0;

    for (int i = 0; i < nnew; i++) {
        const SessDecl *d = &new_decls[i];
        if (d->kind != WGSL_SESSION_DECL_FUNCTION) continue;
        if (!d->name || d->name_len == 0) continue;
        const SessDecl *p = sess_find_prev(s->prev, s->nprev, d);
        if (!p) continue;
        if (p->content_hash != d->content_hash) continue;
        /* Prior diags in this function must be re-emitted → re-type. */
        if (sess_prev_diag_in_span(s->result, p->span_off, p->span_len))
            continue;

        if (n + 1 > cap) {
            int nc = cap ? cap * 2 : 4;
            const char **gn = (const char **)realloc(
                names, (size_t)nc * sizeof *gn);
            uint32_t *gl = (uint32_t *)realloc(lens, (size_t)nc * sizeof *gl);
            if (!gn || !gl) {
                free(gn ? (void *)gn : (void *)names);
                free(gl ? (void *)gl : (void *)lens);
                return 0;
            }
            names = gn;
            lens = gl;
            cap = nc;
        }
        names[n] = d->name;
        lens[n] = d->name_len;
        n++;
    }
    *out_names = names;
    *out_lens = lens;
    *out_n = n;
    return 1;
}

const WGSLResult *wgsl_session_check_n(WGSLSession *s, const char *src,
                                       size_t src_len)
{
    if (!s) return NULL;
    if (!src) src = "";

    uint64_t h = sess_fnv_bytes(src, src_len);

    /* Full cache hit: byte-identical source. */
    if (s->result && s->src && s->src_len == src_len && s->src_hash == h &&
        (src_len == 0 || memcmp(s->src, src, src_len) == 0))
    {
        s->info.cache_hit = 1;
        s->info.iface_changed = 0;
        s->info.decls_changed = 0;
        s->info.decls_removed = 0;
        s->info.decls_total = s->ndecls;
        s->info.fn_bodies_typed = 0;
        s->info.fn_bodies_skipped = 0;
        s->info.skipped_nodes = 0;
        s->info.typecheck_ms = 0;
        s->info.src_hash = h;
        s->info.iface_hash = s->iface_hash;
        s->nchanged = 0;
        return s->result;
    }

    /* Staged pipeline: parse, fingerprint, partial typecheck, validate. */
    WGSLResult *r = wgsl_pipeline_begin(src, src_len);
    if (!r) return NULL;

    /* Rotate fingerprints: current → prev (prev still names the last
     * completed check; s->result is still the previous pipeline result
     * until we swap at the end). */
    sess_free_decls(s->prev, s->nprev);
    s->prev = s->decls;
    s->nprev = s->ndecls;
    s->decls = NULL;
    s->ndecls = 0;

    uint64_t iface = 0;
    if (!sess_fingerprint(r, &s->decls, &s->ndecls, &iface)) {
        s->decls = NULL;
        s->ndecls = 0;
        iface = 0;
    }

    int iface_changed = 1;
    if (s->result && s->iface_hash == iface) iface_changed = 0;

    /* Body-skip plan: only when the module interface is stable so the
     * typing environment for unchanged function bodies is identical. */
    const char **skip_names = NULL;
    uint32_t *skip_lens = NULL;
    int skip_n = 0;
    WGSLTypecheckOpts tc_opts;
    memset(&tc_opts, 0, sizeof tc_opts);
    if (!iface_changed && r->ast.root) {
        if (!sess_build_skip_bodies(s, s->decls, s->ndecls,
                                    &skip_names, &skip_lens, &skip_n))
        {
            skip_names = NULL;
            skip_lens = NULL;
            skip_n = 0;
        }
        if (skip_n > 0) {
            tc_opts.skip_body_names = skip_names;
            tc_opts.skip_body_name_lens = skip_lens;
            tc_opts.skip_body_count = skip_n;
        }
    }

    int early_ok = 1;
    uint64_t tc_start = sess_now_us();
    int tc_ok = 0;
    if (r->ast.root) {
        tc_ok = wgsl_pipeline_typecheck(
            r, skip_n > 0 ? &tc_opts : NULL);
    }
    uint64_t tc_end = sess_now_us();
    if (r->ast.root) {
        int val_ok = wgsl_pipeline_validate(r);
        early_ok = tc_ok && val_ok;
    } else {
        early_ok = 0;
    }
    wgsl_pipeline_finalize(r, early_ok);

    int bodies_typed = r->tc.bodies_typed;
    int bodies_skipped = r->tc.bodies_skipped;
    int skipped_nodes = r->tc.skipped_nodes;
    uint32_t typecheck_ms = sess_elapsed_ms(tc_start, tc_end);
    free(skip_names);
    free(skip_lens);

    /* Replace owned source + result */
    free(s->src);
    s->src = (char *)malloc(src_len + 1);
    if (s->src) {
        if (src_len) memcpy(s->src, src, src_len);
        s->src[src_len] = '\0';
        s->src_len = src_len;
    } else {
        s->src_len = 0;
    }
    s->src_hash = h;
    s->iface_hash = iface;

    if (s->result) wgsl_free(s->result);
    s->result = r;

    memset(&s->info, 0, sizeof s->info);
    s->info.cache_hit = 0;
    s->info.iface_changed = iface_changed;
    s->info.fn_bodies_typed = bodies_typed;
    s->info.fn_bodies_skipped = bodies_skipped;
    s->info.skipped_nodes = skipped_nodes;
    s->info.typecheck_ms = typecheck_ms;
    s->info.src_hash = h;
    s->info.iface_hash = iface;
    sess_diff(s);

    return s->result;
}
