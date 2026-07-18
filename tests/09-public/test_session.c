/**
 * §3.2 — content-addressed session: cache hits + decl change tracking.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} SBuf;

static int sb_grow(SBuf *b, size_t need) {
    if (b->len + need + 1 <= b->cap) return 1;
    size_t nc = b->cap ? b->cap : 4096;
    while (nc < b->len + need + 1) {
        if (nc > ((size_t)-1) / 2) return 0;
        nc *= 2;
    }
    char *g = (char *)realloc(b->buf, nc);
    if (!g) return 0;
    b->buf = g;
    b->cap = nc;
    return 1;
}

static int sb_puts(SBuf *b, const char *s) {
    size_t n = strlen(s);
    if (!sb_grow(b, n)) return 0;
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = 0;
    return 1;
}

static int sb_printf(SBuf *b, const char *fmt, int v) {
    char tmp[96];
    int n = snprintf(tmp, sizeof tmp, fmt, v);
    if (n < 0 || (size_t)n >= sizeof tmp) return 0;
    return sb_puts(b, tmp);
}

static char *make_large_body_src(int changed_value, int body_lines) {
    SBuf b = {0};
    if (!sb_puts(&b, "fn stable() {\n  var x: i32 = 0;\n")) goto oom;
    for (int i = 0; i < body_lines; i++) {
        if (!sb_puts(&b, "  x = x + 1;\n")) goto oom;
    }
    if (!sb_puts(&b, "  _ = x;\n}\n")) goto oom;
    if (!sb_printf(&b, "fn changed() -> i32 { return %d; }\n",
                   changed_value)) goto oom;
    return b.buf;
oom:
    free(b.buf);
    return NULL;
}

int main(void) {
    /* Identity cache. */
    {
        WGSLSession *s = wgsl_session_new();
        CHECK(s != NULL, "session new");
        const char *src =
            "fn add(a: i32) -> i32 { return a; }\n"
            "fn main() { _ = add(1); }\n";
        const WGSLResult *r1 = wgsl_session_check(s, src);
        CHECK(r1 != NULL && wgsl_ok(r1), "first check ok");
        const WGSLSessionInfo *i1 = wgsl_session_info(s);
        CHECK(i1 && i1->cache_hit == 0, "first: not cache hit");
        CHECK(i1->decls_total >= 2, "first: ≥2 decls");

        const WGSLResult *r2 = wgsl_session_check(s, src);
        CHECK(r2 == r1, "second: same result pointer");
        const WGSLSessionInfo *i2 = wgsl_session_info(s);
        CHECK(i2 && i2->cache_hit == 1, "second: cache hit");
        CHECK(i2->decls_changed == 0, "second: no decls changed");

        wgsl_session_destroy(s);
    }

    /* Body-only edit: iface stable, one decl changed. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *a =
            "const C: i32 = 1;\n"
            "fn f() -> i32 { return C; }\n"
            "fn g() -> i32 { return 2; }\n";
        const char *b =
            "const C: i32 = 1;\n"
            "fn f() -> i32 { return C + 1; }\n"  /* body changed */
            "fn g() -> i32 { return 2; }\n";
        CHECK(wgsl_ok(wgsl_session_check(s, a)), "body: a ok");
        const WGSLSessionInfo *ia = wgsl_session_info(s);
        uint64_t iface_a = ia->iface_hash;
        /* First check types every function body (nothing to skip). */
        CHECK(ia->fn_bodies_typed >= 2, "body: a typed ≥2 fns");
        CHECK(ia->fn_bodies_skipped == 0, "body: a skipped 0");

        CHECK(wgsl_ok(wgsl_session_check(s, b)), "body: b ok");
        const WGSLSessionInfo *ib = wgsl_session_info(s);
        CHECK(ib->cache_hit == 0, "body: not cache hit");
        CHECK(ib->iface_changed == 0, "body: interface stable");
        CHECK(ib->iface_hash == iface_a, "body: iface hash equal");
        CHECK(ib->decls_changed == 1, "body: exactly 1 decl changed");
        /* Partial TC: re-type f, skip g (unchanged, diagnostic-free). */
        CHECK(ib->fn_bodies_typed >= 1, "body: b typed ≥1 (f)");
        CHECK(ib->fn_bodies_skipped >= 1, "body: b skipped ≥1 (g)");

        const char *nm = NULL;
        uint32_t nl = 0;
        WGSLSessionDeclKind k = 0;
        CHECK(wgsl_session_changed_decl(s, 0, &nm, &nl, &k) == 1, "changed[0]");
        CHECK(nl == 1 && nm && nm[0] == 'f', "changed name is f");
        CHECK(k == WGSL_SESSION_DECL_FUNCTION, "changed kind function");
        CHECK(wgsl_session_changed_decl(s, 1, &nm, &nl, &k) == 0, "no changed[1]");

        wgsl_session_destroy(s);
    }

    /* Partial TC: error in unchanged body is re-typed (not skipped). */
    {
        WGSLSession *s = wgsl_session_new();
        const char *bad =
            "fn good() -> i32 { return 1; }\n"
            "fn bad() -> i32 { return 1u; }\n";  /* type error */
        const char *bad2 =
            "fn good() -> i32 { return 2; }\n"   /* body changed */
            "fn bad() -> i32 { return 1u; }\n";
        const WGSLResult *r1 = wgsl_session_check(s, bad);
        CHECK(r1 != NULL && !wgsl_ok(r1), "err: a has error");
        const WGSLResult *r2 = wgsl_session_check(s, bad2);
        CHECK(r2 != NULL && !wgsl_ok(r2), "err: b still has error");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->iface_changed == 0, "err: iface stable");
        /* `bad` still has a prior diag → must be re-typed, not skipped. */
        CHECK(i->fn_bodies_skipped == 0, "err: no body skipped (diag carrier)");
        CHECK(i->fn_bodies_typed >= 2, "err: both bodies typed");
        wgsl_session_destroy(s);
    }

    /* Diagnostic carrier is retyped and remapped after preceding edit.. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *bad =
            "fn changed() { let x: i32 = 1; _ = x; }\n"
            "fn bad() -> i32 { return 1u; }\n";
        const char *shifted =
            "fn changed() {\n"
            "  let x: i32 = 1;\n"
            "  _ = x;\n"
            "}\n"
            "fn bad() -> i32 { return 1u; }\n";
        const WGSLResult *r1 = wgsl_session_check(s, bad);
        CHECK(r1 != NULL && !wgsl_ok(r1), "remap: first bad");
        const WGSLDiagnostic *d1 = wgsl_diagnostic(r1, 0);
        CHECK(d1 && d1->line == 2, "remap: original bad line 2");
        const WGSLResult *r2 = wgsl_session_check(s, shifted);
        CHECK(r2 != NULL && !wgsl_ok(r2), "remap: shifted still bad");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->iface_changed == 0, "remap: iface stable");
        CHECK(i->fn_bodies_skipped == 0, "remap: bad body not skipped");
        const WGSLDiagnostic *d2 = wgsl_diagnostic(r2, 0);
        CHECK(d2 && d2->line == 5, "remap: bad line shifted to 5");
        wgsl_session_destroy(s);
    }

    /* Partial TC + hover lazy fill on skipped body. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *a =
            "fn f() { let x: i32 = 1; }\n"
            "fn g() { let y: i32 = 2; }\n";
        const char *b =
            "fn f() { let x: i32 = 3; }\n"  /* f body changed */
            "fn g() { let y: i32 = 2; }\n";
        CHECK(wgsl_ok(wgsl_session_check(s, a)), "hover: a ok");
        const WGSLResult *r = wgsl_session_check(s, b);
        CHECK(r != NULL && wgsl_ok(r), "hover: b ok");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->fn_bodies_skipped >= 1, "hover: g skipped");
        /* Offset of `y` in g — "fn f() { let x: i32 = 3; }\nfn g() { let y"
         * count: find 'y' after second let. */
        const char *yp = strstr(b, "let y");
        CHECK(yp != NULL, "hover: found let y");
        uint32_t yoff = (uint32_t)(yp - b) + 4; /* 'y' */
        WGSLHover h = wgsl_hover_at(r, yoff);
        CHECK(h.present, "hover: present on y");
        CHECK(h.type_repr && strstr(h.type_repr, "i32"), "hover: y is i32");
        wgsl_session_destroy(s);
    }

    /* Signature edit: iface changes. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *a = "fn f(x: i32) -> i32 { return x; }\n";
        const char *b = "fn f(x: u32) -> u32 { return x; }\n";
        CHECK(wgsl_ok(wgsl_session_check(s, a)), "sig: a ok");
        CHECK(wgsl_ok(wgsl_session_check(s, b)), "sig: b ok");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->iface_changed == 1, "sig: interface changed");
        CHECK(i->decls_changed >= 1, "sig: decl changed");
        wgsl_session_destroy(s);
    }

    /* New decl appears. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *a = "fn f() {}\n";
        const char *b = "fn f() {}\nfn g() {}\n";
        CHECK(wgsl_ok(wgsl_session_check(s, a)), "new: a ok");
        CHECK(wgsl_ok(wgsl_session_check(s, b)), "new: b ok");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->decls_changed >= 1, "new: g reported changed/new");
        CHECK(i->decls_total == 2, "new: 2 decls total");
        wgsl_session_destroy(s);
    }

    /* Decl removed. */
    {
        WGSLSession *s = wgsl_session_new();
        const char *a = "fn f() {}\nfn g() {}\n";
        const char *b = "fn f() {}\n";
        CHECK(wgsl_ok(wgsl_session_check(s, a)), "rm: a ok");
        CHECK(wgsl_ok(wgsl_session_check(s, b)), "rm: b ok");
        const WGSLSessionInfo *i = wgsl_session_info(s);
        CHECK(i->decls_removed == 1, "rm: one removed");
        wgsl_session_destroy(s);
    }

    /* 10k-line body edit: typecheck telemetry shows real skip.. */
    {
        enum { LINES = 10000 };
        WGSLSession *s = wgsl_session_new();
        char *a = make_large_body_src(1, LINES);
        char *b = make_large_body_src(2, LINES);
        CHECK(a != NULL && b != NULL, "large: source build");
        const WGSLResult *r1 = wgsl_session_check(s, a ? a : "");
        CHECK(r1 != NULL && wgsl_ok(r1), "large: first ok");
        const WGSLSessionInfo *ia = wgsl_session_info(s);
        uint32_t full_ms = ia ? ia->typecheck_ms : 0;
        CHECK(ia && ia->fn_bodies_skipped == 0, "large: first skips 0");
        CHECK(ia && ia->skipped_nodes == 0, "large: first skipped_nodes 0");
        CHECK(ia && ia->typecheck_ms > 0, "large: first typecheck_ms > 0");

        const WGSLResult *r2 = wgsl_session_check(s, b ? b : "");
        CHECK(r2 != NULL && wgsl_ok(r2), "large: second ok");
        const WGSLSessionInfo *ib = wgsl_session_info(s);
        fprintf(stderr,
                "NOTE  session 10k body edit: full_tc=%u ms partial_tc=%u ms skipped_nodes=%d\n",
                full_ms, ib ? ib->typecheck_ms : 0,
                ib ? ib->skipped_nodes : 0);
        CHECK(ib && ib->iface_changed == 0, "large: iface stable");
        CHECK(ib && ib->fn_bodies_skipped >= 1, "large: stable body skipped");
        CHECK(ib && ib->skipped_nodes >= LINES, "large: skipped ≥10k nodes");
        CHECK(ib && ib->typecheck_ms < full_ms,
              "large: partial typecheck faster than full");
        free(a);
        free(b);
        wgsl_session_destroy(s);
    }

    /* NULL session / empty source. */
    {
        CHECK(wgsl_session_check(NULL, "fn f() {}") == NULL, "null session");
        WGSLSession *s = wgsl_session_new();
        const WGSLResult *r = wgsl_session_check(s, "");
        CHECK(r != NULL && wgsl_ok(r), "empty source ok");
        CHECK(wgsl_session_check(s, "") == r, "empty cache hit");
        wgsl_session_destroy(s);
    }

    fprintf(stderr, "%s  test_session  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
