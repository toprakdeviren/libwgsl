/**
 * Deterministic mutational fuzzer (no libFuzzer required).
 *
 * Starts from a fixed seed corpus, applies reproducible mutations
 * (byte flip / delete / insert / splice / WGSL keyword inject), and
 * runs `wgsl_check` + `wgsl_format` + session re-check.  The goal is
 * crash-freedom and ASan-clean operation on garbage and near-valid
 * WGSL — not a particular accept/reject ratio.
 *
 * Iterations: FUZZ_ITERS env (default 2500 for `make test` smoke).
 * RNG seed:   FUZZ_SEED env (default 0xC0FFEE).
 */
#include "wgsl.h"

#include <stdint.h>
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

/* tiny xorshift64. */

static uint64_t rng_state = 0xC0FFEEull;

static uint64_t rng_u64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static uint32_t rng_u32(void) { return (uint32_t)rng_u64(); }

static size_t rng_below(size_t n) {
    if (n == 0) return 0;
    return (size_t)(rng_u64() % n);
}

/* seed corpus (always available; files optional). */

static const char *k_seeds[] = {
    "fn f() {}\n",
    "fn f() { let x: i32 = 1; _ = x; }\n",
    "fn { }\n",
    "fn f() { let x = 1i + 1u; }\n",
    "@compute @workgroup_size(1)\nfn main() {}\n",
    ("@compute @workgroup_size(64)\n"
     "fn main(@builtin(global_invocation_id) g: vec3u) { _ = g.x; }\n"),
    "struct S { a: f32, b: u32 }\nfn f(s: S) { _ = s.a; }\n",
    "const N: u32 = 4u;\nfn f() { var a: array<f32, N>; _ = a[0]; }\n",
    "fn f() { for (var i = 0; i < 3; i++) { _ = i; } }\n",
    "fn f() { if (true) { return; } else { discard; } }\n",
    "enable f16;\nfn f() -> f16 { return 1.0h; }\n",
    "/* block */\nfn f() { return; } // trail\n",
    "",
    "@@@@@@@@@@@@@@@@",
    /* Unclosed body — stresses parse recovery. */
    "fn f() { ",
};

static const char *k_snippets[] = {
    "fn ", "let ", "var ", "const ", "struct ", "return ",
    "if (", "else ", "loop ", "for (", "while (",
    "vec3f", "mat4x4f", "array<", "atomic<",
    "@compute ", "@workgroup_size(1) ", "@builtin(position) ",
    "-> i32 ", "-> void ", "; ", "{ ", "} ",
    "1i", "1u", "1.0", "true", "false",
    "/*x*/", "//y\n",
};

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} Buf;

static int buf_reserve(Buf *b, size_t n) {
    if (b->len + n <= b->cap) return 1;
    size_t nc = b->cap ? b->cap : 64;
    while (nc < b->len + n) {
        if (nc > (SIZE_MAX / 2)) return 0;
        nc *= 2;
    }
    char *g = (char *)realloc(b->data, nc);
    if (!g) return 0;
    b->data = g;
    b->cap = nc;
    return 1;
}

static int buf_set(Buf *b, const char *s, size_t n) {
    if (!buf_reserve(b, n + 1)) return 0;
    if (n) memcpy(b->data, s, n);
    b->len = n;
    b->data[n] = 0;
    return 1;
}

static int buf_insert(Buf *b, size_t at, const char *s, size_t n) {
    if (at > b->len) at = b->len;
    if (!buf_reserve(b, n + 1)) return 0;
    memmove(b->data + at + n, b->data + at, b->len - at);
    if (n) memcpy(b->data + at, s, n);
    b->len += n;
    b->data[b->len] = 0;
    return 1;
}

static void buf_delete(Buf *b, size_t at, size_t n) {
    if (at >= b->len) return;
    if (at + n > b->len) n = b->len - at;
    memmove(b->data + at, b->data + at + n, b->len - at - n);
    b->len -= n;
    b->data[b->len] = 0;
}

static void mutate(Buf *b) {
    if (b->len == 0) {
        /* Grow from empty. */
        const char *sn = k_snippets[rng_below(sizeof k_snippets / sizeof k_snippets[0])];
        (void)buf_set(b, sn, strlen(sn));
        return;
    }
    unsigned kind = (unsigned)(rng_u32() % 6u);
    switch (kind) {
    case 0: { /* flip a byte */
        size_t i = rng_below(b->len);
        b->data[i] = (char)(b->data[i] ^ (char)(1u + (rng_u32() & 0x1fu)));
        break;
    }
    case 1: { /* delete a short run */
        size_t i = rng_below(b->len);
        size_t n = 1 + rng_below(8);
        buf_delete(b, i, n);
        break;
    }
    case 2: { /* insert random bytes */
        size_t i = rng_below(b->len + 1);
        char tmp[8];
        size_t n = 1 + rng_below(sizeof tmp);
        for (size_t k = 0; k < n; k++)
            tmp[k] = (char)(32 + (rng_u32() % 95));
        (void)buf_insert(b, i, tmp, n);
        break;
    }
    case 3: { /* inject WGSL snippet */
        size_t i = rng_below(b->len + 1);
        const char *sn = k_snippets[rng_below(sizeof k_snippets / sizeof k_snippets[0])];
        (void)buf_insert(b, i, sn, strlen(sn));
        break;
    }
    case 4: { /* splice with another seed */
        const char *s = k_seeds[rng_below(sizeof k_seeds / sizeof k_seeds[0])];
        size_t sl = strlen(s);
        if (sl == 0) break;
        size_t i = rng_below(b->len + 1);
        size_t take = 1 + rng_below(sl);
        size_t off = rng_below(sl);
        if (off + take > sl) take = sl - off;
        (void)buf_insert(b, i, s + off, take);
        break;
    }
    default: { /* truncate or append brace noise */
        if (rng_u32() & 1u) {
            if (b->len > 0) b->len = rng_below(b->len);
            b->data[b->len] = 0;
        } else {
            (void)buf_insert(b, b->len, "}};;", 4);
        }
        break;
    }
    }
    /* Cap size so a runaway insert cannot OOM the smoke run. */
    if (b->len > 8192) {
        b->len = 8192;
        b->data[b->len] = 0;
    }
}

static void exercise(const char *src, size_t n) {
    /* NUL-terminated copy for APIs that want C strings. */
    char *copy = (char *)malloc(n + 1);
    if (!copy) return;
    if (n) memcpy(copy, src, n);
    copy[n] = 0;

    WGSLResult *r = wgsl_check_n(copy, n);
    if (r) {
        (void)wgsl_ok(r);
        (void)wgsl_diagnostic_count(r);
        (void)wgsl_module_json(r);
        (void)wgsl_error(r);
        /* Lightweight LSP queries on valid offsets. */
        if (n > 0) {
            (void)wgsl_hover_at(r, 0);
            (void)wgsl_definition_at(r, (uint32_t)(n > 1 ? n / 2 : 0));
        }
        wgsl_free(r);
    }

    char *fmt = wgsl_format_n(copy, n);
    if (fmt) wgsl_free_string(fmt);

    WGSLSession *s = wgsl_session_new();
    if (s) {
        (void)wgsl_session_check_n(s, copy, n);
        /* Second check: either cache hit (identical) or re-pipeline. */
        (void)wgsl_session_check_n(s, copy, n);
        wgsl_session_destroy(s);
    }

    free(copy);
}

static unsigned long env_ul(const char *name, unsigned long def,
                            unsigned long cap)
{
    const char *e = getenv(name);
    if (!e || !*e) return def;
    char *end = NULL;
    unsigned long v = strtoul(e, &end, 0);
    if (end == e) return def;
    if (cap && v > cap) v = cap;
    return v;
}

int main(void) {
    unsigned iters = (unsigned)env_ul("FUZZ_ITERS", 2500ul, 10000000ul);
    uint64_t seed  = env_ul("FUZZ_SEED", 0xC0FFEEull, 0); /* no cap */
    if (seed == 0) seed = 0xC0FFEEull;
    rng_state = seed;

    Buf b = {0};
    /* Warm up: exercise every static seed once. */
    for (size_t i = 0; i < sizeof k_seeds / sizeof k_seeds[0]; i++) {
        const char *s = k_seeds[i];
        exercise(s, strlen(s));
    }

    /* Mutational loop. */
    const char *base = k_seeds[0];
    CHECK(buf_set(&b, base, strlen(base)), "buf init");

    for (unsigned i = 0; i < iters; i++) {
        if ((i % 64u) == 0u) {
            /* Periodically reseed from a clean corpus entry. */
            const char *s = k_seeds[rng_below(sizeof k_seeds / sizeof k_seeds[0])];
            (void)buf_set(&b, s, strlen(s));
        }
        mutate(&b);
        exercise(b.data ? b.data : "", b.len);
    }

    free(b.data);

    fprintf(stderr,
            "%s  test_fuzz_mutational  iters=%u seed=0x%llx  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", iters,
            (unsigned long long)seed,
            fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
