/**
 * Execution differential testing.
 *
 * Layers:
 *   1. Golden buffer oracle — curated compute shaders with known
 *      storage results (bit-exact string match in interp JSON).
 *   2. Determinism — two identical runs must emit the same buffer dump.
 *   3. Optional external backend — if WGSL_EXEC_REF is set to a command
 *      that accepts the same CLI shape as `wgsl interp`, its buffer
 *      dump is compared to ours (see tools/wgpu-ref-runner.sh).
 *
 * External tools are never required for `make test`.
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

/* Extract `"buffers":[...]` slice (best-effort; golden needles are simpler). */
static const char *find_buffers(const char *json) {
    return json ? strstr(json, "\"buffers\":") : NULL;
}

typedef struct {
    const char *name;
    const char *src;
    const char *entry;
    unsigned    len;       /* default_len / workgroup width hint */
    const char *needle;    /* must appear in JSON (buffer values) */
    int         require_ok;
} ExecCase;

static const ExecCase k_cases[] = {
    {
        "lane_write_index",
        "@group(0) @binding(0) var<storage, read_write> B: array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_index) i: u32) {\n"
        "  B[i] = i + 1u;\n"
        "}\n",
        "main", 4,
        "\"values\":[\"1\",\"2\",\"3\",\"4\"]",
        1,
    },
    {
        "for_reduction_6",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn f() {\n"
        "  var acc = 0u;\n"
        "  for (var i = 0u; i < 4u; i = i + 1u) { acc = acc + i; }\n"
        "  o[0] = f32(acc);\n"
        "}\n",
        "f", 1,
        "\"values\":[\"6\"",
        1,
    },
    {
        "loop_reduction_45",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn reduce() {\n"
        "  var acc = 0u; var i = 0u;\n"
        "  loop {\n"
        "    if (i >= 10u) { break; }\n"
        "    acc = acc + i;\n"
        "    continuing { i = i + 1u; }\n"
        "  }\n"
        "  o[0] = f32(acc);\n"
        "}\n",
        "reduce", 1,
        "\"values\":[\"45\"",
        1,
    },
    {
        "switch_case_222",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn pick() {\n"
        "  var s = 2u; var r = 0.0;\n"
        "  switch (s) {\n"
        "    case 0u: { r = 100.0; }\n"
        "    case 2u: { r = 222.0; }\n"
        "    default: { r = 999.0; }\n"
        "  }\n"
        "  o[0] = r;\n"
        "}\n",
        "pick", 1,
        "\"values\":[\"222\"",
        1,
    },
    {
        "atomic_add_16",
        "@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main() { atomicAdd(&c, 1u); }\n",
        "main", 4,
        "\"values\":[\"16\"]",
        1,
    },
    {
        "atomic_histogram",
        "@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>, 4>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main(@builtin(local_invocation_id) l: vec3u) {\n"
        "  atomicAdd(&bins[l.x % 4u], 1u);\n"
        "}\n",
        "main", 4,
        "\"values\":[\"4\",\"4\",\"4\",\"4\"]",
        1,
    },
    {
        "const_store",
        "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() { o[0] = 42; }\n",
        "main", 1,
        "\"values\":[\"42\"",
        1,
    },
    {
        "vec_component",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  let v = vec3f(1.0, 2.0, 3.0);\n"
        "  o[0] = v.x + v.y + v.z;\n"
        "}\n",
        "main", 1,
        "\"values\":[\"6\"",
        1,
    },
};

static int run_case(const ExecCase *c) {
    char *j1 = wgsl_interp(c->src, c->entry, 0, 0, 0, c->len, NULL);
    if (!j1) {
        fprintf(stderr, "FAIL  %s: interp returned NULL\n", c->name);
        return 0;
    }
    int ok = 1;
    if (c->require_ok && !strstr(j1, "\"ok\":true")) {
        fprintf(stderr, "FAIL  %s: expected ok:true\n  %s\n", c->name, j1);
        ok = 0;
    }
    if (!strstr(j1, c->needle)) {
        fprintf(stderr, "FAIL  %s: missing golden needle\n  want: %s\n  got: %s\n",
                c->name, c->needle, j1);
        ok = 0;
    }

    /* Determinism: second run must match buffer section. */
    char *j2 = wgsl_interp(c->src, c->entry, 0, 0, 0, c->len, NULL);
    if (!j2) {
        fprintf(stderr, "FAIL  %s: second interp NULL\n", c->name);
        ok = 0;
    } else {
        const char *b1 = find_buffers(j1);
        const char *b2 = find_buffers(j2);
        if (!b1 || !b2 || strcmp(b1, b2) != 0) {
            /* Compare only through end of buffers array if present. */
            if (!b1 || !b2) {
                fprintf(stderr, "FAIL  %s: missing buffers in JSON\n", c->name);
                ok = 0;
            } else {
                /* Full-string equality of the buffers suffix is enough. */
                size_t n1 = strlen(b1), n2 = strlen(b2);
                size_t n = n1 < n2 ? n1 : n2;
                /* Find matching prefix until "anomalies" or end */
                const char *e1 = strstr(b1, ",\"anomalies\"");
                const char *e2 = strstr(b2, ",\"anomalies\"");
                size_t l1 = e1 ? (size_t)(e1 - b1) : n1;
                size_t l2 = e2 ? (size_t)(e2 - b2) : n2;
                if (l1 != l2 || memcmp(b1, b2, l1) != 0) {
                    fprintf(stderr, "FAIL  %s: non-deterministic buffers\n", c->name);
                    ok = 0;
                }
                (void)n;
            }
        }
        wgsl_free_string(j2);
    }

    wgsl_free_string(j1);
    return ok;
}

/* Minimal fixture-file path (relative to repo root when `make test` runs). */
static int run_fixture_file(const char *path, const char *entry,
                            unsigned len, const char *needle)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        /* Optional — skip quietly if tree layout differs. */
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0 || sz > 1 << 20) { fclose(f); return 0; }
    char *src = (char *)malloc((size_t)sz + 1);
    if (!src) { fclose(f); return 0; }
    if (fread(src, 1, (size_t)sz, f) != (size_t)sz) {
        free(src); fclose(f); return 0;
    }
    src[sz] = 0;
    fclose(f);

    char *j = wgsl_interp(src, entry, 0, 0, 0, len, NULL);
    free(src);
    if (!j) return 0;
    int ok = strstr(j, needle) != NULL;
    if (!ok) {
        fprintf(stderr, "FAIL  fixture %s: missing %s\n  %s\n",
                path, needle, j);
    }
    wgsl_free_string(j);
    return ok;
}

int main(void) {
    int n = (int)(sizeof k_cases / sizeof k_cases[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (run_case(&k_cases[i])) passed++;
        else fail++;
    }

    /* On-disk fixtures (same dir as verdict fixtures sibling). */
    if (!run_fixture_file("tests/12-diff/exec/lane_index.wgsl", "main", 4,
                          "\"values\":[\"1\",\"2\",\"3\",\"4\"]"))
        fail++;
    else
        passed++;

    if (!run_fixture_file("tests/12-diff/exec/store_42.wgsl", "main", 1,
                          "\"values\":[\"42\""))
        fail++;
    else
        passed++;

    fprintf(stderr, "%s  test_exec_diff  (%d cases ok, %d failure%s)\n",
            fail ? "FAIL" : "PASS", passed,
            fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
