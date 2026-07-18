/**
 * WGSL-native golden reference conformance suite.
 *
 * Pins interpreter semantics via (shader, entry, len, expected buffer
 * needles) cases.  Each case must be deterministic across two runs.
 *
 * This is the reference corpus for backend divergence hunting: any
 * GPU/driver that disagrees with these needles is either wrong or
 * documents a deliberate numeric difference.
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,m); fail++; } } while(0)

typedef struct {
    const char *name;
    const char *src;
    const char *entry;
    unsigned    len;
    int         expect_ok;
    const char *needles[6]; /* NULL-terminated via last NULL in initializer */
} Case;

static const Case k_cases[] = {
    {
        "store_const_42",
        "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() { o[0] = 42; }\n",
        "main", 1, 1,
        { "\"values\":[\"42\"", NULL },
    },
    {
        "lane_index_plus_one",
        "@group(0) @binding(0) var<storage, read_write> B: array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_index) i: u32) {\n"
        "  B[i] = i + 1u;\n"
        "}\n",
        "main", 4, 1,
        { "\"values\":[\"1\",\"2\",\"3\",\"4\"]", NULL },
    },
    {
        "for_sum_0_to_3",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  var acc = 0u;\n"
        "  for (var i = 0u; i < 4u; i = i + 1u) { acc = acc + i; }\n"
        "  o[0] = f32(acc);\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"6\"", NULL },
    },
    {
        "loop_sum_0_to_9",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  var acc = 0u; var i = 0u;\n"
        "  loop {\n"
        "    if (i >= 10u) { break; }\n"
        "    acc = acc + i;\n"
        "    continuing { i = i + 1u; }\n"
        "  }\n"
        "  o[0] = f32(acc);\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"45\"", NULL },
    },
    {
        "switch_case",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  var s = 2u; var r = 0.0;\n"
        "  switch (s) {\n"
        "    case 0u: { r = 100.0; }\n"
        "    case 2u: { r = 222.0; }\n"
        "    default: { r = 999.0; }\n"
        "  }\n"
        "  o[0] = r;\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"222\"", NULL },
    },
    {
        "atomic_add_nlanes",
        "@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main() { atomicAdd(&c, 1u); }\n",
        "main", 4, 1,
        { "\"values\":[\"16\"]", NULL },
    },
    {
        "atomic_hist_4bins",
        "@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>, 4>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main(@builtin(local_invocation_id) l: vec3u) {\n"
        "  atomicAdd(&bins[l.x % 4u], 1u);\n"
        "}\n",
        "main", 4, 1,
        { "\"values\":[\"4\",\"4\",\"4\",\"4\"]", NULL },
    },
    {
        "vec_sum",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  let v = vec3f(1.0, 2.0, 3.0);\n"
        "  o[0] = v.x + v.y + v.z;\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"6\"", NULL },
    },
    {
        "if_else_true",
        "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  var r = 0;\n"
        "  if (true) { r = 7; } else { r = 9; }\n"
        "  o[0] = r;\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"7\"", NULL },
    },
    {
        "bit_ops",
        "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  o[0] = (0xFu << 2u) | 1u;\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"61\"", NULL }, /* 15<<2 | 1 = 60|1 = 61 */
    },
    {
        "min_max",
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  o[0] = min(3.0, max(1.0, 2.0));\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"2\"", NULL },
    },
    {
        "abs_neg",
        "@group(0) @binding(0) var<storage, read_write> o: array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {\n"
        "  o[0] = abs(-5);\n"
        "}\n",
        "main", 1, 1,
        { "\"values\":[\"5\"", NULL },
    },
};

int main(void) {
    int n = (int)(sizeof k_cases / sizeof k_cases[0]);
    int passed = 0;
    for (int i = 0; i < n; i++) {
        const Case *c = &k_cases[i];
        /* Build needle list for API */
        const char *needles[8];
        int ni = 0;
        for (int k = 0; k < 6 && c->needles[k]; k++)
            needles[ni++] = c->needles[k];
        needles[ni] = NULL;
        if (wgsl_golden_run(c->name, c->src, c->entry, c->len,
                            c->expect_ok, needles))
            passed++;
        else
            fail++;
    }

    /* On-disk fixtures (shared with execution-diff fixtures). */
    {
        static const char *disk[][4] = {
            /* path, entry, len_str, needle */
            { "tests/12-diff/exec/store_42.wgsl", "main", "1", "\"values\":[\"42\"" },
            { "tests/12-diff/exec/lane_index.wgsl", "main", "4",
              "\"values\":[\"1\",\"2\",\"3\",\"4\"]" },
            { "tests/12-diff/exec/reduce_sum.wgsl", "main", "1", "\"values\":[\"6\"" },
        };
        for (size_t i = 0; i < sizeof disk / sizeof disk[0]; i++) {
            FILE *f = fopen(disk[i][0], "rb");
            if (!f) continue; /* optional if cwd differs */
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz <= 0 || sz > 1 << 20) { fclose(f); continue; }
            char *src = (char *)malloc((size_t)sz + 1);
            if (!src) { fclose(f); continue; }
            if (fread(src, 1, (size_t)sz, f) != (size_t)sz) {
                free(src); fclose(f); continue;
            }
            src[sz] = 0;
            fclose(f);
            unsigned len = (unsigned)atoi(disk[i][2]);
            const char *needles[] = { disk[i][3], NULL };
            char label[128];
            snprintf(label, sizeof label, "disk:%s", disk[i][0]);
            if (wgsl_golden_run(label, src, disk[i][1], len, 1, needles))
                passed++;
            else
                fail++;
            free(src);
        }
    }

    fprintf(stderr, "%s  test_golden_ref  (%d cases ok, %d failure%s)\n",
            fail ? "FAIL" : "PASS", passed, fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
