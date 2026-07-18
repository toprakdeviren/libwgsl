/**
 * N-lane debugger product wedge: CI correctness oracle + explainer JSON.
 */
#include "wgsl.h"

#include <stdio.h>
#include <string.h>

static int fail;
#define CHECK(c, m) do { if (!(c)) { \
  fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); fail++; } } while (0)

static int has(const char *s, const char *needle) {
    return s && needle && strstr(s, needle) != NULL;
}

int main(void) {
    {
        const char *src =
            "var<workgroup> tile : array<u32, 4>;\n"
            "@compute @workgroup_size(4)\n"
            "fn main(@builtin(local_invocation_id) l: vec3u) {\n"
            "  tile[l.x] = l.x;\n"
            "  workgroupBarrier();\n"
            "  _ = tile[l.x];\n"
            "}\n";
        char *j = wgsl_debug_oracle(src, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "oracle pass: non-null");
        CHECK(has(j, "\"verdict\":\"pass\""), "oracle pass: verdict pass");
        CHECK(has(j, "\"issue_count\":0"), "oracle pass: zero issues");
        CHECK(has(j, "\"wedge\":\"correctness-oracle+explainer\""),
              "oracle pass: wedge marker");
        wgsl_free_string(j);
    }
    {
        const char *src =
            "var<workgroup> tile : array<u32, 4>;\n"
            "@compute @workgroup_size(4)\n"
            "fn main(@builtin(local_invocation_id) l: vec3u) {\n"
            "  tile[0] = l.x;\n"
            "}\n";
        char *j = wgsl_debug_oracle(src, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "oracle race: non-null");
        CHECK(has(j, "\"verdict\":\"fail\""), "oracle race: verdict fail");
        CHECK(has(j, "\"kind\":\"data_race\""), "oracle race: data_race issue");
        CHECK(has(j, "\"ci\":\"fail\""), "oracle race: CI fail marker");
        CHECK(has(j, "\"why\":\"Two or more lanes touched"),
              "oracle race: explanatory why");
        wgsl_free_string(j);
    }
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
            "@compute @workgroup_size(1)\n"
            "fn main(@builtin(global_invocation_id) g: vec3u) {\n"
            "  let z = f32(g.x);\n"
            "  o[0] = 1.0 / z;\n"
            "}\n";
        char *j = wgsl_debug_oracle(src, "main", 0, 0, 0, 1, NULL);
        CHECK(j != NULL, "oracle numeric: non-null");
        CHECK(has(j, "\"verdict\":\"fail\""), "oracle numeric: verdict fail");
        CHECK(has(j, "\"kind\":\"numeric_anomaly\""),
              "oracle numeric: numeric issue");
        CHECK(has(j, "\"inf\":1"), "oracle numeric: inf counted");
        CHECK(has(j, "non-finite value"), "oracle numeric: explanation");
        wgsl_free_string(j);
    }

    if (fail) {
        fprintf(stderr, "FAIL  test_debug_oracle  (%d failures)\n", fail);
        return 1;
    }
    printf("PASS  test_debug_oracle\n");
    return 0;
}
