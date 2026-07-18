/**
 * §5.3 — module_json golden (byte-exact) + structural key checks.
 *
 * The golden file is the full canonical JSON for a fixed compute shader.
 * Spec-pin is matched dynamically so a pin bump only needs the golden
 * regenerated via `make gen-goldens` (or re-run this test with UPDATE=1).
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        fail += 1;                                                     \
    }                                                                  \
} while (0)

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

/* Replace the "spec_pin":"…" value in golden with the live pin so the
 * rest of the document can stay byte-exact across pin bumps when only
 * the pin string changes. */
static char *patch_spec_pin(const char *golden, const char *pin) {
    const char *key = "\"spec_pin\":\"";
    const char *p = strstr(golden, key);
    if (!p) return NULL;
    p += strlen(key);
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    size_t prefix = (size_t)(p - golden);
    size_t suffix = strlen(end);
    size_t pin_len = strlen(pin);
    char *out = (char *)malloc(prefix + pin_len + suffix + 1);
    if (!out) return NULL;
    memcpy(out, golden, prefix);
    memcpy(out + prefix, pin, pin_len);
    memcpy(out + prefix + pin_len, end, suffix + 1);
    return out;
}

static int has_key_path(const char *json, const char *key) {
    char needle[128];
    snprintf(needle, sizeof needle, "\"%s\":", key);
    return json && strstr(json, needle) != NULL;
}

int main(void) {
    static const char *src =
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n"
        "@compute @workgroup_size(64, 1, 1)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
        "  out[gid.x] = f32(gid.x);\n"
        "}\n";

    WGSLResult *r = wgsl_check(src);
    CHECK(wgsl_ok(r), "simple compute: ok");
    const char *j = wgsl_module_json(r);
    CHECK(j && j[0], "module_json non-empty");

    /* Structural keys always present for a valid module. */
    CHECK(has_key_path(j, "spec_pin"), "key: spec_pin");
    CHECK(has_key_path(j, "entry_points"), "key: entry_points");
    CHECK(has_key_path(j, "pipelines"), "key: pipelines");
    CHECK(has_key_path(j, "resources"), "key: resources");
    CHECK(has_key_path(j, "structs"), "key: structs");
    CHECK(has_key_path(j, "overrides"), "key: overrides");
    CHECK(strstr(j, "\"name\":\"main\"") != NULL, "entry name main");
    CHECK(strstr(j, "\"stage\":\"compute\"") != NULL, "stage compute");
    CHECK(strstr(j, "\"workgroup_size\":[64,1,1]") != NULL, "wgs 64");
    CHECK(strstr(j, "\"name\":\"out\"") != NULL, "resource out");

    /* Byte-exact golden (with live spec pin patched in). */
    const char *paths[] = {
        "tests/09-public/goldens/simple_compute.json",
        "goldens/simple_compute.json",
        NULL,
    };
    char *golden = NULL;
    for (int i = 0; paths[i] && !golden; i++) {
        golden = read_file(paths[i], NULL);
    }
    CHECK(golden != NULL, "golden file readable");
    if (golden) {
        /* Strip trailing newline if present. */
        size_t gl = strlen(golden);
        while (gl > 0 && (golden[gl - 1] == '\n' || golden[gl - 1] == '\r')) {
            golden[--gl] = '\0';
        }
        char *expected = patch_spec_pin(golden, wgsl_spec_pin());
        CHECK(expected != NULL, "patch spec_pin");
        if (expected) {
            int match = strcmp(j, expected) == 0;
            CHECK(match, "module_json byte-exact golden");
            if (!match) {
                fprintf(stderr, "  got len=%zu expected len=%zu\n",
                        strlen(j), strlen(expected));
                fprintf(stderr, "  got:      %.200s…\n", j);
                fprintf(stderr, "  expected: %.200s…\n", expected);
            }
            free(expected);
        }
        free(golden);
    }

    wgsl_free(r);

    if (fail == 0) {
        printf("PASS  test_module_json_golden  (0 failures)\n");
        return 0;
    }
    fprintf(stderr, "FAIL  test_module_json_golden  (%d failures)\n", fail);
    return 1;
}
