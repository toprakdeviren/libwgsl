/* interp_demo.c — native smoke test for the single-invocation interpreter.
 *
 * Drives the public `wgsl_interp` API (src/interp/) and prints the JSON
 * execution trace for a gid-driven compute kernel + an anomaly kernel.
 *
 * Build:  cc -Iinclude -Isrc -Ivendor/unicode/include -std=c11 \
 *             tools/interp_demo.c .build/libwgsl.a \
 *             vendor/unicode/libunicode.native.a -lpthread -lm -o .build/interp
 * Run:    ./.build/interp
 */
#include "wgsl.h"

#include <stdio.h>
#include <stdlib.h>

static void run(const char *title, const char *src, const char *entry,
                unsigned gx, unsigned gy, unsigned gz, unsigned len) {
    printf("══ %s — %s(gid = %u,%u,%u) ══\n", title, entry, gx, gy, gz);
    char *json = wgsl_interp(src, entry, gx, gy, gz, len, NULL);
    printf("%s\n\n", json ? json : "(null)");
    wgsl_free_string(json);
}

int main(void) {
    static const char *SRC =
        "@group(0) @binding(0) var<storage, read_write> out : array<f32>;\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn mandel(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  let cx = f32(gid.x) * 0.35 - 2.0;\n"
        "  let cy = 0.0;\n"
        "  var zx = 0.0;\n"
        "  var zy = 0.0;\n"
        "  var it = 0u;\n"
        "  for (var k = 0u; k < 30u; k = k + 1u) {\n"
        "    let nx = zx * zx - zy * zy + cx;\n"
        "    let ny = 2.0 * zx * zy + cy;\n"
        "    zx = nx;\n"
        "    zy = ny;\n"
        "    if (zx * zx + zy * zy > 4.0) { break; }\n"
        "    it = it + 1u;\n"
        "  }\n"
        "  out[gid.x] = f32(it);\n"
        "}\n"
        "\n"
        "@compute @workgroup_size(64)\n"
        "fn nandemo(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  let d = f32(gid.x) - 3.0;\n"
        "  out[gid.x] = 1.0 / d;\n"
        "}\n";

    run("Demo A · compute trace", SRC, "mandel", 7, 0, 0, 8);
    run("Demo B · anomaly flag",  SRC, "nandemo", 3, 0, 0, 8);
    return 0;
}
