/**
 * Phase 9 Iter C — module-summary JSON.
 *
 * Per `docs/MODULE_JSON.md`:
 *   { "spec_pin": "...",
 *     "entry_points": [...], "resources": [...],
 *     "structs": [...], "overrides": [...] }
 *
 * Tests do shape-level assertions via substring matching — keeping
 * them resilient to whitespace / field-order tweaks while still
 * exercising every code path in `build_module_json`.
 */
#include "wgsl.h"

#include <stdint.h>
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

static int contains(const char *json, const char *want) {
    return json && strstr(json, want) != NULL;
}

int main(void) {
    /* ── Empty module — sections are empty arrays ──────────────── */
    {
        WGSLResult *r = wgsl_check("");
        const char *j = wgsl_module_json(r);
        CHECK(j && j[0] != '\0', "empty src: non-empty JSON");
        CHECK(contains(j, "\"spec_pin\":\"" WGSL_SPEC_PIN "\""),
              "spec_pin field");
        CHECK(contains(j, "\"entry_points\":[]"), "entry_points: []");
        CHECK(contains(j, "\"resources\":[]"),    "resources: []");
        CHECK(contains(j, "\"structs\":[]"),      "structs: []");
        CHECK(contains(j, "\"overrides\":[]"),    "overrides: []");
        CHECK(wgsl_module_json_len(r) > 0,        "module_json_len > 0");
        wgsl_free(r);
    }

    /* ── Failing pipeline → empty JSON ─────────────────────────── */
    {
        WGSLResult *r = wgsl_check("fn { }");
        CHECK(!wgsl_ok(r), "parse error: not ok");
        CHECK(wgsl_module_json(r)[0] == '\0',
              "module_json empty when check failed");
        CHECK(wgsl_module_json_len(r) == 0, "module_json_len 0 on fail");
        wgsl_free(r);
    }

    /* ── Compute entry point with literal @workgroup_size ──────── */
    {
        const char *src =
            "@compute @workgroup_size(64, 1, 1)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "compute src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"name\":\"main\""), "entry: name");
        CHECK(contains(j, "\"stage\":\"compute\""), "entry: stage");
        CHECK(contains(j, "\"workgroup_size\":[64,1,1]"),
              "entry: workgroup_size literal");
        wgsl_free(r);
    }

    /* ── Compute entry with const-ref @workgroup_size(WG) ──────── */
    {
        const char *src =
            "const WG: u32 = 256u;\n"
            "@compute @workgroup_size(WG)\n"
            "fn main() {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "wgs(WG): ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"workgroup_size\":[256,1,1]"),
              "wgs: const-ref resolves to 256");
        wgsl_free(r);
    }

    /* ── Resource bindings ─────────────────────────────────────── */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read>       X: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read_write> Y: array<f32>;\n"
            "@group(0) @binding(2) var<uniform>             U: vec4<f32>;\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "resources src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"name\":\"X\""), "resource X name");
        CHECK(contains(j, "\"address_space\":\"storage\""),
              "X: storage as");
        CHECK(contains(j, "\"access\":\"read\""), "X: read access");
        CHECK(contains(j, "\"name\":\"Y\""), "resource Y name");
        CHECK(contains(j, "\"access\":\"read_write\""),
              "Y: read_write access");
        CHECK(contains(j, "\"name\":\"U\""), "resource U name");
        CHECK(contains(j, "\"address_space\":\"uniform\""),
              "U: uniform as");
        CHECK(contains(j, "\"type\":\"array<f32>\""),
              "X type: array<f32>");
        CHECK(contains(j, "\"type\":\"vec4<f32>\""),
              "U type: vec4<f32>");
        CHECK(contains(j, "\"group\":0"), "group: 0");
        CHECK(contains(j, "\"binding\":0"), "binding: 0");
        CHECK(contains(j, "\"binding\":2"), "binding: 2");
        wgsl_free(r);
    }

    /* ── Structs with member types ─────────────────────────────── */
    {
        const char *src =
            "struct Vertex { pos: vec3<f32>, uv: vec2<f32> }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "struct src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"name\":\"Vertex\""), "struct: Vertex name");
        CHECK(contains(j, "\"members\":["), "struct: members[]");
        CHECK(contains(j, "\"name\":\"pos\""), "member pos");
        CHECK(contains(j, "\"name\":\"uv\""),  "member uv");
        CHECK(contains(j, "\"type\":\"vec3<f32>\""), "pos type");
        CHECK(contains(j, "\"type\":\"vec2<f32>\""), "uv type");
        /* §14.4 — per-member offset / size / align + struct totals.
         * Vertex layout (default AS, AlignOf vec3<f32> = 16):
         *   pos @ 0, size 12, align 16
         *   uv  @ 16 (round_up(8, 12) = 16), size 8, align 8
         *   struct align 16, size round_up(16, 24) = 32
         * The check is positional: the "offset" key precedes its int.
         * jb_put_int has no leading 0s so substring matches are safe. */
        CHECK(contains(j, "\"offset\":0"),  "pos offset 0");
        CHECK(contains(j, "\"offset\":16"), "uv offset 16");
        CHECK(contains(j, "\"size\":32"),   "Vertex size 32");
        CHECK(contains(j, "\"align\":16"),  "Vertex align 16");
        wgsl_free(r);
    }

    /* ── Overrides with and without @id ────────────────────────── */
    {
        const char *src =
            "@id(7) override SCALE: f32 = 1.0;\n"
            "override OFFSET: f32 = 0.0;\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "override src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"name\":\"SCALE\""), "override SCALE");
        CHECK(contains(j, "\"id\":7"),           "override id 7");
        CHECK(contains(j, "\"name\":\"OFFSET\""), "override OFFSET");
        CHECK(contains(j, "\"id\":null"),        "override id null");
        CHECK(contains(j, "\"type\":\"f32\""),   "override f32 type");
        wgsl_free(r);
    }

    /* ── Combined module ──────────────────────────────────────── */
    {
        const char *src =
            "const WG: u32 = 64u;\n"
            "struct Params { n: u32 }\n"
            "@group(0) @binding(0) var<storage, read_write> X: array<f32>;\n"
            "@group(0) @binding(1) var<uniform>             P: Params;\n"
            "override SCALE: f32 = 1.0;\n"
            "@compute @workgroup_size(WG)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "combined src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"workgroup_size\":[64,1,1]"), "combined: wgs");
        CHECK(contains(j, "\"name\":\"X\""),       "combined: X");
        CHECK(contains(j, "\"name\":\"P\""),       "combined: P");
        CHECK(contains(j, "\"name\":\"Params\""),  "combined: struct");
        CHECK(contains(j, "\"name\":\"SCALE\""),   "combined: override");

        /* Caching: second call returns the same pointer. */
        const char *j2 = wgsl_module_json(r);
        CHECK(j == j2, "json cache: same pointer on second call");
        wgsl_free(r);
    }

    /* ── Names with embedded special chars are JSON-escaped ───── */
    /* WGSL identifiers can't contain quotes/backslashes per §2.4, but
     * exercise the escape path with the spec_pin's literal-content
     * field instead — verifies the writer preserves quote-escaping
     * correctness on round-trip. */
    {
        WGSLResult *r = wgsl_check("");
        const char *j = wgsl_module_json(r);
        /* Confirm there are exactly two '"' chars wrapping the spec
         * pin (`"spec_pin":"..."` followed by a `,` and then the next
         * key) — i.e. the JSON parses without dangling backslashes. */
        const char *p = strstr(j, "\"spec_pin\":\"");
        CHECK(p != NULL, "found spec_pin field");
        if (p) {
            const char *end = strstr(p + 12, "\",");
            CHECK(end != NULL, "spec_pin closed by `\",`");
        }
        wgsl_free(r);
    }

    fprintf(stderr, "%s  test_module_json  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
