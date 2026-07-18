/**
 * Module-summary JSON tests.
 *
 * Per `docs/libwgsl.md`:
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
    /* Empty module - sections are empty arrays. */
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

    /* Failing pipeline -> empty JSON. */
    {
        WGSLResult *r = wgsl_check("fn { }");
        CHECK(!wgsl_ok(r), "parse error: not ok");
        CHECK(wgsl_module_json(r)[0] == '\0',
              "module_json empty when check failed");
        CHECK(wgsl_module_json_len(r) == 0, "module_json_len 0 on fail");
        wgsl_free(r);
    }

    /* Compute entry point with literal @workgroup_size. */
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

    /* Compute entry with const-ref @workgroup_size(WG). */
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

    /* Resource bindings. */
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

    /* Structs with member types. */
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

    /* Overrides with and without @id. */
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

    /* Pipeline override: symbolic @workgroup_size reflection. */
    {
        const char *src =
            "@id(3) override SZ: u32;\n"
            "@compute @workgroup_size(SZ, 2, 1)\n"
            "fn cs() {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "pipeline override wgs src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"workgroup_size\":[\"SZ\",2,1]"),
              "pipeline override wgs is symbolic, not zero");
        CHECK(contains(j, "\"pipelines\":[{\"kind\":\"compute\",\"entry_point\":\"cs\","
                          "\"workgroup_size\":[\"SZ\",2,1]"),
              "pipeline override wgs mirrored in compute pipeline");
        CHECK(contains(j, "\"name\":\"SZ\"") && contains(j, "\"id\":3"),
              "pipeline override metadata retains @id");
        wgsl_free(r);
    }

    {
        const char *src =
            "override SZ: u32;\n"
            "@compute @workgroup_size(SZ + 1u)\n"
            "fn cs() {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "pipeline override expr wgs src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"workgroup_size\":[\"SZ + 1u\",1,1]"),
              "pipeline override expression wgs is symbolic");
        wgsl_free(r);
    }

    /* Combined module. */
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
        /* A struct-typed binding names its struct (not the generic
         * "struct" tag) so embedders can join it to the structs table. */
        CHECK(contains(j, "\"type\":\"Params\""),  "combined: P type is struct name");
        CHECK(contains(j, "\"name\":\"SCALE\""),   "combined: override");

        /* Caching: second call returns the same pointer. */
        const char *j2 = wgsl_module_json(r);
        CHECK(j == j2, "json cache: same pointer on second call");
        wgsl_free(r);
    }

    /* Names with embedded special chars are JSON-escaped. */
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

    /* >64-member struct: layout must not silently fail past the old
     * fixed offsets[64] cap. */
    {
        char src[4096];
        int w = snprintf(src, sizeof src, "struct Big { ");
        for (int i = 0; i < 70; i++)
            w += snprintf(src + w, sizeof src - (size_t)w,
                          "%sf%d: u32", i ? ", " : "", i);
        snprintf(src + w, sizeof src - (size_t)w,
                 " }\n@group(0) @binding(0) var<storage> b: Big;\n"
                 "@compute @workgroup_size(1) fn m() {}\n");
        WGSLResult *r = wgsl_check(src);
        const char *j = wgsl_module_json(r);
        CHECK(wgsl_ok(r),                   ">64-member struct: check ok");
        CHECK(contains(j, "\"size\":280"),  ">64-member struct size = 70*4 (was 0)");
        CHECK(contains(j, "\"f69\""),       "70th member present in reflection");
        wgsl_free(r);
    }

    /* entry-scope bindings: transitive call-graph usage. */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read>       A: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read_write> B: array<f32>;\n"
            "@group(1) @binding(0) var<uniform>             U: vec4<f32>;\n"
            /* helper touches A and U; main calls helper + writes B */
            "fn helper(i: u32) -> f32 { return A[i] + U.x; }\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            "  B[gid.x] = helper(gid.x);\n"
            "}\n"
            /* second entry touches nothing */
            "@compute @workgroup_size(1) fn idle() {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "src: ok");
        const char *j = wgsl_module_json(r);
        /* main: A (via helper), B (direct), U (via helper), sorted. */
        CHECK(contains(j, "\"name\":\"main\",\"stage\":\"compute\",\"workgroup_size\":[64,1,1],"
                          "\"bindings\":[{\"group\":0,\"binding\":0},"
                          "{\"group\":0,\"binding\":1},{\"group\":1,\"binding\":0}]"),
              "main: transitive bindings sorted");
        /* idle: empty binding set. */
        CHECK(contains(j, "\"name\":\"idle\",\"stage\":\"compute\","
                          "\"workgroup_size\":[1,1,1],\"bindings\":[]"),
              "idle: no bindings");
        wgsl_free(r);
    }

    /* A binding referenced only through a chain of helpers is still
     * attributed to the entry (BFS closure), and a resource no entry
     * touches appears in resources[] but in no entry's bindings[]. */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read> T: array<u32>;\n"
            "@group(0) @binding(9) var<storage, read> UNUSED: array<u32>;\n"
            "fn leaf() -> u32 { return T[0]; }\n"
            "fn mid()  -> u32 { return leaf(); }\n"
            "@compute @workgroup_size(1) fn e() { let x = mid(); }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "chain src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"bindings\":[{\"group\":0,\"binding\":0}]"),
              "deep chain reaches T");
        CHECK(!contains(j, "\"binding\":9}"),
              "UNUSED not attributed to entry");
        wgsl_free(r);
    }

    /* per-entry features + workgroup_bytes. */
    {
        const char *src =
            "enable f16;\n"
            "enable subgroups;\n"
            "@group(0) @binding(0) var<storage, read_write> B: array<f32>;\n"
            "var<workgroup> tile: array<f32, 256>;\n"   /* 1024 */
            "var<workgroup> acc: f32;\n"                 /* +4 → 1028 */
            "fn reduce(x: f32) -> f32 { return subgroupAdd(x); }\n"
            "@compute @workgroup_size(64)\n"
            "fn main(@builtin(local_invocation_index) li: u32) {\n"
            "  tile[li] = B[li];\n"
            "  acc = reduce(tile[li]);\n"
            "  let h: f16 = f16(acc);\n"
            "  B[li] = f32(h);\n"
            "}\n"
            "@vertex fn vs() -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "feat src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"features\":[\"subgroups\",\"f16\"]"),
              "main features subgroups+f16");
        CHECK(contains(j, "\"workgroup_bytes\":1028"),
              "main workgroup_bytes packed");
        /* vertex entry: no workgroup_bytes, empty features */
        CHECK(contains(j, "\"name\":\"vs\",\"stage\":\"vertex\",\"bindings\":[],"
                          "\"features\":[]"),
              "vs: no workgroup_bytes, empty features");
        wgsl_free(r);
    }

    /* bind-group-layout synthesis (per-entry GPUBGL style). */
    {
        const char *src =
            "@group(0) @binding(0) var<storage, read_write> B: array<f32, 16>;\n"
            "@group(0) @binding(1) var<storage, read> RO: array<u32, 4>;\n"
            "@group(1) @binding(0) var<uniform> U: vec4<f32>;\n"
            "@group(2) @binding(1) var samp: sampler;\n"
            "@group(2) @binding(2) var tex: texture_2d<f32>;\n"
            "@group(3) @binding(0) var stor: texture_storage_2d<rgba8unorm, write>;\n"
            "fn compute_helper() -> f32 {\n"
            "  return B[0] + U.x + f32(RO[0]);\n"
            "}\n"
            "@compute @workgroup_size(4)\n"
            "fn cs(@builtin(local_invocation_index) li: u32) {\n"
            "  B[li] = compute_helper() + f32(li);\n"
            "  textureStore(stor, vec2<i32>(0, 0), vec4<f32>(0.0));\n"
            "}\n"
            "fn sample_tex() -> vec4<f32> {\n"
            "  return textureSample(tex, samp, vec2<f32>(0.0, 0.0));\n"
            "}\n"
            "@fragment fn fs() -> @location(0) vec4<f32> {\n"
            "  return sample_tex();\n"
            "}\n"
            "@vertex fn vs() -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(0.0);\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "synth src: ok");
        const char *j = wgsl_module_json(r);
        /* compute: storage (rw) + read-only-storage + uniform + storageTexture,
         * grouped by group index, visibility = compute. */
        CHECK(contains(j, "\"name\":\"cs\",\"stage\":\"compute\",\"workgroup_size\":[4,1,1],"
                          "\"bindings\":[{\"group\":0,\"binding\":0},{\"group\":0,\"binding\":1},"
                          "{\"group\":1,\"binding\":0},{\"group\":3,\"binding\":0}],"
                          "\"features\":[],\"bind_group_layouts\":["
                          "{\"group\":0,\"entries\":["
                          "{\"binding\":0,\"visibility\":\"compute\","
                          "\"buffer\":{\"type\":\"storage\",\"hasDynamicOffset\":false,\"minBindingSize\":64}},"
                          "{\"binding\":1,\"visibility\":\"compute\","
                          "\"buffer\":{\"type\":\"read-only-storage\",\"hasDynamicOffset\":false,\"minBindingSize\":16}}]},"
                          "{\"group\":1,\"entries\":[{\"binding\":0,\"visibility\":\"compute\","
                          "\"buffer\":{\"type\":\"uniform\",\"hasDynamicOffset\":false,\"minBindingSize\":16}}]},"
                          "{\"group\":3,\"entries\":[{\"binding\":0,\"visibility\":\"compute\","
                          "\"storageTexture\":{\"access\":\"write-only\",\"format\":\"rgba8unorm\","
                          "\"viewDimension\":\"2d\"}}]}]"),
              "compute bind-group-layout (storage/ro-storage/uniform/storageTexture)");
        CHECK(contains(j, "\"name\":\"fs\",\"stage\":\"fragment\",\"bindings\":[{\"group\":2,\"binding\":1},"
                          "{\"group\":2,\"binding\":2}],\"features\":[],\"bind_group_layouts\":["
                          "{\"group\":2,\"entries\":[{\"binding\":1,\"visibility\":\"fragment\","
                          "\"sampler\":{\"type\":\"filtering\"}},"
                          "{\"binding\":2,\"visibility\":\"fragment\","
                          "\"texture\":{\"sampleType\":\"float\",\"viewDimension\":\"2d\"}}]}"),
              "fragment bind-group-layout entries for sampler+texture");
        /* vertex touches nothing → empty layouts array still present. */
        CHECK(contains(j, "\"name\":\"vs\",\"stage\":\"vertex\",\"bindings\":[],"
                          "\"features\":[],\"bind_group_layouts\":[]"),
              "empty bind_group_layouts when entry touches no resources");
        /* full: pipelines[] — compute + singleton render pair. */
        CHECK(contains(j, "\"pipelines\":[{\"kind\":\"compute\",\"entry_point\":\"cs\","
                          "\"workgroup_size\":[4,1,1],\"workgroup_bytes\":0"),
              "pipelines: compute entry");
        CHECK(contains(j, "\"kind\":\"render\",\"vertex\":\"vs\",\"fragment\":\"fs\""),
              "pipelines: auto-paired render vs+fs");
        /* Shared sampler/tex only on fragment → visibility [fragment] on render. */
        CHECK(contains(j, "\"binding\":1,\"visibility\":[\"fragment\"],\"sampler\"") ||
              contains(j, "\"visibility\":[\"fragment\"],\"sampler\":{\"type\":\"filtering\"}"),
              "render layout: fragment-only sampler visibility array");
        wgsl_free(r);
    }

    /* full: cross-entry visibility union + vertex attrs. */
    {
        const char *src =
            "struct VIn { @location(0) pos: vec3<f32>, @location(1) uv: vec2<f32> }\n"
            "@group(0) @binding(0) var<uniform> Cam: vec4<f32>;\n"
            "@group(0) @binding(1) var samp: sampler;\n"
            "@group(0) @binding(2) var tex: texture_2d<f32>;\n"
            "@vertex fn vs(v: VIn) -> @builtin(position) vec4<f32> {\n"
            "  return vec4<f32>(v.pos + Cam.xyz, 1.0);\n"
            "}\n"
            "@fragment fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
            "  return textureSample(tex, samp, uv) + Cam;\n"
            "}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "union src: ok");
        const char *j = wgsl_module_json(r);
        /* Cam used by both stages → visibility ["vertex","fragment"]. */
        CHECK(contains(j, "\"kind\":\"render\",\"vertex\":\"vs\",\"fragment\":\"fs\""),
              "union: render pipeline pair");
        CHECK(contains(j, "\"binding\":0,\"visibility\":[\"vertex\",\"fragment\"],"
                          "\"buffer\":{\"type\":\"uniform\""),
              "union: Cam visible to both stages");
        CHECK(contains(j, "\"binding\":1,\"visibility\":[\"fragment\"],"
                          "\"sampler\":{\"type\":\"filtering\"}"),
              "union: sampler fragment-only");
        CHECK(contains(j, "\"vertex_attributes\":["
                          "{\"shaderLocation\":0,\"format\":\"float32x3\",\"type\":\"vec3<f32>\"},"
                          "{\"shaderLocation\":1,\"format\":\"float32x2\",\"type\":\"vec2<f32>\"}]"),
              "vertex_attributes from @location inputs");
        CHECK(contains(j, "\"color_targets\":[{\"location\":0,\"type\":\"vec4<f32>\"}]"),
              "color_targets from fragment @location outputs");
        /* No inferred flag when only one candidate. */
        CHECK(!contains(j, "\"inferred\":true"),
              "singleton pair is not marked inferred");
        wgsl_free(r);
    }

    /* Multi vertex×fragment: all pairs, marked inferred; cap sanity. */
    {
        const char *src =
            "@vertex fn va() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@vertex fn vb() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }\n"
            "@fragment fn fa() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@fragment fn fb() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "multi src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"vertex\":\"va\",\"fragment\":\"fa\",\"inferred\":true"),
              "multi: va×fa inferred");
        CHECK(contains(j, "\"vertex\":\"va\",\"fragment\":\"fb\",\"inferred\":true"),
              "multi: va×fb inferred");
        CHECK(contains(j, "\"vertex\":\"vb\",\"fragment\":\"fa\",\"inferred\":true"),
              "multi: vb×fa inferred");
        CHECK(contains(j, "\"vertex\":\"vb\",\"fragment\":\"fb\",\"inferred\":true"),
              "multi: vb×fb inferred");
        CHECK(!contains(j, "\"analysis_truncated\""),
              "multi 2×2 under cap: no analysis_truncated");
        wgsl_free(r);
    }

    /* Render-pipeline cap: V×F product > 16 omits render pipelines and sets
     * analysis_truncated. */
    {
        /* 5 vertex × 4 fragment = 20 > REFLECT_PIPELINE_RENDER_CAP (16). */
        const char *src =
            "@vertex fn v0() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@vertex fn v1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@vertex fn v2() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@vertex fn v3() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@vertex fn v4() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@fragment fn f0() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@fragment fn f1() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@fragment fn f2() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n"
            "@fragment fn f3() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "render cap V×F>16 src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"analysis_truncated\":true"),
              "render cap V×F>16: analysis_truncated true");
        CHECK(!contains(j, "\"kind\":\"render\""),
              "render cap V×F>16: render pipelines omitted");
        wgsl_free(r);
    }

    /* texture/sampler handle records + AS fix. */
    {
        const char *src =
            "@group(0) @binding(0) var samp: sampler;\n"
            "@group(0) @binding(1) var cmp: sampler_comparison;\n"
            "@group(0) @binding(2) var tex: texture_2d<f32>;\n"
            "@group(0) @binding(3) var dep: texture_depth_2d;\n"
            "@group(0) @binding(4) var uarr: texture_2d_array<u32>;\n"
            "@group(0) @binding(5) var stor: texture_storage_2d<rgba8unorm, write>;\n"
            "@fragment fn fs() -> @location(0) vec4<f32> {\n"
            "  return textureSample(tex, samp, vec2<f32>(0.0));\n}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "handle src: ok");
        const char *j = wgsl_module_json(r);
        /* AS fix: handles are "handle", not "private". */
        CHECK(!contains(j, "\"address_space\":\"private\""),
              "no handle mislabelled as private");
        CHECK(contains(j, "\"handle\":{\"kind\":\"sampler\"}"),
              "plain sampler handle");
        CHECK(contains(j, "\"handle\":{\"kind\":\"comparison_sampler\"}"),
              "comparison sampler handle");
        CHECK(contains(j, "\"handle\":{\"kind\":\"texture\",\"view_dimension\":\"2d\","
                          "\"sample_type\":\"float\"}"),
              "sampled texture handle");
        CHECK(contains(j, "\"handle\":{\"kind\":\"depth_texture\",\"view_dimension\":\"2d\","
                          "\"sample_type\":\"depth\"}"),
              "depth texture handle");
        CHECK(contains(j, "\"handle\":{\"kind\":\"texture\",\"view_dimension\":\"2d-array\","
                          "\"sample_type\":\"uint\"}"),
              "arrayed uint texture handle");
        CHECK(contains(j, "\"handle\":{\"kind\":\"storage_texture\",\"view_dimension\":\"2d\","
                          "\"format\":\"rgba8unorm\",\"access\":\"write\"}"),
              "storage texture handle");
        wgsl_free(r);
    }

    /* uniform-AS binding size + member offsets. */
    {
        /* Nested-struct uniform where AS bumps alignment: Inner{x:f32}
         * as an array element gets stride round_up(16, ...) in uniform,
         * so member offsets differ from the AS_NONE baseline. */
        const char *src =
            "struct Uni { a: f32, b: vec3<f32>, c: f32 }\n"
            "@group(0) @binding(0) var<uniform> U: Uni;\n"
            "@group(0) @binding(1) var<storage, read_write> S: array<f32>;\n"
            "@fragment fn fs() -> @location(0) vec4<f32> {\n"
            "  return vec4<f32>(U.a + S[0]);\n}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "size src: ok");
        const char *j = wgsl_module_json(r);
        /* U: uniform struct → size/align + AS-correct member offsets. */
        CHECK(contains(j, "\"type\":\"Uni\",\"size\":32,\"align\":16,\"layout\":["
                          "{\"name\":\"a\",\"offset\":0},"
                          "{\"name\":\"b\",\"offset\":16},"
                          "{\"name\":\"c\",\"offset\":28}]"),
              "uniform binding size + offsets");
        /* S: runtime array → size 0 (unbounded), align 4. */
        CHECK(contains(j, "\"type\":\"array<f32>\",\"size\":0,\"align\":4"),
              "runtime storage size 0");
        wgsl_free(r);
    }

    /* IO reflection: inputs/outputs, struct fan-out. */
    {
        const char *src =
            "struct VIn  { @location(0) pos: vec3<f32>, @location(1) uv: vec2<f32> }\n"
            "struct VOut { @builtin(position) clip: vec4<f32>,\n"
            "              @location(0) @interpolate(perspective, center) col: vec4<f32>,\n"
            "              @location(1) @interpolate(flat) id: u32 }\n"
            "@vertex fn vs(v: VIn, @builtin(vertex_index) vi: u32) -> VOut {\n"
            "  return VOut(vec4<f32>(v.pos, 1.0), vec4<f32>(v.uv, 0.0, 1.0), vi);\n}\n"
            "@fragment fn fs(o: VOut) -> @location(0) vec4<f32> { return o.col; }\n"
            "@compute @workgroup_size(8,8)\n"
            "fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "io src: ok");
        const char *j = wgsl_module_json(r);
        /* vertex inputs: struct params fan out to members + a builtin. */
        CHECK(contains(j, "\"inputs\":[{\"name\":\"pos\",\"type\":\"vec3<f32>\",\"location\":0},"
                          "{\"name\":\"uv\",\"type\":\"vec2<f32>\",\"location\":1},"
                          "{\"name\":\"vi\",\"type\":\"u32\",\"builtin\":\"vertex_index\"}]"),
              "vertex inputs fan-out + builtin");
        /* vertex outputs: builtin + interpolated locations. */
        CHECK(contains(j, "\"outputs\":[{\"name\":\"clip\",\"type\":\"vec4<f32>\",\"builtin\":\"position\"},"
                          "{\"name\":\"col\",\"type\":\"vec4<f32>\",\"location\":0,"
                          "\"interpolate\":{\"type\":\"perspective\",\"sampling\":\"center\"}},"
                          "{\"name\":\"id\",\"type\":\"u32\",\"location\":1,"
                          "\"interpolate\":{\"type\":\"flat\"}}]"),
              "vertex outputs builtin + interpolate");
        /* fragment direct return: no name, just type+location. */
        CHECK(contains(j, "\"outputs\":[{\"type\":\"vec4<f32>\",\"location\":0}]"),
              "fragment direct return output");
        /* compute inputs = builtins; empty outputs. */
        CHECK(contains(j, "\"name\":\"gid\",\"type\":\"vec3<u32>\",\"builtin\":\"global_invocation_id\""),
              "compute builtin input");
        wgsl_free(r);
    }

    /* enable/requires/diagnostic mirror. */
    {
        const char *src =
            "enable f16, subgroups;\n"
            "requires readonly_and_readwrite_storage_textures;\n"
            "diagnostic(off, derivative_uniformity);\n"
            "diagnostic(warning, chromium.unreachable_code);\n"
            "@compute @workgroup_size(1) fn m() {}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "directives src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"enable\":[\"f16\",\"subgroups\"]"),
              "enable mirror");
        CHECK(contains(j, "\"requires\":[\"readonly_and_readwrite_storage_textures\"]"),
              "requires mirror");
        CHECK(contains(j, "\"diagnostics\":[{\"severity\":\"off\",\"rule\":\"derivative_uniformity\"},"
                          "{\"severity\":\"warning\",\"rule\":\"chromium.unreachable_code\"}]"),
              "diagnostic mirror incl. dotted rule");
        wgsl_free(r);
    }

    /* A user struct/alias constructor whose name starts with
     * "subgroup"/"quad" must NOT false-trigger the subgroups feature
     * (review finding): only real predeclared builtins classify. */
    {
        const char *src =
            "alias subgroupVec = vec4<f32>;\n"
            "@fragment fn main() -> @location(0) vec4<f32> {\n"
            "  return subgroupVec(0.0, 0.0, 0.0, 1.0);\n}\n";
        WGSLResult *r = wgsl_check(src);
        CHECK(wgsl_ok(r), "alias-ctor src: ok");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"features\":[]"),
              "subgroup-named constructor not a subgroups feature");
        wgsl_free(r);
    }

    /* empty module: directive arrays present and empty. */
    {
        WGSLResult *r = wgsl_check("");
        const char *j = wgsl_module_json(r);
        CHECK(contains(j, "\"enable\":[]"),      "empty: enable []");
        CHECK(contains(j, "\"requires\":[]"),    "empty: requires []");
        CHECK(contains(j, "\"diagnostics\":[]"), "empty: diagnostics []");
        wgsl_free(r);
    }

    fprintf(stderr, "%s  test_module_json  (%d failure%s)\n",
            fail ? "FAIL" : "PASS", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
