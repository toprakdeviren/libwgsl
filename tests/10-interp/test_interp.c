/**
 * Native interpreter test — drives the public `wgsl_interp` API directly
 * (no Emscripten needed) and asserts the JSON execution trace.  Focuses on
 * loop and branch control flow: loop / continuing /
 * break-if / switch.  (Previously these were silent no-ops, so a loop-based
 * reduction returned its initial value and a switch never picked a case.)
 */
#include "wgsl.h"
#include "internal/interp_priv.h"  /* wgsl_test_race_cells_cap (race-cell cap) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
#define CHECK(cond, msg) do {                                          \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d  %s\n",                 \
                          __FILE__, __LINE__, msg); fail += 1; }       \
} while (0)

/* Run `entry` and assert the JSON trace contains `needle` (e.g. a buffer
 * value).  gid = (gx,0,0); buffer len = 1 so out[0] is unambiguous. */
static void expect(const char *label, const char *src, const char *entry,
                   unsigned gx, const char *needle) {
    char *j = wgsl_interp(src, entry, gx, 0, 0, 1, NULL);
    CHECK(j != NULL, label);
    if (j) {
        int ok = strstr(j, needle) != NULL;
        CHECK(ok, label);
        if (!ok) fprintf(stderr, "  want substring: %s\n  got: %s\n", needle, j);
        wgsl_free_string(j);
    }
}

int main(void) {
    /* loop + continuing: acc = 0+1+…+9 = 45. */
    static const char *LOOP =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn reduce(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var acc = 0u; var i = 0u;\n"
        "  loop { if (i >= 10u) { break; } acc = acc + i;\n"
        "         continuing { i = i + 1u; } }\n"
        "  o[0] = f32(acc);\n"
        "}\n";
    expect("loop reduction -> 45", LOOP, "reduce", 0, "\"values\":[\"45\"");

    /* loop + `break if`: acc = 0+1+…+5 = 15 (break when i>5). */
    static const char *BREAKIF =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn bif(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var acc = 0; var i = 0;\n"
        "  loop { acc = acc + i;\n"
        "         continuing { i = i + 1; break if i > 5; } }\n"
        "  o[0] = f32(acc);\n"
        "}\n";
    expect("loop break-if -> 15", BREAKIF, "bif", 0, "\"values\":[\"15\"");

    /* switch: selector 2 matches `case 2` -> 222. */
    static const char *SWITCH =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn pick(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 2u; var r = 0.0;\n"
        "  switch (s) {\n"
        "    case 0u: { r = 100.0; }\n"
        "    case 2u: { r = 222.0; }\n"
        "    default: { r = 999.0; }\n"
        "  }\n"
        "  o[0] = r;\n"
        "}\n";
    expect("switch case match -> 222", SWITCH, "pick", 0, "\"values\":[\"222\"");

    /* switch: selector 7 matches nothing -> default 999. */
    static const char *SWITCH_DEF =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn pd(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 7u; var r = 0.0;\n"
        "  switch (s) { case 0u: { r = 1.0; } default: { r = 999.0; } }\n"
        "  o[0] = r;\n"
        "}\n";
    expect("switch default -> 999", SWITCH_DEF, "pd", 0, "\"values\":[\"999\"");

    /* Sanity: a plain for-loop reduction still works (regression guard). */
    static const char *FORLOOP =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn f(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var acc = 0u;\n"
        "  for (var i = 0u; i < 4u; i = i + 1u) { acc = acc + i; }\n"
        "  o[0] = f32(acc);\n"   /* 0+1+2+3 = 6 */
        "}\n";
    expect("for reduction -> 6", FORLOOP, "f", 0, "\"values\":[\"6\"");

    /* Atomics: single-lane RMW = load-modify-store, exact at N=1.
     * Histogram-style accumulate: atomicAdd of 0..9 = 45. */
    static const char *ATOMIC_ADD =
        "@group(0) @binding(0) var<storage, read_write> c : atomic<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn h(@builtin(global_invocation_id) g : vec3u) {\n"
        "  for (var i = 0u; i < 10u; i = i + 1u) { atomicAdd(&c, i); }\n"
        "  o[0] = atomicLoad(&c);\n"   /* 45 */
        "}\n";
    expect("atomicAdd histogram -> 45", ATOMIC_ADD, "h", 0, "\"name\":\"c\",\"type\":\"atomic<u32>\",\"values\":[\"45\"]");

    /* atomicMax (signed) and atomicExchange return the old value. */
    static const char *ATOMIC_MAX =
        "@group(0) @binding(0) var<storage, read_write> m : atomic<i32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn f(@builtin(global_invocation_id) g : vec3u) {\n"
        "  atomicMax(&m, 3); atomicMax(&m, -7); atomicMax(&m, 8);\n"
        "  o[0] = atomicExchange(&m, 0);\n"   /* old max = 8 */
        "}\n";
    expect("atomicMax/Exchange -> old 8", ATOMIC_MAX, "f", 0, "\"values\":[\"8\"");

    /* arrayLength(&buf) reports the seeded element count. */
    static const char *ALEN =
        "@group(0) @binding(0) var<storage, read> buf : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn f(@builtin(global_invocation_id) g : vec3u) {\n"
        "  o[0] = arrayLength(&buf);\n"
        "}\n";
    /* interp seeds runtime arrays to `len` elements (default 1 here). */
    expect("arrayLength -> seeded count", ALEN, "f", 0, "\"name\":\"o\"");

    /* atomicCompareExchangeWeak → {old_value, exchanged} (complete result shape).
     * Pack three u32 results into one cell via weighted sum (default_len=1).
     * Success: old=0, exchanged=1, c=5 → 0 + 1*10 + 5*100 = 510. */
    static const char *CAS_OK =
        "@group(0) @binding(0) var<storage, read_write> c : atomic<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn k(@builtin(global_invocation_id) g : vec3u) {\n"
        "  let r = atomicCompareExchangeWeak(&c, 0u, 5u);\n"
        "  let e = select(0u, 1u, r.exchanged);\n"
        "  o[0] = r.old_value + e * 10u + atomicLoad(&c) * 100u;\n"
        "}\n";
    expect("CAS success pack 510", CAS_OK, "k", 0, "\"values\":[\"510\"");

    /* CAS miss: cmp 99 vs 0 → old=0, exchanged=0, c=0 → pack 0. */
    static const char *CAS_MISS =
        "@group(0) @binding(0) var<storage, read_write> c : atomic<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn k(@builtin(global_invocation_id) g : vec3u) {\n"
        "  let r = atomicCompareExchangeWeak(&c, 99u, 5u);\n"
        "  let e = select(0u, 1u, r.exchanged);\n"
        "  o[0] = r.old_value + e * 10u + atomicLoad(&c) * 100u;\n"
        "}\n";
    expect("CAS miss pack 0", CAS_MISS, "k", 0, "\"values\":[\"0\"");

    /* Also check the struct trace form for CAS success. */
    {
        char *j = wgsl_interp(CAS_OK, "k", 0, 0, 0, 1, NULL);
        CHECK(j != NULL, "CAS struct trace");
        if (j) {
            CHECK(strstr(j, "struct(0, true)") != NULL ||
                  strstr(j, "struct(0,true)") != NULL,
                  "CAS result struct(old, exchanged)");
            wgsl_free_string(j);
        }
    }

    /* cooperative ops: real cross-lane primitives. */

    /* subgroupAdd reduces l.x across the whole subgroup: 0+1+…+7 = 28.  Every
     * lane sees the total; no longer a 0-fold, no warning. */
    static const char *SG_ADD =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  o[l.x] = subgroupAdd(l.x);\n"
        "}\n";
    {
        char *j = wgsl_interp(SG_ADD, "main", 0, 0, 0, 8, NULL);
        CHECK(j != NULL, "subgroupAdd: result");
        if (j) {
            CHECK(strstr(j, "\"values\":[\"28\",\"28\",\"28\",\"28\",\"28\",\"28\",\"28\",\"28\"]") != NULL,
                  "subgroupAdd reduces to 28 on every lane");
            CHECK(strstr(j, "\"ok\":true") != NULL, "subgroupAdd: ok, no warning");
            wgsl_free_string(j);
        }
    }

    /* subgroupExclusiveAdd = prefix scan: o[l] = 0+1+…+(l-1). */
    static const char *SG_SCAN =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  o[l.x] = subgroupExclusiveAdd(l.x);\n"
        "}\n";
    {
        char *j = wgsl_interp(SG_SCAN, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"0\",\"0\",\"1\",\"3\",\"6\",\"10\",\"15\",\"21\"]") != NULL,
              "subgroupExclusiveAdd prefix scan");
        wgsl_free_string(j);
    }

    /* subgroupBroadcast(_, 3): all lanes get lane 3's value (3*10 = 30). */
    static const char *SG_BCAST =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  o[l.x] = subgroupBroadcast(l.x * 10u, 3u);\n"
        "}\n";
    {
        char *j = wgsl_interp(SG_BCAST, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"30\",\"30\",\"30\",\"30\",\"30\",\"30\",\"30\",\"30\"]") != NULL,
              "subgroupBroadcast broadcasts lane 3");
        wgsl_free_string(j);
    }

    /* subgroupMax: max over the subgroup = 7 for every lane. */
    static const char *SG_MAX =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  o[l.x] = subgroupMax(l.x);\n"
        "}\n";
    {
        char *j = wgsl_interp(SG_MAX, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"7\",\"7\",\"7\",\"7\",\"7\",\"7\",\"7\",\"7\"]") != NULL,
              "subgroupMax over subgroup");
        wgsl_free_string(j);
    }

    /* N-lane SIMT. */

    /* Whole workgroup runs: each of 4 lanes writes its own storage index using
     * its per-lane @builtin ids.  Focus gid.x=8 lands in workgroup 2 (wgsize=4),
     * so the 4 lanes have gid.x = 8..11 → g.x*100 + l.x*10 = [800,910,1020,1130]. */
    static const char *NLANE =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u,\n"
        "        @builtin(global_invocation_id) g : vec3u) {\n"
        "  o[l.x] = g.x * 100u + l.x * 10u;\n"
        "}\n";
    {
        char *j = wgsl_interp(NLANE, "main", 8, 0, 0, 8, NULL);
        CHECK(j != NULL, "nlane: result");
        if (j) {
            CHECK(strstr(j, "\"lanes\":4") != NULL, "nlane: 4 lanes");
            CHECK(strstr(j, "\"workgroup_size\":[4,1,1]") != NULL, "nlane: wgsize");
            /* lanes 0..3 → [800,910,1020,1130]; shared buffer aliased. */
            CHECK(strstr(j, "\"values\":[\"800\",\"910\",\"1020\",\"1130\",") != NULL,
                  "nlane: per-lane writes to shared buffer");
            CHECK(strstr(j, "\"ok\":true") != NULL, "nlane: ok");
            wgsl_free_string(j);
        }
    }

    /* Divergent `if`: lanes 0,1 take then; 2,3 take else; reconverge after. */
    static const char *DIV_IF =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  if (l.x < 2u) { o[l.x] = 111u; } else { o[l.x] = 222u; }\n"
        "}\n";
    {
        char *j = wgsl_interp(DIV_IF, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"111\",\"111\",\"222\",\"222\",") != NULL,
              "divergent if reconverges");
        wgsl_free_string(j);
    }

    /* Divergent loop: lane l loops l times → o[l] = l.  Per-lane trip counts. */
    static const char *DIV_LOOP =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  var acc = 0u;\n"
        "  for (var i = 0u; i < l.x; i = i + 1u) { acc = acc + 1u; }\n"
        "  o[l.x] = acc;\n"
        "}\n";
    {
        char *j = wgsl_interp(DIV_LOOP, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"0\",\"1\",\"2\",\"3\",") != NULL,
              "divergent loop trip counts");
        wgsl_free_string(j);
    }

    /* Workgroup reduction with real barrier join: 8 lanes fill var<workgroup>
     * shared memory, barrier, lane 0 sums 1+2+…+8 = 36.  Barrier no longer
     * badges the trace partial. */
    static const char *WG_REDUCE =
        "var<workgroup> tile : array<u32, 8>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  tile[l.x] = l.x + 1u;\n"
        "  workgroupBarrier();\n"
        "  if (l.x == 0u) {\n"
        "    var s = 0u;\n"
        "    for (var i = 0u; i < 8u; i = i + 1u) { s = s + tile[i]; }\n"
        "    o[0] = s;\n"
        "  }\n"
        "}\n";
    {
        char *j = wgsl_interp(WG_REDUCE, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "wg reduce: result");
        if (j) {
            CHECK(strstr(j, "\"name\":\"o\",\"type\":\"array<u32>\",\"values\":[\"36\"") != NULL,
                  "workgroup reduction -> 36");
            CHECK(strstr(j, "\"ok\":true") != NULL, "barrier join: ok, no warning");
            CHECK(strstr(j, "\"warnings\":[]") != NULL, "barrier no longer warns");
            wgsl_free_string(j);
        }
    }

    /* Atomics contention: contending lanes serialize correctly
     * because a group-step runs its lanes in sequence. */
    static const char *AT_CONTEND =
        "@group(0) @binding(0) var<storage, read_write> c : atomic<u32>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  atomicAdd(&c, 1u);\n"
        "}\n";
    {
        char *j = wgsl_interp(AT_CONTEND, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"name\":\"c\",\"type\":\"atomic<u32>\",\"values\":[\"16\"]") != NULL,
              "16 lanes atomicAdd(1) -> 16");
        wgsl_free_string(j);
    }
    static const char *AT_HIST =
        "@group(0) @binding(0) var<storage, read_write> bins : array<atomic<u32>, 4>;\n"
        "@compute @workgroup_size(16)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  atomicAdd(&bins[l.x % 4u], 1u);\n"
        "}\n";
    {
        char *j = wgsl_interp(AT_HIST, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"4\",\"4\",\"4\",\"4\"]") != NULL,
              "atomic histogram: each of 4 bins gets 4");
        wgsl_free_string(j);
    }

    /* Review fix: a user call (nested single-lane VM) followed by a cooperative
     * op in the SAME statement must not read the freed active-group array.
     * dbl(l)=2l, subgroupAdd(l)=28 → o[l] = 2l+28 = [28,30,…,42]. */
    static const char *UAF =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "fn dbl(x : u32) -> u32 { return x * 2u; }\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  o[l.x] = dbl(l.x) + subgroupAdd(l.x);\n"
        "}\n";
    {
        char *j = wgsl_interp(UAF, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"values\":[\"28\",\"30\",\"32\",\"34\",\"36\",\"38\",\"40\",\"42\"]") != NULL,
              "user-call then subgroup op in one statement (no UAF)");
        wgsl_free_string(j);
    }

    /* Review fix: a cooperative op inside a called function is modelled
     * single-lane, so it must flag the trace partial (not silently mis-reduce). */
    static const char *HELPER_SG =
        "enable subgroups;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "fn red(x : u32) -> u32 { return subgroupAdd(x); }\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { o[l.x] = red(l.x); }\n";
    {
        char *j = wgsl_interp(HELPER_SG, "main", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"ok\":false") != NULL, "subgroup in helper -> ok:false");
        CHECK(j && strstr(j, "called function") != NULL, "helper-coop warning present");
        wgsl_free_string(j);
    }

    /* Review fix: `var local = agg` must deep-copy, not alias.  Each lane copies
     * shared workgroup memory into a local and mutates it; the shared array and
     * the peer lanes' copies must be unaffected (o stays 7, not 100/101). */
    static const char *ALIAS =
        "var<workgroup> sh : array<u32, 4>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(2)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  sh[0] = 7u; sh[1] = 8u;\n"
        "  var local = sh;\n"
        "  local[0] = l.x + 100u;\n"
        "  o[l.x] = sh[0];\n"
        "}\n";
    {
        char *j = wgsl_interp(ALIAS, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "alias: result");
        if (j) {
            const char *o = strstr(j, "\"name\":\"o\"");
            CHECK(o && strstr(o, "\"values\":[\"7\",\"7\",") != NULL,
                  "var local = agg deep-copies (no cross-lane / source aliasing)");
            wgsl_free_string(j);
        }
    }

    /* divergence / occupancy. */

    /* Fully convergent: all 8 lanes take the same path → occupancy 1.0, tax 0,
     * and the store line shows all 8 active. */
    static const char *CONV =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { o[l.x] = l.x * 2u; }\n";
    {
        char *j = wgsl_interp(CONV, "main", 0, 0, 0, 8, NULL);
        CHECK(j != NULL, "conv: result");
        if (j) {
            CHECK(strstr(j, "\"occupancy\":1.0000") != NULL, "convergent -> occupancy 1.0");
            CHECK(strstr(j, "\"tax\":0.0000") != NULL, "convergent -> tax 0");
            CHECK(strstr(j, "\"max_active\":8") != NULL, "heatmap max_active 8");
            CHECK(strstr(j, "\"active\":8") != NULL, "per-step active count present");
            wgsl_free_string(j);
        }
    }

    /* Divergent loop (lane l loops l times): occupancy < 1, tax > 0, and the
     * loop-body line's active count drops below N (min_active 1). */
    static const char *DIV =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(8)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  var a = 0u; for (var i = 0u; i < l.x; i = i + 1u) { a = a + 1u; } o[l.x] = a;\n"
        "}\n";
    {
        char *j = wgsl_interp(DIV, "main", 0, 0, 0, 8, NULL);
        CHECK(j != NULL, "div: result");
        if (j) {
            CHECK(strstr(j, "\"divergence\":{") != NULL, "divergence object present");
            CHECK(strstr(j, "\"min_active\":1") != NULL, "divergent loop drops to 1 active lane");
            /* not fully converged */
            CHECK(strstr(j, "\"occupancy\":1.0000") == NULL, "divergent -> occupancy < 1");
            wgsl_free_string(j);
        }
    }

    /* roofline / instruction-mix / cost asymmetry fix. */

    /* Memory-bound: one load + one store + one mul → tiny intensity → the
     * roofline places it on the memory side of every arch's ridge. */
    static const char *MEMB =
        "@group(0) @binding(0) var<storage, read> a : array<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) { o[g.x] = a[g.x] * 2.0; }\n";
    {
        char *j = wgsl_interp(MEMB, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "roofline: result");
        if (j) {
            CHECK(strstr(j, "\"mix\":{\"fadd\":0,\"fmul\":1") != NULL, "mix: one fmul");
            CHECK(strstr(j, "\"roofline\":{") != NULL, "roofline object present");
            CHECK(strstr(j, "\"bound\":\"memory\"") != NULL, "memory-bound verdict");
            wgsl_free_string(j);
        }
    }

    /* Asymmetry fix: a float comparison now counts (cmp bucket + flop), and a
     * unary negate bills.  `if (x < 0.0) { x = -x; }` → cmp≥1, fadd≥1. */
    static const char *ASYM =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var x = f32(g.x) - 5.0;\n"
        "  if (x < 0.0) { x = -x; }\n"
        "  o[g.x] = x;\n"
        "}\n";
    {
        char *j = wgsl_interp(ASYM, "main", 0, 0, 0, 4, NULL);
        CHECK(j != NULL, "asym: result");
        if (j) {
            /* g.x=0 → x=-5 → x<0 true → x=5.  cmp counted, negate counted. */
            CHECK(strstr(j, "\"cmp\":1") != NULL, "float comparison billed (cmp=1)");
            CHECK(strstr(j, "\"values\":[\"5\"") != NULL, "abs via -x works");
            wgsl_free_string(j);
        }
    }

    /* coalescing + bank conflict. */

    /* Coalesced: 32 lanes write consecutive o[l] → one 128B line → coalescing 1.0. */
    static const char *COAL =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(32)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { o[l.x] = 1.0; }\n";
    {
        char *j = wgsl_interp(COAL, "main", 0, 0, 0, 32, NULL);
        CHECK(j && strstr(j, "\"coalescing\":1.0000") != NULL, "consecutive writes coalesce (1.0)");
        wgsl_free_string(j);
    }

    /* Bank conflict: 32 lanes write tile[l*32] → all land on bank 0 → 31 extra
     * serialized shared transactions. */
    static const char *BANK =
        "var<workgroup> tile : array<u32, 1024>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(32)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { tile[l.x * 32u] = l.x; o[l.x] = 1u; }\n";
    {
        char *j = wgsl_interp(BANK, "main", 0, 0, 0, 32, NULL);
        CHECK(j && strstr(j, "\"bank_conflicts\":31") != NULL, "32-way bank conflict -> 31");
        wgsl_free_string(j);
    }

    /* No conflict: tile[l] → distinct banks → 0 conflicts. */
    static const char *NOBANK =
        "var<workgroup> tile : array<u32, 64>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(32)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { tile[l.x] = l.x; o[l.x] = 1u; }\n";
    {
        char *j = wgsl_interp(NOBANK, "main", 0, 0, 0, 32, NULL);
        CHECK(j && strstr(j, "\"bank_conflicts\":0") != NULL, "distinct banks -> 0 conflicts");
        wgsl_free_string(j);
    }

    /* mem_process groups sites/cache-lines in hash tables.  Use a
     * synthetic large warp-step buffer so the guard is deterministic and does
     * not depend on timing. */
    {
        Interp ip;
        WGSLNode sites[2];
        memset(&ip, 0, sizeof ip);
        memset(sites, 0, sizeof sites);
        ip.macc_cap = 4096u;
        ip.macc_n = 4096u;
        ip.macc = (struct MemAcc *)calloc(ip.macc_cap, sizeof *ip.macc);
        CHECK(ip.macc != NULL, "mem_process hash: alloc");
        if (ip.macc) {
            for (uint32_t i = 0; i < 2048u; i++) {
                ip.macc[i].node = &sites[0];
                ip.macc[i].offset = i * 4u;
                ip.macc[i].elem_bytes = 4u;
                ip.macc[i].is_wg = 0u;
            }
            for (uint32_t i = 2048u; i < 4096u; i++) {
                ip.macc[i].node = &sites[1];
                ip.macc[i].offset = 0u;
                ip.macc[i].elem_bytes = 4u;
                ip.macc[i].is_wg = 1u;
            }
            mem_process(&ip);
            CHECK(ip.macc_n == 0, "mem_process hash: consumed buffer");
            CHECK(ip.dram_theoretical == 8192u,
                  "mem_process hash: theoretical bytes");
            CHECK(ip.dram_effective == 8192u,
                  "mem_process hash: cache-line dedupe");
            CHECK(ip.shared_accesses == 1u,
                  "mem_process hash: one workgroup site");
            CHECK(ip.bank_conflicts == 2047u,
                  "mem_process hash: bank conflict count");
            free(ip.macc);
        }
    }

    /* static interval cost band. */

    /* Const-bounded loop → exact static band: 10 iters × (mul+add) = 20 flops,
     * min == max, method "interval", loop trip [10,10] bounded. */
    static const char *CBLOOP =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 0.0; for (var i = 0u; i < 10u; i = i + 1u) { s = s + f32(i) * 2.0; } o[g.x] = s;\n"
        "}\n";
    {
        char *j = wgsl_interp(CBLOOP, "main", 0, 0, 0, 16, NULL);
        CHECK(j != NULL, "band: result");
        if (j) {
            CHECK(strstr(j, "\"cost_band\":{\"method\":\"interval\"") != NULL, "static interval method");
            CHECK(strstr(j, "\"flops\":{\"min\":20,\"max\":20}") != NULL, "exact flop band 20");
            CHECK(strstr(j, "\"savings_hint\":") != NULL, "cost band savings hint");
            CHECK(strstr(j, "\"trip_min\":10,\"trip_max\":10,\"bounded\":true") != NULL, "loop trip [10,10]");
            wgsl_free_string(j);
        }
    }

    /* Data-dependent branch → band spans both paths: else (0 flops) .. then
     * (sin+exp+mul = 9 flops); dd_branches counted. */
    static const char *DDB =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 0.0; if (g.x < 8u) { s = sin(f32(g.x)) * exp(s); } else { s = 1.0; } o[g.x] = s;\n"
        "}\n";
    {
        char *j = wgsl_interp(DDB, "main", 0, 0, 0, 16, NULL);
        CHECK(j != NULL, "ddband: result");
        if (j) {
            CHECK(strstr(j, "\"flops\":{\"min\":0,\"max\":9}") != NULL, "branch fork band [0,9]");
            CHECK(strstr(j, "\"data_dependent_branches\":1") != NULL, "one dd-branch");
            CHECK(strstr(j, "\"savings_hint\":") != NULL, "dd band savings hint");
            wgsl_free_string(j);
        }
    }

    /* Unbounded loop: bound read from a storage buffer (⊤ → unknowable
     * statically) → capped band + method "interval+sampled". */
    static const char *OVL =
        "@group(0) @binding(0) var<storage, read> cnt : array<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 0.0; var n = cnt[0]; for (var i = 0u; i < n; i = i + 1u) { s = s + 1.0; } o[g.x] = s;\n"
        "}\n";
    {
        char *j = wgsl_interp(OVL, "main", 0, 0, 0, 16, NULL);
        CHECK(j && strstr(j, "\"method\":\"interval+sampled\"") != NULL, "unbounded loop -> sampled fallback");
        CHECK(j && strstr(j, "\"bounded\":false") != NULL, "buffer-bounded loop flagged unbounded");
        wgsl_free_string(j);
    }

    /* f16 / packed-math awareness. */

    /* f16 arithmetic → tracked separately, discounted 2× in effective_flops. */
    static const char *F16 =
        "enable f16;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<f16>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var a : f16 = 1.0h; var b : f16 = 2.0h; o[g.x] = a * b + a;\n"
        "}\n";
    {
        char *j = wgsl_interp(F16, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"f16_flops\":2") != NULL, "f16 flops tracked");
        CHECK(j && strstr(j, "\"effective_flops\":1.0") != NULL, "f16 discounted 2x in effective");
        wgsl_free_string(j);
    }

    /* Packed dot → tensor-path eligible. */
    static const char *PACKED =
        "@group(0) @binding(0) var<storage, read_write> o : array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  o[g.x] = dot4I8Packed(0x01020304u, 0x05060708u);\n"
        "}\n";
    {
        char *j = wgsl_interp(PACKED, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"packed_ops\":1") != NULL, "packed op counted");
        CHECK(j && strstr(j, "\"tensor_path\":true") != NULL, "tensor-path flagged");
        wgsl_free_string(j);
    }

    /* data-race linter. */

    /* All lanes write the same workgroup cell with no barrier → data race. */
    static const char *RACE =
        "var<workgroup> tile : array<u32, 4>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { tile[0] = l.x; o[l.x] = tile[0]; }\n";
    {
        char *j = wgsl_interp(RACE, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"races\":{\"count\":0") == NULL, "shared-cell race detected");
        CHECK(j && strstr(j, "write-write") != NULL, "WAW race classified");
        wgsl_free_string(j);
    }

    /* Each lane its own slot → no race. */
    static const char *NORACE =
        "var<workgroup> tile : array<u32, 4>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) { tile[l.x] = l.x; o[l.x] = tile[l.x]; }\n";
    {
        char *j = wgsl_interp(NORACE, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"races\":{\"count\":0") != NULL, "per-lane slots: no race");
        wgsl_free_string(j);
    }

    /* A barrier establishes happens-before → no false positive. */
    static const char *BARROK =
        "var<workgroup> tile : array<u32, 4>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  tile[l.x] = l.x; workgroupBarrier(); o[l.x] = tile[0];\n"
        "}\n";
    {
        char *j = wgsl_interp(BARROK, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"races\":{\"count\":0") != NULL, "barrier separates epochs: no race");
        wgsl_free_string(j);
    }

    /* Race-cell table cap is internal (default 65536).  Inject cap=4 so
     * many distinct workgroup slots trip truncation without a huge run. */
    {
        static const char *MANY =
            "var<workgroup> tile : array<u32, 64>;\n"
            "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
            "@compute @workgroup_size(8)\n"
            "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
            "  tile[l.x] = l.x;\n"
            "  tile[l.x + 8u] = l.x;\n"
            "  tile[l.x + 16u] = l.x;\n"
            "  tile[l.x + 24u] = l.x;\n"
            "  tile[l.x + 32u] = l.x;\n"
            "  o[l.x] = tile[l.x];\n"
            "}\n";
        wgsl_test_race_cells_cap = 4;
        char *j = wgsl_interp(MANY, "main", 0, 0, 0, 8, NULL);
        wgsl_test_race_cells_cap = 0;
        CHECK(j != NULL, "race-cell cap: interp returns JSON");
        CHECK(j && strstr(j, "\"truncated\":true") != NULL,
              "race-cell cap: races.truncated true under injected cap=4");
        /* Common case (default cap) must not set truncated. */
        char *j2 = wgsl_interp(NORACE, "main", 0, 0, 0, 4, NULL);
        CHECK(j2 && strstr(j2, "\"truncated\":true") == NULL,
              "race-cell cap: default cap does not set truncated");
        wgsl_free_string(j);
        wgsl_free_string(j2);
    }

    /* Race cell lookup is hash-indexed, not O(n) over prior cells. */
    {
        WGSLConstEvaluator cev;
        Interp ip;
        WGSLSymbol roots[2];
        memset(&cev, 0, sizeof cev);
        memset(&ip, 0, sizeof ip);
        memset(roots, 0, sizeof roots);
        ip.cev = &cev;
        for (uint32_t i = 0; i < 4096u; i++) {
            cev.simt.cur_lane = i & 31u;
            race_check(&ip, &roots[i & 1u], i * 4u, 1, 1);
        }
        CHECK(ip.ncells == 4096u, "race hash: distinct cells tracked");
        CHECK(ip.cell_htab != NULL && ip.cell_htab_cap >= 8192u,
              "race hash: htab built");
        uint32_t before = ip.ncells;
        for (uint32_t i = 0; i < 4096u; i++) {
            cev.simt.cur_lane = (i + 1u) & 31u;
            race_check(&ip, &roots[i & 1u], i * 4u, 1, 1);
        }
        CHECK(ip.ncells == before, "race hash: repeat lookup reuses cell");
        free(ip.cells);
        free(ip.cell_htab);
    }

    /* time-travel timeline. */

    /* Focus-lane steps carry the barrier epoch and a barrier marker segments
     * the timeline, so the studio can scrub forward/back by epoch. */
    static const char *TT =
        "var<workgroup> tile : array<u32, 4>;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  var a = l.x; tile[l.x] = a; workgroupBarrier(); var b = tile[0]; o[l.x] = b;\n"
        "}\n";
    {
        char *j = wgsl_interp(TT, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"epoch\":0") != NULL, "pre-barrier steps in epoch 0");
        CHECK(j && strstr(j, "\"op\":\"barrier\"") != NULL, "barrier marker in timeline");
        CHECK(j && strstr(j, "\"epoch\":1") != NULL, "post-barrier steps in epoch 1");
        wgsl_free_string(j);
    }

    /* small correctness items. */

    /* Phony assign evaluates its RHS (cost + side effects captured): sqrt = 4
     * SFU flops + 1 add = 5. */
    static const char *PHONY =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) { _ = sqrt(f32(g.x) + 1.0); o[g.x] = 1.0; }\n";
    {
        char *j = wgsl_interp(PHONY, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"flops\":5") != NULL, "phony assign RHS evaluated (cost captured)");
        wgsl_free_string(j);
    }

    /* Override default is folded → override-bounded loop runs concretely (64)
     * and its static band is bounded; seeding N overrides it. */
    static const char *OVF =
        "override N : u32 = 64u;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = 0.0; for (var i = 0u; i < N; i = i + 1u) { s = s + 1.0; } o[g.x] = s;\n"
        "}\n";
    {
        char *j = wgsl_interp(OVF, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"64\"") != NULL, "override default folded (loop runs 64)");
        CHECK(j && strstr(j, "\"trip_min\":64,\"trip_max\":64,\"bounded\":true") != NULL,
              "override loop now statically bounded");
        wgsl_free_string(j);
        char *j2 = wgsl_interp(OVF, "main", 0, 0, 0, 4, "N=5");
        CHECK(j2 && strstr(j2, "\"values\":[\"5\"") != NULL, "override seedable (N=5 -> loop runs 5)");
        wgsl_free_string(j2);
    }

    /* Runtime wrap: u32 `1u << 40u` masks the shift amount (→ 1u<<8 = 256), and
     * i32 INT32_MIN / -1 wraps to INT32_MIN — neither drops the statement. */
    static const char *WRAP =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) { var x : u32 = 1u; o[g.x] = x << 40u; }\n";
    {
        char *j = wgsl_interp(WRAP, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"256\"") != NULL, "u32 shift-amount wrap (1u<<40 = 256)");
        CHECK(j && strstr(j, "\"ok\":true") != NULL, "shift wrap does not drop the statement");
        wgsl_free_string(j);
    }
    static const char *DIVW =
        "@group(0) @binding(0) var<storage, read_write> o : array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) { var a : i32 = -2147483648; var b : i32 = -1; o[g.x] = a / b; }\n";
    {
        char *j = wgsl_interp(DIVW, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"-2147483648\"") != NULL, "i32 MIN/-1 wraps to MIN");
        CHECK(j && strstr(j, "\"ok\":true") != NULL, "div overflow wrap does not drop the statement");
        wgsl_free_string(j);
    }

    /* Adversarial review fixes. */

    /* >> over-range wraps at runtime like << (was: hard error, dropped stmt). */
    static const char *RSHIFT =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) { var x : u32 = 256u; o[g.x] = x >> 40u; }\n";
    {
        char *j = wgsl_interp(RSHIFT, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"1\"") != NULL, "u32 >> amount wrap (256>>8 = 1)");
        CHECK(j && strstr(j, "\"ok\":true") != NULL, ">> wrap does not drop the statement");
        wgsl_free_string(j);
    }

    /* Distinct struct members must not collide into one race cell (no false
     * positive): lane 0 writes s.a[0], lane 1 writes s.b[0]. */
    static const char *STRUCTRACE =
        "struct S { a : array<u32, 4>, b : array<u32, 4> };\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> s : S;\n"
        "@compute @workgroup_size(2)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  if (l.x == 0u) { s.a[0] = 7u; } if (l.x == 1u) { s.b[0] = 9u; } o[l.x] = 1u;\n"
        "}\n";
    {
        char *j = wgsl_interp(STRUCTRACE, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"races\":{\"count\":0") != NULL, "distinct struct members: no false race");
        wgsl_free_string(j);
    }

    /* Else-if chain forks (cheapest..costliest arm), it does not sum the arms:
     * then sqrt=4, else-if sin+cos+add=9, else exp=4 → band [4,9], not [.,13]. */
    static const char *ELIF =
        "@group(0) @binding(0) var<storage, read_write> o : array<f32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var s = f32(g.x);\n"
        "  if (g.x < 2u) { s = sqrt(s); } else if (g.x < 4u) { s = sin(s) + cos(s); } else { s = exp(s); }\n"
        "  o[g.x] = s;\n"
        "}\n";
    {
        char *j = wgsl_interp(ELIF, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"flops\":{\"min\":4,\"max\":9}") != NULL,
              "else-if chain forks (not sums) the cost band");
        wgsl_free_string(j);
    }

    /* Ported from tests/10-wasm/interp_smoke.js (§5.1). */

    /* mandel + Inf anomaly + finite lane (wasm smoke). */
    static const char *MANDEL =
        "@group(0) @binding(0) var<storage, read_write> out : array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn mandel(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  let cx = f32(gid.x) * 0.35 - 2.0;\n"
        "  var zx = 0.0; var zy = 0.0; var it = 0u;\n"
        "  for (var k = 0u; k < 30u; k = k + 1u) {\n"
        "    let nx = zx * zx - zy * zy + cx;\n"
        "    let ny = 2.0 * zx * zy;\n"
        "    zx = nx; zy = ny;\n"
        "    if (zx * zx + zy * zy > 4.0) { break; }\n"
        "    it = it + 1u;\n"
        "  }\n"
        "  out[gid.x] = f32(it);\n"
        "}\n"
        "@compute @workgroup_size(64)\n"
        "fn nandemo(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  out[gid.x] = 1.0 / (f32(gid.x) - 3.0);\n"
        "}\n";
    {
        char *j = wgsl_interp(MANDEL, "mandel", 7, 0, 0, 16, NULL);
        CHECK(j != NULL, "mandel: result");
        if (j) {
            CHECK(strstr(j, "\"values\":[") != NULL, "mandel: buffer values");
            CHECK(strstr(j, "\"5\"") != NULL, "mandel: out[7] contains 5");
            CHECK(strstr(j, "\"ok\":true") != NULL, "mandel: ok");
            wgsl_free_string(j);
        }
        j = wgsl_interp(MANDEL, "nandemo", 3, 0, 0, 16, NULL);
        CHECK(j != NULL, "nandemo Inf: result");
        if (j) {
            CHECK(strstr(j, "\"Inf\"") != NULL, "nandemo: Inf value");
            CHECK(strstr(j, "\"inf\":1") != NULL ||
                  strstr(j, "\"inf\": 1") != NULL,
                  "nandemo: one Inf anomaly");
            wgsl_free_string(j);
        }
        j = wgsl_interp(MANDEL, "nandemo", 5, 0, 0, 16, NULL);
        CHECK(j && strstr(j, "\"0.5\"") != NULL, "nandemo lane 5: 0.5");
        wgsl_free_string(j);
        j = wgsl_interp(MANDEL, "nope", 0, 0, 0, 16, NULL);
        CHECK(j && strstr(j, "\"ok\":false") != NULL, "unknown entry: ok false");
        wgsl_free_string(j);
    }

    /* Cost model (softmaxish vs raw). */
    static const char *COST =
        "@group(0) @binding(0) var<storage, read_write> out : array<f32>;\n"
        "@compute @workgroup_size(64)\n"
        "fn softmaxish(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  let x = f32(gid.x) * 0.01;\n"
        "  out[gid.x] = exp(x) + sqrt(x + 1.0) + tanh(x);\n"
        "}\n"
        "@compute @workgroup_size(64)\n"
        "fn raw(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  out[gid.x] = f32(gid.x) * 2.0 + 1.0;\n"
        "}\n";
    {
        char *j = wgsl_interp(COST, "softmaxish", 3, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"flops\":20") != NULL, "cost softmaxish flops=20");
        CHECK(j && strstr(j, "\"bytes_stored\":4") != NULL, "cost bytes_stored");
        wgsl_free_string(j);
        j = wgsl_interp(COST, "raw", 3, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"flops\":2") != NULL, "cost raw flops=2");
        wgsl_free_string(j);
    }

    /* Uniform seed (cfg.n). */
    static const char *SEED =
        "struct Cfg { n: u32, _pad: u32 }\n"
        "@group(0) @binding(0) var<storage, read_write> out : array<f32>;\n"
        "@group(0) @binding(1) var<uniform> cfg : Cfg;\n"
        "@compute @workgroup_size(1)\n"
        "fn seedloop(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  var acc = 0.0;\n"
        "  for (var i = 0u; i < cfg.n; i = i + 1u) { acc = acc + f32(i) * 2.0; }\n"
        "  out[gid.x] = acc;\n"
        "}\n";
    {
        char *j = wgsl_interp(SEED, "seedloop", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"flops\":0") != NULL, "seed unseeded flops=0");
        wgsl_free_string(j);
        j = wgsl_interp(SEED, "seedloop", 0, 0, 0, 8, "cfg.n=4");
        CHECK(j && strstr(j, "\"flops\":8") != NULL, "seed cfg.n=4 flops=8");
        wgsl_free_string(j);
    }

    /* Augmented assignment + empty for-init. */
    static const char *AUG =
        "@group(0) @binding(0) var<storage, read_write> out : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn aug(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  var acc : u32 = 0u;\n"
        "  for (var i : u32 = 0u; i < 5u; i++) { acc += i; }\n"
        "  var s : u32 = 128u; var cnt : u32 = 0u;\n"
        "  for (; s > 0u; s >>= 1u) { cnt++; }\n"
        "  out[0] = acc;\n"
        "  out[1] = cnt;\n"
        "  out[2] = 7u;  out[2] -= 2u;\n"
        "  out[3] = 3u;  out[3] *= 4u;\n"
        "  out[4] = 1u;  out[4] <<= 4u;\n"
        "  out[5] = 13u; out[5] %= 5u;\n"
        "}\n";
    {
        char *j = wgsl_interp(AUG, "aug", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"ok\":true") != NULL, "aug: terminates");
        CHECK(j && strstr(j,
            "\"values\":[\"10\",\"8\",\"5\",\"12\",\"16\",\"3\"") != NULL,
            "aug: compound-assign values");
        wgsl_free_string(j);
    }

    /* Integer ALU cost. */
    static const char *IOP =
        "@group(0) @binding(0) var<storage, read_write> out : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn iop(@builtin(global_invocation_id) gid : vec3u) {\n"
        "  out[0] = 3u + 4u;\n"
        "  out[1] = 10u - 2u;\n"
        "  out[2] = 6u * 7u;\n"
        "  out[3] = 1u << 3u;\n"
        "  out[4] = u32(5u > 3u);\n"
        "}\n";
    {
        char *j = wgsl_interp(IOP, "iop", 0, 0, 0, 8, NULL);
        CHECK(j && strstr(j, "\"iops\":5") != NULL, "iops: 5 integer ops");
        CHECK(j && strstr(j, "\"flops\":0") != NULL, "iops: 0 flops");
        CHECK(j && strstr(j,
            "\"values\":[\"7\",\"8\",\"42\",\"8\",\"1\"") != NULL,
            "iops: values");
        wgsl_free_string(j);
    }

    /* First-class pointer / lvalue value model. */

    /* (a) let p = &x; *p = 9 — function-scope pointer store. */
    static const char *PTR_STORE =
        "@group(0) @binding(0) var<storage, read_write> o : array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var x = 1;\n"
        "  let p = &x;\n"
        "  *p = 9;\n"
        "  o[0] = x;\n"
        "}\n";
    expect("pointer lvalue (a) *p store through let ptr -> 9", PTR_STORE, "main", 0,
           "\"values\":[\"9\"");

    /* (b) 1-D array via pointer: let p = &a[i]; *p = v. */
    static const char *PTR_INDEX =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var a : array<u32, 4>;\n"
        "  a[0] = 1u; a[1] = 2u; a[2] = 3u; a[3] = 4u;\n"
        "  let p = &a[2];\n"
        "  *p = 99u;\n"
        "  o[0] = a[0]; o[1] = a[1]; o[2] = a[2]; o[3] = a[3];\n"
        "}\n";
    {
        char *j = wgsl_interp(PTR_INDEX, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"1\",\"2\",\"99\",\"4\"") != NULL,
              "pointer lvalue (b) *(&a[i]) store hits only index i");
        wgsl_free_string(j);
    }

    /* (c) Shared workgroup store through pointer races (all lanes *p same cell). */
    static const char *PTR_RACE =
        "var<workgroup> cell : u32;\n"
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  let p = &cell;\n"
        "  *p = l.x;\n"
        "  o[l.x] = cell;\n"
        "}\n";
    {
        char *j = wgsl_interp(PTR_RACE, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"races\":{\"count\":0") == NULL,
              "pointer lvalue (c) *p shared store races across lanes");
        CHECK(j && strstr(j, "write-write") != NULL,
              "pointer lvalue (c) WAW via pointer store path");
        wgsl_free_string(j);
    }

    /* (d) Lane isolation: each lane's private var via & is independent. */
    static const char *PTR_LANE =
        "@group(0) @binding(0) var<storage, read_write> o : array<u32>;\n"
        "@compute @workgroup_size(4)\n"
        "fn main(@builtin(local_invocation_id) l : vec3u) {\n"
        "  var x = 0u;\n"
        "  let p = &x;\n"
        "  *p = l.x + 10u;\n"
        "  o[l.x] = x;\n"
        "}\n";
    {
        char *j = wgsl_interp(PTR_LANE, "main", 0, 0, 0, 4, NULL);
        CHECK(j && strstr(j, "\"values\":[\"10\",\"11\",\"12\",\"13\"") != NULL,
              "pointer lvalue (d) baked lane: each lane's *p is private");
        CHECK(j && strstr(j, "\"races\":{\"count\":0") != NULL,
              "pointer lvalue (d) private ptr stores: no shared race");
        wgsl_free_string(j);
    }

    /* (e) let-copy isolation: *(&x) must not clobber a prior value-copy `let y=x`. */
    static const char *PTR_LETCOPY =
        "@group(0) @binding(0) var<storage, read_write> o : array<i32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) g : vec3u) {\n"
        "  var x = 1;\n"
        "  let y = x;\n"
        "  *(&x) = 9;\n"
        "  o[0] = y;\n"
        "  o[1] = x;\n"
        "}\n";
    {
        char *j = wgsl_interp(PTR_LETCOPY, "main", 0, 0, 0, 2, NULL);
        CHECK(j && strstr(j, "\"values\":[\"1\",\"9\"") != NULL,
              "pointer lvalue (e) let y=x then *(&x)=9 → y==1, x==9");
        wgsl_free_string(j);
    }

    if (fail == 0) { printf("PASS  test_interp\n"); return 0; }
    fprintf(stderr, "FAIL  test_interp  %d check(s) failed\n", fail);
    return 1;
}
