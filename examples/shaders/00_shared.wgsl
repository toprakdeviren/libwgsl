// _shared.wgsl — Helpers prepended to every kernel by engine.js.
//
// IMPORTANT: This file has NO `// --- KERNEL: <name> ---` markers.
// The engine treats its entire content as a global preamble that's
// concatenated in front of each per-file preamble before splitting.
//
// Anything declared here is visible inside every compiled kernel,
// even if unused (WGSL keeps unreferenced module-scope vars but the
// driver/Tint typically dead-code-eliminates them). Keep this file
// tight: helpers used by ≥2 shader files.

const WG: u32 = 256u;
const MAX_WG_DIM: u32 = 65535u;
const NEG_INF: f32 = -3.4028234e38;
// Largest finite f32 value (just under 2^128). Used by is_finite to
// reject ±Inf without relying on isInf (not a WGSL builtin).
const F32_MAX: f32 = 3.4028234e38;

// SIMD/subgroup width assumed by wg_reduce_* — 32 lanes covers Apple M*,
// NVIDIA, Intel, and most AMD GPUs (AMD CDNA/RDNA defaults to 64 but the
// `subgroups` feature on RDNA in Chrome reports 32 as the wave size). With
// WG=256 this gives NUM_SUBGROUPS=8 — the first subgroup does the final
// reduction across the 8 partial sums.
const SUBGROUP_SIZE: u32 = 32u;
const NUM_SUBGROUPS: u32 = 8u;   // WG / SUBGROUP_SIZE

// Linearize a 1D problem dispatched as a 2D grid (workgroup_size = WG×1×1).
// Kernels use this when total workgroups exceed MAX_WG_DIM and the host
// falls back to a 2D dispatch (see dispatch1D in engine.js).
fn flat_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

// Reject NaN and ±Inf without isInf. (x == x) is false only for NaN;
// abs(x) < F32_MAX excludes both ±Inf.
fn is_finite(x: f32) -> bool {
    return (x == x) && (abs(x) < F32_MAX);
}

// Replace NaN/Inf with 0. Used at backward boundaries to keep gradient
// flow stable when a downstream op produces overflow.
fn nan_guard(x: f32) -> f32 {
    return select(0.0, x, is_finite(x));
}

// Workgroup-shared scratch used by wg_reduce_sum / wg_reduce_max. Only
// NUM_SUBGROUPS entries are written, but we keep 256 to avoid changing
// the rest of the kernels that may incidentally probe higher indices.
var<workgroup> sh_red: array<f32, 256>;

// Subgroup-accelerated workgroup-wide reductions. SUBGROUP_SIZE=32 +
// WG=256 → NUM_SUBGROUPS=8. Replaces 8-iteration tree (~8 barriers,
// log₂256 ALU ops) with 2 subgroup ops + 1 barrier.
//
// Subgroup-uniform CF requirement: WGSL uniform analyzer rejects
// `subgroupAdd` inside `if (sg_id == 0u)` because `sg_id` derives from
// `local_invocation_index` (non-uniform). Workaround: ALL subgroups call
// the second subgroupAdd uniformly. Every subgroup loads the same 8
// partial sums (redundant work — 7×) and computes the same final value.
// The redundant ops cost less than the barriers + roundtrip we'd need
// to gate this to a single subgroup, so this is the fast path.
fn wg_reduce_sum(tid: u32, val: f32) -> f32 {
    let sg_sum = subgroupAdd(val);
    let sg_id  = tid / SUBGROUP_SIZE;
    let lane   = tid % SUBGROUP_SIZE;

    if (lane == 0u) { sh_red[sg_id] = sg_sum; }
    workgroupBarrier();

    // All subgroups load partials (only first NUM_SUBGROUPS lanes get
    // real data; rest stay at 0). subgroupAdd called by every subgroup
    // — each ends up with the same total in every lane.
    var v: f32 = 0.0;
    if (lane < NUM_SUBGROUPS) { v = sh_red[lane]; }
    return subgroupAdd(v);
}

fn wg_reduce_max(tid: u32, val: f32) -> f32 {
    let sg_max = subgroupMax(val);
    let sg_id  = tid / SUBGROUP_SIZE;
    let lane   = tid % SUBGROUP_SIZE;

    if (lane == 0u) { sh_red[sg_id] = sg_max; }
    workgroupBarrier();

    // Same pattern as wg_reduce_sum. NEG_INF identity for max so lanes
    // outside the partial range don't pull the result down.
    var v: f32 = NEG_INF;
    if (lane < NUM_SUBGROUPS) { v = sh_red[lane]; }
    return subgroupMax(v);
}
