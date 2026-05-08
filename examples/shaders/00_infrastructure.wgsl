/**
 * common.wgsl — Element-wise utilities for LLM training.
 *
 * Each kernel is delimited by `// --- KERNEL: <name> ---`. The host
 * splits at those markers and prepends two preambles to each module:
 *   1. _shared.wgsl  — global helpers (constants, flat_id, wg_reduce_*)
 *   2. file-local    — everything in this file before the first marker
 *
 * All buffers are f32. Atomics are used for multi-writer gradient
 * accumulation (CAS-add via bitcast — WGSL has no native atomic<f32>).
 * Subgroup ops are not used (portable across browsers).
 */

// --- KERNEL: fill_zero ---
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn fill_zero(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = 0.0;
}


// --- KERNEL: fill_const ---
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;
@group(0) @binding(2) var<uniform> value: f32;

@compute @workgroup_size(256, 1, 1)
fn fill_const(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = value;
}


// --- KERNEL: scale ---
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;
@group(0) @binding(2) var<uniform> alpha: f32;

@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>,
         @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = dst[i] * alpha;
}


// --- KERNEL: axpy ---
// dst += alpha * src
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;
@group(0) @binding(3) var<uniform> alpha: f32;

@compute @workgroup_size(256, 1, 1)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = dst[i] + alpha * src[i];
}


// --- KERNEL: copy ---
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = src[i];
}


// --- KERNEL: clamp_inplace ---
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;
@group(0) @binding(2) var<uniform> max_abs: f32;

@compute @workgroup_size(256, 1, 1)
fn clamp_inplace(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = clamp(dst[i], -max_abs, max_abs);
}
