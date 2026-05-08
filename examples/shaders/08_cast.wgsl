/**
 * cast.wgsl — fp32 ↔ fp16 conversions for the mixed precision pipeline.
 *
 * Phase 1 of mixed precision: these kernels exist standalone and aren't yet
 * wired into the forward/backward path. They'll be used in later phases:
 *
 *   Phase 5 (AdamW): master weights stay fp32 for stable accumulation;
 *     after each adamw_update we cast master → forward weight (fp16) once.
 *     `cast_f32_to_f16` runs over each parameter tensor.
 *
 *   Phase 2-4 (forward/backward): kernels load fp16 storage and accumulate
 *     in fp32 inline (no intermediate buffer) — these cast kernels are NOT
 *     part of the hot path. They only run during the optimizer step.
 *
 * f32→f16 clamps to f16's finite range (±65504) before the cast. Healthy
 * training keeps |w| < ~5 so this is normally a no-op, but it costs one
 * instruction per element and prevents a single outlier (e.g. a transient
 * grad spike pre-clip) from poisoning the forward f16 mirror with ±Inf,
 * which would then corrupt all activations through that weight.
 *
 * IMPORTANT: This file is loaded only when adapter.features.has('shader-f16').
 * The host (engine.js) prepends `enable f16;` to every kernel's shared preamble
 * when the feature is available. On Safari (no shader-f16) this file is skipped.
 */


// --- KERNEL: cast_f32_to_f16 ---
// dst[i] = f16(src[i])
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@group(0) @binding(2) var<uniform>             n:   u32;

@compute @workgroup_size(256, 1, 1)
fn cast_f32_to_f16(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = f16(clamp(src[i], -65504.0, 65504.0));
}


// --- KERNEL: cast_f16_to_f32 ---
// dst[i] = f32(src[i])
@group(0) @binding(0) var<storage, read>       src: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             n:   u32;

@compute @workgroup_size(256, 1, 1)
fn cast_f16_to_f32(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    dst[i] = f32(src[i]);
}
