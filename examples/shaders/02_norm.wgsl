/**
 * norm.wgsl — RMSNorm forward + backward.
 *
 * RMSNorm:
 *   rms_i = sqrt((1/d) * sum_k x_i[k]^2 + eps)
 *   y_i[d] = x_i[d] / rms_i * w[d]
 *
 * One workgroup per row. Workgroup-shared reduction (no subgroups,
 * for portability). Workgroup size = 256.
 */

// Reductions use the shared sh_red buffer + wg_reduce_sum from _shared.wgsl.
// CAS-based f32 atomic add is inlined per call site (WGSL forbids ptr-to-storage params).


// --- KERNEL: rmsnorm_forward ---
// One workgroup per row of [seq_len, d_model].
@group(0) @binding(0) var<storage, read>       x:    array<f32>;
@group(0) @binding(1) var<storage, read>       w:    array<f32>;
@group(0) @binding(2) var<storage, read_write> y:    array<f32>;
@group(0) @binding(3) var<storage, read_write> rms_out: array<f32>; // [seq_len], saved for backward
@group(0) @binding(4) var<uniform> dims: vec4<u32>; // (seq_len, d_model, _, _)
@group(0) @binding(5) var<uniform> params: vec4<f32>; // (eps, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_forward(@builtin(workgroup_id) wgid: vec3<u32>,
                   @builtin(local_invocation_index) tid: u32) {
    let row = wgid.x;
    let seq_len = dims.x;
    let d_model = dims.y;
    let eps = params.x;
    if (row >= seq_len) { return; }

    let base = row * d_model;

    // Phase 1: sum of squares
    var local_sum = 0.0;
    for (var i = tid; i < d_model; i = i + WG) {
        let v = x[base + i];
        local_sum = local_sum + v * v;
    }
    let total = wg_reduce_sum(tid, local_sum);

    // Phase 2: write normalized output
    let rms_inv = inverseSqrt(total / f32(d_model) + eps);
    if (tid == 0u) { rms_out[row] = rms_inv; }
    for (var i = tid; i < d_model; i = i + WG) {
        y[base + i] = x[base + i] * rms_inv * w[i];
    }
}


// --- KERNEL: rmsnorm_backward ---
// Computes dx and accumulates dw.
//   A = sum_k w[k] * x[k] * dy[k]                   (per row, via wg reduce)
//   dx[d] = w[d] * dy[d] * rms_inv
//         - x[d] * rms_inv^3 * A / d_model
//   dw[d] += x[d] * rms_inv * dy[d]                 (CAS atomic add across rows)
@group(0) @binding(0) var<storage, read>       x:        array<f32>;
@group(0) @binding(1) var<storage, read>       w:        array<f32>;
@group(0) @binding(2) var<storage, read>       dy:       array<f32>;
@group(0) @binding(3) var<storage, read>       rms_in:   array<f32>; // saved [seq_len]
@group(0) @binding(4) var<storage, read_write> dx:       array<f32>;
@group(0) @binding(5) var<storage, read_write> dw:       array<atomic<u32>>;
@group(0) @binding(6) var<uniform> dims: vec4<u32>; // (seq_len, d_model, _, _)

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_backward(@builtin(workgroup_id) wgid: vec3<u32>,
                    @builtin(local_invocation_index) tid: u32) {
    let row = wgid.x;
    let seq_len = dims.x;
    let d_model = dims.y;
    if (row >= seq_len) { return; }

    let base = row * d_model;
    let rms_inv = rms_in[row];

    // Reduce A = sum_k w[k] * x[k] * dy[k]
    var local_a = 0.0;
    for (var i = tid; i < d_model; i = i + WG) {
        local_a = local_a + w[i] * x[base + i] * dy[base + i];
    }
    let A = wg_reduce_sum(tid, local_a);

    let coeff = rms_inv * rms_inv * rms_inv * A / f32(d_model);

    for (var i = tid; i < d_model; i = i + WG) {
        let xi = x[base + i];
        let dyi = dy[base + i];
        let wi = w[i];
        dx[base + i] = wi * dyi * rms_inv - xi * coeff;
        let dw_val = xi * rms_inv * dyi;
        if (is_finite(dw_val) && dw_val != 0.0) {
            var old_bits = atomicLoad(&dw[i]);
            loop {
                let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + dw_val);
                let res = atomicCompareExchangeWeak(&dw[i], old_bits, new_bits);
                if (res.exchanged) { break; }
                old_bits = res.old_value;
            }
        }
    }
}


// ════════════════════════════════════════════════════════════
// Mixed-precision (f16 weight γ) variant
// ════════════════════════════════════════════════════════════
// --- KERNEL: rmsnorm_forward_w16 ---
@group(0) @binding(0) var<storage, read>       x_w16:       array<f32>;
@group(0) @binding(1) var<storage, read>       w_w16:       array<f16>;
@group(0) @binding(2) var<storage, read_write> y_w16:       array<f32>;
@group(0) @binding(3) var<storage, read_write> rms_out_w16: array<f32>;
@group(0) @binding(4) var<uniform> dims_w16: vec4<u32>;
@group(0) @binding(5) var<uniform> params_w16: vec4<f32>;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_forward_w16(@builtin(workgroup_id) wgid: vec3<u32>,
                       @builtin(local_invocation_index) tid: u32) {
    let row = wgid.x;
    let seq_len = dims_w16.x;
    let d_model = dims_w16.y;
    let eps = params_w16.x;
    if (row >= seq_len) { return; }

    let base = row * d_model;

    var local_sum = 0.0;
    for (var i = tid; i < d_model; i = i + WG) {
        let v = x_w16[base + i];
        local_sum = local_sum + v * v;
    }
    let total = wg_reduce_sum(tid, local_sum);

    let rms_inv = inverseSqrt(total / f32(d_model) + eps);
    if (tid == 0u) { rms_out_w16[row] = rms_inv; }
    for (var i = tid; i < d_model; i = i + WG) {
        y_w16[base + i] = x_w16[base + i] * rms_inv * f32(w_w16[i]);
    }
}
