/**
 * activation.wgsl — element-wise activations + their fused FFN combiners.
 *
 *   gelu_inplace:   x = gelu(x)              (tanh approximation)
 *   silu_inplace:   x = x * sigmoid(x)       (a.k.a. swish)
 *   swiglu_combine: hidden = silu(gate) * up (per-element)
 */

fn gelu(x: f32) -> f32 {
    // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let xc = clamp(x, -100.0, 100.0);
    let inner = 0.7978845608 * (xc + 0.044715 * xc * xc * xc);
    return 0.5 * xc * (1.0 + tanh(inner));
}

fn silu(x: f32) -> f32 {
    // Clamp at ±50: exp(50) ≈ 5e21 (within f32). exp(100) ≈ 2.7e43 overflows f32 (max ~3.4e38),
    // producing Inf intermediates that yield correct results only by accident (Inf in denominator → 0).
    // ±50 saturates sigmoid to ~0/1 already, so result is unchanged.
    let xc = clamp(x, -50.0, 50.0);
    return xc / (1.0 + exp(-xc));
}


// --- KERNEL: gelu_inplace ---
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn gelu_inplace(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    x[i] = gelu(x[i]);
}


// --- KERNEL: swiglu_combine ---
// hidden[i] = silu(gate[i]) * up[i]
@group(0) @binding(0) var<storage, read>       gate:   array<f32>;
@group(0) @binding(1) var<storage, read>       up:     array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn swiglu_combine(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    hidden[i] = silu(gate[i]) * up[i];
}
