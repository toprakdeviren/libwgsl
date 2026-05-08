/**
 * backward_ffn.wgsl — FFN element-wise backward.
 *
 *   ffn_gelu_backward:
 *     grad_pre[i] = grad_hidden[i] * gelu'(pre[i])
 *
 *   ffn_swiglu_backward:
 *     grad_gate[i] = grad_hidden[i] * up[i] * silu'(gate[i])
 *     grad_up[i]   = grad_hidden[i] * silu(gate[i])
 *
 * Both kernels are 1D element-wise — no GEMM here. The `matmul` /
 * `matmul_t` / `matmul_at` kernels in linear.wgsl handle the surrounding
 * weight-and-input gradients.
 */

fn gelu_derivative(x_in: f32) -> f32 {
    // d/dx [0.5 * x * (1 + tanh(inner))]   inner = sqrt(2/pi)*(x + 0.044715*x^3)
    let x = clamp(x_in, -100.0, 100.0);
    let sqrt_2_over_pi = 0.7978845608;
    let coeff = 0.044715;
    let x2 = x * x;
    let x3 = x2 * x;
    let inner = sqrt_2_over_pi * (x + coeff * x3);
    let t = tanh(inner);
    let sech2 = 1.0 - t * t;
    let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner;
}

fn sigmoid(x_in: f32) -> f32 {
    // Clamp at ±50: exp(50) is well within f32. exp(100) overflows to Inf,
    // making sigmoid return 0 by accident through 1/(1+Inf). Saturated by ±50 anyway.
    let x = clamp(x_in, -50.0, 50.0);
    return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    return s * (1.0 + x * (1.0 - s));
}


// --- KERNEL: ffn_gelu_backward ---
// grad_pre[i] = grad_hidden[i] * gelu'(pre[i])
@group(0) @binding(0) var<storage, read>       grad_hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       pre_gelu:    array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_pre:    array<f32>;
@group(0) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn ffn_gelu_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    let g  = grad_hidden[i];
    let pg = pre_gelu[i];
    let r  = g * gelu_derivative(pg);
    grad_pre[i] = nan_guard(r);
}


// --- KERNEL: ffn_swiglu_backward ---
// grad_gate[i] = grad_hidden[i] * up[i] * silu'(gate[i])
// grad_up[i]   = grad_hidden[i] * silu(gate[i])
@group(0) @binding(0) var<storage, read>       grad_hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       gate:        array<f32>;
@group(0) @binding(2) var<storage, read>       up:          array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_gate:   array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_up:     array<f32>;
@group(0) @binding(5) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn ffn_swiglu_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                       @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= n) { return; }
    let g  = grad_hidden[i];
    let gt = gate[i];
    let u  = up[i];
    let s  = silu(gt);
    let dg = g * u * silu_derivative(gt);
    let du = g * s;
    grad_gate[i] = nan_guard(dg);
    grad_up[i]   = nan_guard(du);
}
