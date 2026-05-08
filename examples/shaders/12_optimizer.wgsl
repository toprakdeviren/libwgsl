/**
 * optimizer.wgsl — AdamW step + global gradient-norm reduction.
 *
 *   reduce_norm_sq:      Σ data[i]² → result (CAS atomic, NaN/Inf-skipped)
 *   finalize_grad_stats: gradNorm = √normSq; clipScale = min(1, max_norm/gradNorm)
 *   adamw_update:        fused AdamW update + gradient clip + grad zeroing
 *
 * AdamW (decoupled weight decay):
 *   m  ← β₁·m + (1-β₁)·g·clip
 *   v  ← β₂·v + (1-β₂)·(g·clip)²
 *   m̂  = m / (1-β₁ᵗ)
 *   v̂  = v / (1-β₂ᵗ)
 *   w  ← w - lr·( m̂ / (√v̂ + ε) + λ·w )
 *
 * All buffers are f32. The atomic accumulator uses CAS on bit-cast f32 because
 * WGSL has no native atomic<f32>. Caller zeros the accumulator before launch.
 */

// --- KERNEL: reduce_norm_sq ---
// Σ data[i]² over [0..n) → result (atomic f32 add via CAS).
// Skips NaN/Inf to avoid contaminating the global norm.
// Caller must initialize `result` to 0 before launch.
@group(0) @binding(0) var<storage, read>       data:   array<f32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(2) var<uniform>             n:      u32;

@compute @workgroup_size(256, 1, 1)
fn reduce_norm_sq(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>,
                  @builtin(local_invocation_index) tid: u32) {
    let grid_size = nwg.x * nwg.y * WG;
    var local: f32 = 0.0;
    var i = flat_id(gid, nwg);
    loop {
        if (i >= n) { break; }
        let v = data[i];
        if (is_finite(v)) { local = local + v * v; }
        i = i + grid_size;
    }

    sh_red[tid] = local;
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { sh_red[tid] = sh_red[tid] + sh_red[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        let val = sh_red[0];
        var old_bits = atomicLoad(&result);
        loop {
            let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
            let res = atomicCompareExchangeWeak(&result, old_bits, new_bits);
            if (res.exchanged) { break; }
            old_bits = res.old_value;
        }
    }
}


// --- KERNEL: finalize_grad_stats ---
// One thread. grad_stats[0] = gradNorm, grad_stats[1] = clipScale.
@group(0) @binding(0) var<storage, read>       norm_sq:    array<f32>;     // [1]
@group(0) @binding(1) var<storage, read_write> grad_stats: array<f32>;     // [2]
@group(0) @binding(2) var<uniform>             max_grad_norm: f32;

@compute @workgroup_size(1, 1, 1)
fn finalize_grad_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let normSq    = max(norm_sq[0], 0.0);
    let gradNorm  = sqrt(normSq);
    let clipScale = min(1.0, max_grad_norm / max(gradNorm, 1e-6));
    grad_stats[0] = gradNorm;
    grad_stats[1] = clipScale;
}


// --- KERNEL: adamw_update ---
// Per-parameter AdamW step. Reads clip_scale from grad_stats[1].
// Zeroes the gradient buffer for the next step.
@group(0) @binding(0) var<storage, read_write> w:          array<f32>;     // master weights
@group(0) @binding(1) var<storage, read_write> g:          array<f32>;     // grad (zeroed after)
@group(0) @binding(2) var<storage, read_write> m_buf:      array<f32>;     // 1st moment
@group(0) @binding(3) var<storage, read_write> v_buf:      array<f32>;     // 2nd moment
@group(0) @binding(4) var<storage, read>       grad_stats: array<f32>;     // [norm, clip]
@group(0) @binding(5) var<uniform>             size:       u32;
@group(0) @binding(6) var<uniform>             hp:         vec4<f32>;      // (lr, beta1, beta2, wd)
@group(0) @binding(7) var<uniform>             bias:       vec4<f32>;      // (beta1_t, beta2_t, eps, _)

@compute @workgroup_size(256, 1, 1)
fn adamw_update(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    if (i >= size) { return; }

    let lr      = hp.x;
    let beta1   = hp.y;
    let beta2   = hp.z;
    let wd      = hp.w;
    let beta1_t = bias.x;
    let beta2_t = bias.y;
    let eps     = bias.z;
    let clip    = grad_stats[1];

    var grad = g[i];
    grad = select(0.0, grad, is_finite(grad));
    grad = grad * clip;

    let m_old = m_buf[i];
    let v_old = v_buf[i];
    let w_old = w[i];

    let m_new = beta1 * m_old + (1.0 - beta1) * grad;
    let v_new = beta2 * v_old + (1.0 - beta2) * grad * grad;

    let b1_corr = 1.0 / max(1.0 - beta1_t, 1e-12);
    let b2_corr = 1.0 / max(1.0 - beta2_t, 1e-12);
    let m_hat   = m_new * b1_corr;
    let v_hat   = max(v_new * b2_corr, 0.0);

    let update = lr * (m_hat / (sqrt(v_hat) + eps) + wd * w_old);
    let upd_ok = is_finite(update);
    let w_new  = select(w_old, w_old - update, upd_ok);

    w[i]     = w_new;
    m_buf[i] = m_new;
    v_buf[i] = v_new;
    g[i]     = 0.0;
}


// ════════════════════════════════════════════════════════════
// 8-bit (int8 packed m/v) variant
// ════════════════════════════════════════════════════════════
/**
 * optimizer_8bit.wgsl — AdamW with int8 quantized first/second moments.
 *
 * Layout (per parameter tensor of size N):
 *   m_packed: array<u32>  size = ceil(N/4) — 4 int8 values per u32
 *   v_packed: array<u32>  same
 *   m_scale:  array<f32>  size = ceil(N/256) — one scale per 256-element block
 *   v_scale:  array<f32>  same
 *
 * Per-block scale: block stores int8 ∈ [-127, 127]. fp32 = int8 * scale.
 * Scale = max_abs_in_block / 127 (recomputed every step).
 *
 * Memory: m + v go from 8 bytes/param (fp32×2) to ~2.03 bytes/param
 *   (1 byte int8 × 2 + 4 bytes scale per 256 elements). ~4× saving.
 *
 * Pipeline (one workgroup per 256-element block):
 *   1. Dequantize old m, v from int8 + scale → fp32 in registers
 *   2. Apply Adam update with current grad (fp32)
 *   3. Write new w (fp32, no quant)
 *   4. Block-wide reduce to find new max(|m_new|), max(|v_new|)
 *   5. Quantize new m, v back to int8 with new scales
 *   6. Pack 4 int8s per u32, store
 *   7. Zero gradient
 *
 * NaN/Inf are guarded at three points: incoming grad, Adam update, and
 * (implicitly) by the clamp-to-int8 step which truncates infinities.
 */

// --- KERNEL: adamw_8bit_update ---
const BLOCK_SIZE: u32 = 256u;       // = workgroup_size

// Per-block staging arrays. Reused across the kernel's phases.
var<workgroup> sh_m_q:   array<i32, 256>;
var<workgroup> sh_v_q:   array<i32, 256>;
var<workgroup> sh_max_m: array<f32, 256>;
var<workgroup> sh_max_v: array<f32, 256>;

// Dispatch: workgroups = ceil(size / 256), each handles one block.
@group(0) @binding(0) var<storage, read_write> w:          array<f32>;
@group(0) @binding(1) var<storage, read_write> g:          array<f32>;
@group(0) @binding(2) var<storage, read_write> m_packed:   array<u32>;
@group(0) @binding(3) var<storage, read_write> v_packed:   array<u32>;
@group(0) @binding(4) var<storage, read_write> m_scale:    array<f32>;
@group(0) @binding(5) var<storage, read_write> v_scale:    array<f32>;
@group(0) @binding(6) var<storage, read>       grad_stats: array<f32>;     // [norm, clip]
@group(0) @binding(7) var<uniform>             size:       u32;
@group(0) @binding(8) var<uniform>             hp:         vec4<f32>;      // (lr, b1, b2, wd)
@group(0) @binding(9) var<uniform>             bias:       vec4<f32>;      // (b1_t, b2_t, eps, _)

@compute @workgroup_size(256, 1, 1)
fn adamw_8bit_update(@builtin(workgroup_id) wgid: vec3<u32>,
                     @builtin(local_invocation_index) tid: u32) {
    let block = wgid.x;
    let i = block * BLOCK_SIZE + tid;
    let valid = i < size;

    let lr      = hp.x;
    let b1      = hp.y;
    let b2      = hp.z;
    let wd      = hp.w;
    let b1_t    = bias.x;
    let b2_t    = bias.y;
    let eps     = bias.z;
    let clip    = grad_stats[1];

    // ── 1) Dequantize old m, v from int8 + scale ──
    let pkg_idx = block * (BLOCK_SIZE / 4u) + (tid / 4u);
    let lane    = tid % 4u;
    var m_old: f32 = 0.0;
    var v_old: f32 = 0.0;
    if (valid) {
        let m_pack = m_packed[pkg_idx];
        let v_pack = v_packed[pkg_idx];
        let m_byte = (m_pack >> (lane * 8u)) & 0xFFu;
        let v_byte = (v_pack >> (lane * 8u)) & 0xFFu;
        // Sign-extend i8: bytes ≥ 128 represent negative numbers.
        let m_q = i32(m_byte) - select(0i, 256i, m_byte >= 128u);
        let v_q = i32(v_byte) - select(0i, 256i, v_byte >= 128u);
        m_old = f32(m_q) * m_scale[block];
        v_old = f32(v_q) * v_scale[block];
        // Defense-in-depth: a corrupted (NaN/Inf) scale would propagate
        // through the Adam update and silently destroy training. Cheaper
        // than asserting host-side every step.
        m_old = nan_guard(m_old);
        v_old = nan_guard(v_old);
    }

    // ── 2) Update m, v with current grad (fp32) ──
    var grad: f32 = 0.0;
    if (valid) {
        grad = g[i];
        grad = select(0.0, grad, is_finite(grad));
        grad = grad * clip;
    }
    let m_new = b1 * m_old + (1.0 - b1) * grad;
    let v_new = b2 * v_old + (1.0 - b2) * grad * grad;

    // ── 3) AdamW update on master weight ──
    //
    // Quant-induced safety: when a small v is dwarfed by a block outlier, it
    // rounds to 0. Dequantized v_hat → sqrt → eps → m_hat / eps is huge and
    // blows up the weight. Two guards:
    //   (a) eps_q  = max(eps, 1e-6)        — denominator floor for 8-bit noise
    //   (b) step   = clamp(raw_step, ±1.0) — proper Adam steps live here anyway
    //                                        (Cauchy-Schwarz: |m̂|/√v̂ ≤ 1 when
    //                                        m,v are correlated). Clipping only
    //                                        bites when quant decorrelates them.
    if (valid) {
        let b1_corr = 1.0 / max(1.0 - b1_t, 1e-12);
        let b2_corr = 1.0 / max(1.0 - b2_t, 1e-12);
        let m_hat   = m_new * b1_corr;
        let v_hat   = max(v_new * b2_corr, 0.0);
        let eps_q   = max(eps, 1e-6);
        let raw     = m_hat / (sqrt(v_hat) + eps_q);
        let step    = clamp(raw, -1.0, 1.0);
        let w_old   = w[i];
        let update  = lr * (step + wd * w_old);
        let upd_ok  = is_finite(update);
        let w_new   = select(w_old, w_old - update, upd_ok);
        w[i] = w_new;
        g[i] = 0.0;
    }

    // ── 4) Block-wide reduce: max(|m_new|), max(|v_new|) ──
    sh_max_m[tid] = abs(m_new);
    sh_max_v[tid] = abs(v_new);
    workgroupBarrier();
    var s = BLOCK_SIZE / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sh_max_m[tid] = max(sh_max_m[tid], sh_max_m[tid + s]);
            sh_max_v[tid] = max(sh_max_v[tid], sh_max_v[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let new_m_scale = max(sh_max_m[0] / 127.0, 1e-12);
    let new_v_scale = max(sh_max_v[0] / 127.0, 1e-12);
    if (tid == 0u) {
        m_scale[block] = new_m_scale;
        v_scale[block] = new_v_scale;
    }

    // ── 5) Quantize: int8 = clamp(round(fp32 / scale), -127, 127) ──
    sh_m_q[tid] = clamp(i32(round(m_new / new_m_scale)), -127i, 127i);
    sh_v_q[tid] = clamp(i32(round(v_new / new_v_scale)), -127i, 127i);
    workgroupBarrier();

    // ── 6) Pack 4 int8s per u32 (one writer thread per group of 4) ──
    if (tid < BLOCK_SIZE / 4u) {
        let base = tid * 4u;
        let m0 = u32(sh_m_q[base + 0u] & 0xFFi);
        let m1 = u32(sh_m_q[base + 1u] & 0xFFi);
        let m2 = u32(sh_m_q[base + 2u] & 0xFFi);
        let m3 = u32(sh_m_q[base + 3u] & 0xFFi);
        let m_pack_val = m0 | (m1 << 8u) | (m2 << 16u) | (m3 << 24u);

        let q0 = u32(sh_v_q[base + 0u] & 0xFFi);
        let q1 = u32(sh_v_q[base + 1u] & 0xFFi);
        let q2 = u32(sh_v_q[base + 2u] & 0xFFi);
        let q3 = u32(sh_v_q[base + 3u] & 0xFFi);
        let v_pack_val = q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u);

        let dst_idx = block * (BLOCK_SIZE / 4u) + tid;
        m_packed[dst_idx] = m_pack_val;
        v_packed[dst_idx] = v_pack_val;
    }
}
