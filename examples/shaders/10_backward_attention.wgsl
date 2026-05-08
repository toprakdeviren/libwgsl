/**
 * backward_attention.wgsl — streaming causal multi-head attention backward + RoPE backward.
 *
 * Two-pass streaming design (no MAX_SEQ cap, O(WG + head_dim) workgroup memory).
 * One workgroup per (query_pos s, head h), 256 threads.
 *
 *   Pass 0:  cache Q row, dO row; init dQ accumulator and D
 *   Pass A:  stream K-blocks of WG=256
 *              recompute P[j] = exp(scale·Q·K[j] − LSE),  dP[j] = Σ_d dO[d]·V[j,d]
 *              D       += Σ_j P[j]·dP[j]   (workgroup-reduced)
 *              dV[j,d] += P[j]·dO[d]       (CAS atomic scatter — D-independent)
 *   Pass B:  stream K-blocks again, with D now known
 *              recompute P[j], dP[j]
 *              dS[j]    = scale · P[j] · (dP[j] − D)
 *              dQ_acc[d]+= Σ_j dS[j]·K[j,d]   ((d_local, k_chunk) parallelized)
 *              dK[j,d] += dS[j]·Q[d]          (CAS atomic scatter)
 *   Final:   write dQ[s,h,d] = current + dQ_acc[d]   (no atomic — owned)
 *
 * dQ has += semantics: if accumulated grads are reused across micro-steps,
 * the host must zero before the first call. dK/dV similarly (atomics
 * accumulate). Compute cost: P, dP each computed twice — ~5% backward
 * step time (attention is ~5% of backward total).
 */

const MAX_HEAD_DIM: u32 = 256u;   // upper bound on head_dim (sh_q sizing)
const MAX_SEQ:      u32 = 1024u;  // upper bound for short-path kernel (sh_p_full)

// Workgroup memory.
// Tint dead-code-eliminates per entry-point, so each pipeline only allocates
// the buffers reachable from its kernel:
//   attention_backward (streaming):
//     sh_q, sh_do, sh_p, sh_dp, sh_partial, sh_dq_acc, sh_D  ≈ 6 KB
//   attention_backward_short (single-pass, seq ≤ 1024):
//     sh_q, sh_do, sh_p_full, sh_dp_full, sh_partial         ≈ 11 KB
//   attention_backward_short_dKdV (key-major dK/dV, no atomics):
//     sh_dk_partial, sh_dv_partial                           ≈ 2 KB
var<workgroup> sh_q:        array<f32, 256>;
var<workgroup> sh_do:       array<f32, 256>;
var<workgroup> sh_partial:  array<f32, 256>;
// streaming variant
var<workgroup> sh_p:        array<f32, 256>;
var<workgroup> sh_dp:       array<f32, 256>;
var<workgroup> sh_dq_acc:   array<f32, 256>;
var<workgroup> sh_D:        array<f32, 1>;
// single-pass variant (full P, dP arrays — no recompute)
var<workgroup> sh_p_full:   array<f32, 1024>;
var<workgroup> sh_dp_full:  array<f32, 1024>;
// dKdV (key-major) variant — partial accumulators for k_par chunk reduce
var<workgroup> sh_dk_partial: array<f32, 256>;
var<workgroup> sh_dv_partial: array<f32, 256>;


// --- KERNEL: attention_backward ---
// Dispatch: workgroups = (seq_len, n_heads, 1)
@group(0) @binding(0) var<storage, read>       Q:   array<f32>;
@group(0) @binding(1) var<storage, read>       K:   array<f32>;
@group(0) @binding(2) var<storage, read>       V:   array<f32>;
@group(0) @binding(3) var<storage, read>       dO:  array<f32>;
@group(0) @binding(4) var<storage, read>       LSE: array<f32>;
@group(0) @binding(5) var<storage, read_write> dQ:  array<f32>;
@group(0) @binding(6) var<storage, read_write> dK:  array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> dV:  array<atomic<u32>>;
@group(0) @binding(8) var<uniform> dims: vec4<u32>;   // (seq_len, n_heads, n_kv_heads, head_dim)
@group(0) @binding(9) var<uniform> params: vec4<f32>; // (scale, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn attention_backward(@builtin(workgroup_id) wgid: vec3<u32>,
                      @builtin(local_invocation_index) tid: u32) {
    let s = wgid.x;
    let h = wgid.y;

    let seq_len    = dims.x;
    let n_heads    = dims.y;
    let n_kv_heads = dims.z;
    let head_dim   = dims.w;
    let scale      = params.x;

    if (s >= seq_len || h >= n_heads) { return; }

    let q_per_group = n_heads / n_kv_heads;
    let kv_h = h / q_per_group;

    let d_model = n_heads * head_dim;
    let kv_dim  = n_kv_heads * head_dim;

    let q_row_base = s * d_model + h * head_dim;
    let causal_len = s + 1u;
    let lse_i = LSE[s * n_heads + h];

    // ── Pass 0: cache Q[s,h,*], dO[s,h,*]; init dQ_acc, D ──
    for (var d = tid; d < head_dim; d = d + WG) {
        sh_q[d]  = Q[q_row_base + d];
        sh_do[d] = dO[q_row_base + d];
    }
    if (tid < head_dim) { sh_dq_acc[tid] = 0.0; }
    if (tid == 0u) { sh_D[0] = 0.0; }
    workgroupBarrier();

    // K-block parallelization parameters (constant)
    let k_par      = max(1u, WG / max(1u, head_dim));    // 4 when head_dim=64
    let n_active     = head_dim * k_par;
    let d_local    = tid / k_par;
    let kc         = tid % k_par;
    let chunk_size = (WG + k_par - 1u) / k_par;

    let n_blocks = (causal_len + WG - 1u) / WG;

    // ════════════════════════════════════════════════════════════════════
    // PASS A: D + dV scatter
    // ════════════════════════════════════════════════════════════════════
    for (var blk = 0u; blk < n_blocks; blk = blk + 1u) {
        let j_start = blk * WG;
        let j = j_start + tid;

        // Compute P[j] and dP[j] for this thread's j
        var P_local: f32  = 0.0;
        var dP_local: f32 = 0.0;
        if (j < causal_len) {
            let kv_base = j * kv_dim + kv_h * head_dim;
            var qk: f32  = 0.0;
            var dov: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                qk  = fma(sh_q[d],  K[kv_base + d], qk);
                dov = fma(sh_do[d], V[kv_base + d], dov);
            }
            P_local  = nan_guard(exp(qk * scale - lse_i));
            dP_local = dov;
        }
        sh_p[tid]  = P_local;
        sh_dp[tid] = dP_local;

        // D += Σ_j P[j]·dP[j]   (workgroup reduce)
        let block_D = wg_reduce_sum(tid, P_local * dP_local);
        if (tid == 0u) { sh_D[0] = sh_D[0] + block_D; }
        workgroupBarrier();   // sh_p, sh_dp visible to all threads for scatter

        // dV scatter: for each (j_local, d), dV[j,d] += P[j]·dO[d]
        // (D-independent — done in Pass A.)
        let total_jd = WG * head_dim;
        for (var k = tid; k < total_jd; k = k + WG) {
            let j_local = k / head_dim;
            let d       = k % head_dim;
            let jj      = j_start + j_local;
            if (jj >= causal_len) { continue; }

            let p_local  = sh_p[j_local];
            let dv_val   = p_local * sh_do[d];
            let dst_base = jj * kv_dim + kv_h * head_dim + d;

            if (is_finite(dv_val)) {
                var old_bits = atomicLoad(&dV[dst_base]);
                loop {
                    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + dv_val);
                    let res = atomicCompareExchangeWeak(&dV[dst_base], old_bits, new_bits);
                    if (res.exchanged) { break; }
                    old_bits = res.old_value;
                }
            }
        }
        workgroupBarrier();   // sh_p, sh_dp will be overwritten next iteration
    }

    let D = sh_D[0];

    // ════════════════════════════════════════════════════════════════════
    // PASS B: dQ accumulator + dK scatter (uses D)
    // ════════════════════════════════════════════════════════════════════
    for (var blk = 0u; blk < n_blocks; blk = blk + 1u) {
        let j_start = blk * WG;
        let j = j_start + tid;

        // Recompute P, dP (same dot products as Pass A)
        var P_local: f32  = 0.0;
        var dP_local: f32 = 0.0;
        if (j < causal_len) {
            let kv_base = j * kv_dim + kv_h * head_dim;
            var qk: f32  = 0.0;
            var dov: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                qk  = fma(sh_q[d],  K[kv_base + d], qk);
                dov = fma(sh_do[d], V[kv_base + d], dov);
            }
            P_local  = nan_guard(exp(qk * scale - lse_i));
            dP_local = dov;
        }
        sh_p[tid]  = P_local;
        sh_dp[tid] = dP_local;
        workgroupBarrier();   // sh_p, sh_dp visible to all threads

        // dQ accumulator: sh_dq_acc[d] += Σ_{j in block} dS[j]·K[j, d]
        // Use (d_local, k_chunk) parallelization (same as forward Phase 4).
        if (tid < n_active) {
            let kl_start   = kc * chunk_size;
            let kl_end_raw = kl_start + chunk_size;
            let kl_end     = select(kl_end_raw, WG, kl_end_raw > WG);

            var partial: f32 = 0.0;
            for (var kl = kl_start; kl < kl_end; kl = kl + 1u) {
                let kk = j_start + kl;
                if (kk < causal_len) {
                    let dS = scale * sh_p[kl] * (sh_dp[kl] - D);
                    partial = fma(dS, K[kk * kv_dim + kv_h * head_dim + d_local], partial);
                }
            }
            sh_partial[d_local * k_par + kc] = partial;
        }
        workgroupBarrier();

        if (kc == 0u && tid < n_active) {
            var sum_dq: f32 = sh_partial[d_local * k_par];
            for (var i = 1u; i < k_par; i = i + 1u) {
                sum_dq = sum_dq + sh_partial[d_local * k_par + i];
            }
            sh_dq_acc[d_local] = sh_dq_acc[d_local] + sum_dq;
        }
        workgroupBarrier();

        // dK scatter: dK[j,d] += dS[j]·Q[d]
        let total_jd = WG * head_dim;
        for (var k = tid; k < total_jd; k = k + WG) {
            let j_local = k / head_dim;
            let d       = k % head_dim;
            let jj      = j_start + j_local;
            if (jj >= causal_len) { continue; }

            let p_local  = sh_p[j_local];
            let dp_local = sh_dp[j_local];
            let dS       = scale * p_local * (dp_local - D);
            let dk_val   = dS * sh_q[d];
            let dst_base = jj * kv_dim + kv_h * head_dim + d;

            if (is_finite(dk_val)) {
                var old_bits = atomicLoad(&dK[dst_base]);
                loop {
                    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + dk_val);
                    let res = atomicCompareExchangeWeak(&dK[dst_base], old_bits, new_bits);
                    if (res.exchanged) { break; }
                    old_bits = res.old_value;
                }
            }
        }
        workgroupBarrier();
    }

    // ── Final: write dQ[s,h,d] += dq_acc[d] (workgroup-owned, no atomic) ──
    if (tid < head_dim) {
        let dq_val = sh_dq_acc[tid];
        let cur = dQ[s * d_model + h * head_dim + tid];
        let nxt = cur + select(0.0, dq_val, dq_val == dq_val);   // NaN guard
        dQ[s * d_model + h * head_dim + tid] = nxt;
    }
}


// --- KERNEL: attention_backward_short ---
// Single-pass causal attention backward for seq_len ≤ MAX_SEQ (1024).
// Owns dQ; emits P and dS to global scratch for the dKdV companion kernel
// to consume. Eliminates the previous Phase-5 atomic CAS scatter (which
// was contended ~1024-way per dK/dV address under GQA, dominating step
// time) by inverting the loop nest in a separate key-major kernel.
//
// Phases (one workgroup per (s, h), 256 threads):
//   0. cache Q[s,h,*] → sh_q,  dO[s,h,*] → sh_do
//   1+2 (fused). P[j] = exp(scale · Q·K[j] − LSE),  dP[j] = Σ_d dO[d]·V[j,d]
//   3. D       = Σ_j P[j] · dP[j]                      (workgroup reduce)
//   3.5 flush per-(s,h,j) P and dS to global for the dKdV kernel.
//        scratch_idx = s · n_heads · seq_len + h · seq_len + j
//        P_out[idx]  = P[j],  dS_out[idx] = scale · P[j] · (dP[j] − D)
//   4. dQ[d]   += Σ_j dS[j] · K[j,d]                   (no atomic — owned)
//
// dK/dV are produced by `attention_backward_short_dKdV` (key-major), which
// reads the P/dS scratch this kernel writes, then does owned writes. No
// atomics anywhere on the short path.
@group(0) @binding(0)  var<storage, read>       Q:      array<f32>;
@group(0) @binding(1)  var<storage, read>       K:      array<f32>;
@group(0) @binding(2)  var<storage, read>       V:      array<f32>;
@group(0) @binding(3)  var<storage, read>       dO:     array<f32>;
@group(0) @binding(4)  var<storage, read>       LSE:    array<f32>;
@group(0) @binding(5)  var<storage, read_write> dQ:     array<f32>;
@group(0) @binding(6)  var<storage, read_write> P_out:  array<f32>;
@group(0) @binding(7)  var<storage, read_write> dS_out: array<f32>;
@group(0) @binding(8)  var<uniform> dims:   vec4<u32>;
@group(0) @binding(9)  var<uniform> params: vec4<f32>;

@compute @workgroup_size(256, 1, 1)
fn attention_backward_short(@builtin(workgroup_id) wgid: vec3<u32>,
                            @builtin(local_invocation_index) tid: u32) {
    let s = wgid.x;
    let h = wgid.y;

    let seq_len    = dims.x;
    let n_heads    = dims.y;
    let n_kv_heads = dims.z;
    let head_dim   = dims.w;
    let scale      = params.x;

    if (s >= seq_len || h >= n_heads) { return; }

    let q_per_group = n_heads / n_kv_heads;
    let kv_h = h / q_per_group;

    let d_model = n_heads * head_dim;
    let kv_dim  = n_kv_heads * head_dim;

    let q_row_base = s * d_model + h * head_dim;
    let causal_len = s + 1u;
    let lse_i = LSE[s * n_heads + h];

    // ── Phase 0: cache Q[s,h,*], dO[s,h,*] ──
    for (var d = tid; d < head_dim; d = d + WG) {
        sh_q[d]  = Q[q_row_base + d];
        sh_do[d] = dO[q_row_base + d];
    }
    workgroupBarrier();

    // ── Phase 1+2 (fused): P[j] = exp(scale · Q·K[j] − LSE),  dP[j] = Σ_d dO[d]·V[j,d] ──
    // K[j,*] and V[j,*] share kv_base; one j-loop avoids a redundant index walk
    // and one workgroupBarrier (the prior split had two phases each ending in a
    // barrier; sh_p_full / sh_dp_full are written by distinct j strides so a
    // single barrier suffices before Phase 3 reads them).
    for (var j = tid; j < causal_len; j = j + WG) {
        if (j >= MAX_SEQ) { break; }
        let kv_base = j * kv_dim + kv_h * head_dim;
        var qk: f32  = 0.0;
        var dov: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            qk  = fma(sh_q[d],  K[kv_base + d], qk);
            dov = fma(sh_do[d], V[kv_base + d], dov);
        }
        sh_p_full[j]  = nan_guard(exp(qk * scale - lse_i));
        sh_dp_full[j] = dov;
    }
    workgroupBarrier();

    // ── Phase 3: D = Σ_j P[j] · dP[j] ──
    var local_d: f32 = 0.0;
    for (var j = tid; j < causal_len; j = j + WG) {
        if (j >= MAX_SEQ) { break; }
        local_d = local_d + sh_p_full[j] * sh_dp_full[j];
    }
    let D = wg_reduce_sum(tid, local_d);

    // ── Phase 3.5: flush per-(s,h,j) P and dS to global scratch ──
    // Consumed by attention_backward_short_dKdV. Layout matches LSE indexing
    // extended with a j axis: idx = s · n_heads · seq_len + h · seq_len + j.
    // Only writes valid (j ≤ s) entries; the dKdV kernel reads only that
    // causal-valid range so stale data above the diagonal is never read.
    let scratch_row = s * (n_heads * seq_len) + h * seq_len;
    for (var j = tid; j < causal_len; j = j + WG) {
        if (j >= MAX_SEQ) { break; }
        let p_val  = sh_p_full[j];
        let dS_val = scale * p_val * (sh_dp_full[j] - D);
        P_out[scratch_row + j]  = p_val;
        dS_out[scratch_row + j] = dS_val;
    }

    // ── Phase 4: dQ[d] += Σ_j dS[j] · K[j,d]  ((d_local, k_chunk) parallel) ──
    {
        let k_par      = max(1u, WG / max(1u, head_dim));
        let n_active   = head_dim * k_par;
        let d_local    = tid / k_par;
        let kc         = tid % k_par;

        if (tid < n_active) {
            let chunk_size = (causal_len + k_par - 1u) / k_par;
            let j_start    = kc * chunk_size;
            let j_end_raw  = j_start + chunk_size;
            let j_end      = select(j_end_raw, causal_len, j_end_raw > causal_len);

            var partial: f32 = 0.0;
            for (var j = j_start; j < j_end; j = j + 1u) {
                if (j >= MAX_SEQ) { break; }
                let dS = scale * sh_p_full[j] * (sh_dp_full[j] - D);
                partial = fma(dS, K[j * kv_dim + kv_h * head_dim + d_local], partial);
            }
            sh_partial[d_local * k_par + kc] = partial;
        }
        workgroupBarrier();

        if (kc == 0u && tid < n_active) {
            var acc: f32 = sh_partial[d_local * k_par];
            for (var i = 1u; i < k_par; i = i + 1u) {
                acc = acc + sh_partial[d_local * k_par + i];
            }
            let cur = dQ[s * d_model + h * head_dim + d_local];
            let nxt = cur + select(0.0, acc, acc == acc);
            dQ[s * d_model + h * head_dim + d_local] = nxt;
        }
    }

    // dK and dV are produced by attention_backward_short_dKdV (no atomics).
}


// --- KERNEL: attention_backward_short_dKdV ---
// Key-major companion to attention_backward_short. One workgroup per
// (j, kv_h); each WG owns dK[j, kv_h, *] and dV[j, kv_h, *] so writes are
// non-atomic. Replaces the previous Phase-5 CAS scatter, which was
// contended ~ (seq − j) · q_per_group ways per (j, kv_h, d) address — the
// dominant cost of the old kernel under GQA.
//
// Reads precomputed P[s,h_q,j] and dS[s,h_q,j] = scale · P · (dP − D[s,h_q])
// from scratch buffers populated by attention_backward_short, so this
// kernel skips Q·K and dO·V reductions entirely. Per iteration each thread
// performs two FMAs and four scalar loads — bandwidth-bound on Q/dO.
//
// Parallelization (256 threads):
//   d_local = tid / k_par,  chunk = tid % k_par
//   k_par   = max(1, WG / head_dim)            // 4 when head_dim=64
//   Each (d_local, chunk) thread accumulates dK[j,kv_h,d_local] and
//   dV[j,kv_h,d_local] over its slice of (s, h_q) iterations, then a
//   k_par-way reduction across `chunk` writes the owned final value.
@group(0) @binding(0) var<storage, read>       Q:    array<f32>;
@group(0) @binding(1) var<storage, read>       dO:   array<f32>;
@group(0) @binding(2) var<storage, read>       P_in: array<f32>;
@group(0) @binding(3) var<storage, read>       dS_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> dK:   array<f32>;
@group(0) @binding(5) var<storage, read_write> dV:   array<f32>;
@group(0) @binding(6) var<uniform> dims:   vec4<u32>;  // (seq_len, n_heads, n_kv_heads, head_dim)

@compute @workgroup_size(256, 1, 1)
fn attention_backward_short_dKdV(@builtin(workgroup_id) wgid: vec3<u32>,
                                 @builtin(local_invocation_index) tid: u32) {
    let j      = wgid.x;
    let kv_h   = wgid.y;

    let seq_len    = dims.x;
    let n_heads    = dims.y;
    let n_kv_heads = dims.z;
    let head_dim   = dims.w;

    if (j >= seq_len || kv_h >= n_kv_heads) { return; }

    let q_per_group = n_heads / n_kv_heads;
    let d_model     = n_heads    * head_dim;
    let kv_dim      = n_kv_heads * head_dim;

    let k_par    = max(1u, WG / max(1u, head_dim));
    let n_active = head_dim * k_par;
    let d_local  = tid / k_par;
    let chunk    = tid % k_par;

    // Iter space: (s, h_q) with s ∈ [j, seq_len), h_q ∈ q_group(kv_h).
    // Layout: it = s_off · q_per_group + h_q_off so consecutive iterations
    // share s — better L1 reuse on Q[s,*,d] / dO[s,*,d] across small h_q
    // strides (head_dim) than the inverted ordering (d_model strides).
    let iters_per_s = q_per_group;
    let s_count     = seq_len - j;
    let iters_total = s_count * iters_per_s;

    var dk_partial: f32 = 0.0;
    var dv_partial: f32 = 0.0;

    if (tid < n_active) {
        let chunk_size = (iters_total + k_par - 1u) / k_par;
        let it_start   = chunk * chunk_size;
        let it_end_raw = it_start + chunk_size;
        let it_end     = select(it_end_raw, iters_total, it_end_raw > iters_total);

        for (var it = it_start; it < it_end; it = it + 1u) {
            let s_off    = it / iters_per_s;
            let h_q_off  = it % iters_per_s;
            let s        = j + s_off;
            let h_q      = kv_h * q_per_group + h_q_off;

            let scratch_idx = s * (n_heads * seq_len) + h_q * seq_len + j;
            let p_val  = P_in[scratch_idx];
            let dS_val = dS_in[scratch_idx];

            let qd_idx = s * d_model + h_q * head_dim + d_local;
            let q_val  = Q[qd_idx];
            let do_val = dO[qd_idx];

            dk_partial = fma(dS_val, q_val,  dk_partial);
            dv_partial = fma(p_val,  do_val, dv_partial);
        }
        sh_dk_partial[d_local * k_par + chunk] = dk_partial;
        sh_dv_partial[d_local * k_par + chunk] = dv_partial;
    }
    workgroupBarrier();

    // Reduce k_par chunk partials per d_local; write owned (single writer).
    if (chunk == 0u && tid < n_active) {
        var dk_sum: f32 = sh_dk_partial[d_local * k_par];
        var dv_sum: f32 = sh_dv_partial[d_local * k_par];
        for (var i = 1u; i < k_par; i = i + 1u) {
            dk_sum = dk_sum + sh_dk_partial[d_local * k_par + i];
            dv_sum = dv_sum + sh_dv_partial[d_local * k_par + i];
        }
        let dst = j * kv_dim + kv_h * head_dim + d_local;
        // NaN guard mirrors the old Phase-5 is_finite check; keeps a single
        // bad row from corrupting downstream RoPE/matmul kernels.
        dK[dst] = select(0.0, dk_sum, dk_sum == dk_sum);
        dV[dst] = select(0.0, dv_sum, dv_sum == dv_sum);
    }
}


// --- KERNEL: rope_q_backward ---
// Inverse rotation on dQ (transpose of forward rotation: orthogonal R, R^T = R^-1).
//   forward:   q0' =  q0*c - q1*s,   q1' =  q0*s + q1*c
//   backward:  dq0 =  dq0'*c + dq1'*s,  dq1 = -dq0'*s + dq1'*c
@group(0) @binding(0) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>;   // (seq_len, n_heads, head_dim, pos_offset)
@group(0) @binding(2) var<uniform> params: vec4<f32>; // (rope_base, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn rope_q_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);

    let seq_len = dims.x;
    let n_heads = dims.y;
    let head_dim = dims.z;
    let pos_offset = dims.w;
    let base = params.x;

    let half_d = head_dim / 2u;
    let pairs_per_pos = n_heads * half_d;
    let total = seq_len * pairs_per_pos;
    if (i >= total) { return; }

    let s = i / pairs_per_pos;
    let rest = i % pairs_per_pos;
    let h = rest / half_d;
    let p = rest % half_d;

    let freq = exp(-log(base) * f32(2u * p) / f32(head_dim));
    let angle = f32(s + pos_offset) * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let row_base = s * (n_heads * head_dim);
    let head_base = row_base + h * head_dim;
    let i0 = head_base + 2u * p;
    let i1 = i0 + 1u;
    let g0 = dQ[i0];
    let g1 = dQ[i1];
    dQ[i0] =  g0 * c + g1 * sn;
    dQ[i1] = -g0 * sn + g1 * c;
}


// --- KERNEL: rope_k_backward ---
@group(0) @binding(0) var<storage, read_write> dK: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>;   // (seq_len, n_kv_heads, head_dim, pos_offset)
@group(0) @binding(2) var<uniform> params: vec4<f32>; // (rope_base, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn rope_k_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);

    let seq_len = dims.x;
    let n_kv = dims.y;
    let head_dim = dims.z;
    let pos_offset = dims.w;
    let base = params.x;

    let half_d = head_dim / 2u;
    let pairs_per_pos = n_kv * half_d;
    let total = seq_len * pairs_per_pos;
    if (i >= total) { return; }

    let s = i / pairs_per_pos;
    let rest = i % pairs_per_pos;
    let h = rest / half_d;
    let p = rest % half_d;

    let freq = exp(-log(base) * f32(2u * p) / f32(head_dim));
    let angle = f32(s + pos_offset) * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let row_base = s * (n_kv * head_dim);
    let head_base = row_base + h * head_dim;
    let i0 = head_base + 2u * p;
    let i1 = i0 + 1u;
    let g0 = dK[i0];
    let g1 = dK[i1];
    dK[i0] =  g0 * c + g1 * sn;
    dK[i1] = -g0 * sn + g1 * c;
}
