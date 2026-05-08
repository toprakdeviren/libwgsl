/**
 * attention.wgsl — multi-head causal attention with online softmax + GQA.
 *
 * Single-pass (online) softmax: maintains running max m and sum-of-exp l
 * while iterating K in blocks of WG=256. No O(seq) score scratch — workgroup
 * memory is O(WG + head_dim) regardless of sequence length.
 *
 * For each (head h, query pos s): one workgroup, 256 threads, blocks of 256
 * keys at a time. Each block:
 *   1. Each thread computes 1 score = Q · K[k] * scale  (NEG_INF if k>=causal_len)
 *   2. m_new = max(running m, block_max)
 *   3. P[k] = exp(score - m_new), block-wide sum → l_partial
 *   4. correction = exp(m_old - m_new); l = l_old·correction + l_partial
 *   5. acc[d] = acc[d]·correction + Σ_k P[k]·V[k,d]
 *   6. m = m_new, l = l_new
 * After all blocks: O[d] = acc[d] / l, save LSE = m + log(l).
 *
 * Layout:
 *   Q: [seq_len, n_heads     * head_dim]
 *   K: [seq_len, n_kv_heads  * head_dim]
 *   V: [seq_len, n_kv_heads  * head_dim]
 *   O: [seq_len, n_heads     * head_dim]
 *   LSE: [seq_len, n_heads]   (saved log-sum-exp for backward)
 */

const MAX_HEAD_DIM: u32 = 256u;   // upper bound on head_dim (sh_q sizing)

// Workgroup memory:
//   sh_q       — Q row cache (head_dim entries used)
//   sh_p       — current K-block scores then P values (WG entries)
//   sh_acc     — running output accumulator (head_dim entries used)
//   sh_partial — V·P partial sum reduction scratch (head_dim × k_par)
//   sh_max     — running max m
//   sh_lse     — running sum-exp l
var<workgroup> sh_q:       array<f32, 256>;
var<workgroup> sh_p:       array<f32, 256>;
var<workgroup> sh_acc:     array<f32, 256>;
var<workgroup> sh_partial: array<f32, 256>;
var<workgroup> sh_max:     array<f32, 1>;
var<workgroup> sh_lse:     array<f32, 1>;


// --- KERNEL: attention_forward ---
// Dispatch: workgroups = (seq_len, n_heads, 1)
@group(0) @binding(0) var<storage, read>       Q:   array<f32>;
@group(0) @binding(1) var<storage, read>       K:   array<f32>;
@group(0) @binding(2) var<storage, read>       V:   array<f32>;
@group(0) @binding(3) var<storage, read_write> O:   array<f32>;
@group(0) @binding(4) var<storage, read_write> LSE: array<f32>;
@group(0) @binding(5) var<uniform> dims: vec4<u32>;   // (seq_len, n_heads, n_kv_heads, head_dim)
@group(0) @binding(6) var<uniform> params: vec4<f32>; // (scale, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn attention_forward(@builtin(workgroup_id) wgid: vec3<u32>,
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

    // Phase 0: cache Q row, init online state
    for (var i = tid; i < head_dim; i = i + WG) {
        sh_q[i] = Q[q_row_base + i];
    }
    if (tid < head_dim) { sh_acc[tid] = 0.0; }
    if (tid == 0u) {
        sh_max[0] = NEG_INF;
        sh_lse[0] = 0.0;
    }
    workgroupBarrier();

    // K-block parallelization parameters (constant across blocks)
    let k_par      = max(1u, WG / max(1u, head_dim));    // 4 when head_dim=64
    let n_active     = head_dim * k_par;
    let d_local    = tid / k_par;
    let kc         = tid % k_par;
    let chunk_size = (WG + k_par - 1u) / k_par;          // 64 when k_par=4

    let n_blocks = (causal_len + WG - 1u) / WG;
    for (var blk = 0u; blk < n_blocks; blk = blk + 1u) {
        let k_start = blk * WG;
        let k = k_start + tid;

        // ── Phase A: each thread computes one score (or NEG_INF if past causal) ──
        var score: f32 = NEG_INF;
        if (k < causal_len) {
            let k_row_base = k * kv_dim + kv_h * head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot = fma(sh_q[d], K[k_row_base + d], dot);
            }
            score = dot * scale;
        }

        // ── Phase B: combine block-wide max with running max ──
        let block_max = wg_reduce_max(tid, score);
        let m_old = sh_max[0];
        let m_new = max(m_old, block_max);

        // ── Phase C: P[k] = exp(score - m_new), block-wide sum, update l ──
        var p_val: f32 = 0.0;
        if (k < causal_len) {
            p_val = exp(score - m_new);
        }
        sh_p[tid] = p_val;
        let block_sum = wg_reduce_sum(tid, p_val);
        // m_old = NEG_INF first iter ⇒ exp underflows to 0 (drops stale state).
        let correction = exp(m_old - m_new);
        let l_old = sh_lse[0];
        let l_new = l_old * correction + block_sum;

        // ── Phase D: update sh_acc with V·P contribution + correction ──
        // Per (d_local, kc): partial = Σ_kl_in_chunk sh_p[kl] · V[k_start+kl, d_local].
        // Reduce partials per d, apply correction to old acc, write back.
        if (tid < n_active) {
            let kl_start   = kc * chunk_size;
            let kl_end_raw = kl_start + chunk_size;
            let kl_end     = select(kl_end_raw, WG, kl_end_raw > WG);

            var partial: f32 = 0.0;
            for (var kl = kl_start; kl < kl_end; kl = kl + 1u) {
                let kk = k_start + kl;
                if (kk < causal_len) {
                    partial = fma(sh_p[kl], V[kk * kv_dim + kv_h * head_dim + d_local], partial);
                }
            }
            sh_partial[d_local * k_par + kc] = partial;
        }
        workgroupBarrier();

        if (kc == 0u && tid < n_active) {
            var sum_pv: f32 = sh_partial[d_local * k_par];
            for (var i = 1u; i < k_par; i = i + 1u) {
                sum_pv = sum_pv + sh_partial[d_local * k_par + i];
            }
            sh_acc[d_local] = sh_acc[d_local] * correction + sum_pv;
        }

        if (tid == 0u) {
            sh_max[0] = m_new;
            sh_lse[0] = l_new;
        }
        workgroupBarrier();
    }

    // Final: O[d] = acc[d] / l, save LSE = m + log(l)
    let l_final = max(sh_lse[0], 1e-30);
    let inv_l = 1.0 / l_final;
    if (tid < head_dim) {
        O[s * d_model + h * head_dim + tid] = nan_guard(sh_acc[tid] * inv_l);
    }
    if (tid == 0u) {
        LSE[s * n_heads + h] = sh_max[0] + log(l_final);
    }
}


// --- KERNEL: attention_decode ---
// Single-query attention against a KV cache. Used during generation:
// for each new token, Q has one row [n_heads, head_dim] and K/V hold the
// past `cache_len` rows. No causal mask check inside — the cache is
// authoritative (everything in [0..cache_len) is a valid past token).
//
// Same online softmax structure as attention_forward, no LSE save.
//
// Layout:
//   Q:  [n_heads     * head_dim]              (single row)
//   K:  [seq_len, n_kv_heads * head_dim]      (cache; only [0..cache_len) read)
//   V:  [seq_len, n_kv_heads * head_dim]
//   O:  [n_heads     * head_dim]              (single row)
//
// Dispatch: workgroups = (n_heads, 1, 1)
@group(0) @binding(0) var<storage, read>       Q_one:   array<f32>;
@group(0) @binding(1) var<storage, read>       K_cache: array<f32>;
@group(0) @binding(2) var<storage, read>       V_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> O_one:   array<f32>;
@group(0) @binding(4) var<uniform> dims_dec:   vec4<u32>;   // (cache_len, n_heads, n_kv_heads, head_dim)
@group(0) @binding(5) var<uniform> params_dec: vec4<f32>;   // (scale, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn attention_decode(@builtin(workgroup_id) wgid: vec3<u32>,
                    @builtin(local_invocation_index) tid: u32) {
    let h = wgid.x;

    let cache_len = dims_dec.x;
    let n_heads   = dims_dec.y;
    let n_kv      = dims_dec.z;
    let head_dim  = dims_dec.w;
    let scale     = params_dec.x;

    if (h >= n_heads) { return; }

    let q_per_group = n_heads / n_kv;
    let kv_h = h / q_per_group;
    let kv_dim = n_kv * head_dim;
    let q_base = h * head_dim;

    // Phase 0: cache Q, init online state
    for (var i = tid; i < head_dim; i = i + WG) {
        sh_q[i] = Q_one[q_base + i];
    }
    if (tid < head_dim) { sh_acc[tid] = 0.0; }
    if (tid == 0u) {
        sh_max[0] = NEG_INF;
        sh_lse[0] = 0.0;
    }
    workgroupBarrier();

    let k_par      = max(1u, WG / max(1u, head_dim));
    let n_active     = head_dim * k_par;
    let d_local    = tid / k_par;
    let kc         = tid % k_par;
    let chunk_size = (WG + k_par - 1u) / k_par;

    let n_blocks = (cache_len + WG - 1u) / WG;
    for (var blk = 0u; blk < n_blocks; blk = blk + 1u) {
        let k_start = blk * WG;
        let k = k_start + tid;

        var score: f32 = NEG_INF;
        if (k < cache_len) {
            let k_base = k * kv_dim + kv_h * head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot = fma(sh_q[d], K_cache[k_base + d], dot);
            }
            score = dot * scale;
        }

        let block_max = wg_reduce_max(tid, score);
        let m_old = sh_max[0];
        let m_new = max(m_old, block_max);

        var p_val: f32 = 0.0;
        if (k < cache_len) {
            p_val = exp(score - m_new);
        }
        sh_p[tid] = p_val;
        let block_sum = wg_reduce_sum(tid, p_val);
        let correction = exp(m_old - m_new);
        let l_old = sh_lse[0];
        let l_new = l_old * correction + block_sum;

        if (tid < n_active) {
            let kl_start   = kc * chunk_size;
            let kl_end_raw = kl_start + chunk_size;
            let kl_end     = select(kl_end_raw, WG, kl_end_raw > WG);

            var partial: f32 = 0.0;
            for (var kl = kl_start; kl < kl_end; kl = kl + 1u) {
                let kk = k_start + kl;
                if (kk < cache_len) {
                    partial = fma(sh_p[kl], V_cache[kk * kv_dim + kv_h * head_dim + d_local], partial);
                }
            }
            sh_partial[d_local * k_par + kc] = partial;
        }
        workgroupBarrier();

        if (kc == 0u && tid < n_active) {
            var sum_pv: f32 = sh_partial[d_local * k_par];
            for (var i = 1u; i < k_par; i = i + 1u) {
                sum_pv = sum_pv + sh_partial[d_local * k_par + i];
            }
            sh_acc[d_local] = sh_acc[d_local] * correction + sum_pv;
        }

        if (tid == 0u) {
            sh_max[0] = m_new;
            sh_lse[0] = l_new;
        }
        workgroupBarrier();
    }

    let l_final = max(sh_lse[0], 1e-30);
    let inv_l = 1.0 / l_final;
    if (tid < head_dim) {
        O_one[h * head_dim + tid] = nan_guard(sh_acc[tid] * inv_l);
    }
}
