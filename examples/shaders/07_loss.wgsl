/**
 * loss.wgsl — fused softmax + cross-entropy + dLogits.
 *
 * Per row s in [0, seq_len):
 *   row_max  = max_v logits[s, v]
 *   denom    = sum_v exp(logits[s, v] - row_max)
 *   loss[s]  = -(logits[s, tgt[s]] - row_max - log(denom))
 *   dLogits[s, v] = (softmax(logits[s])[v] - (v==tgt[s] ? 1 : 0)) / norm
 *
 * In-place output: dLogits is written into the same buffer as logits.
 * Phases 1-3 read logits[row_base + v] only; Phase 4 reads logits[v] then
 * immediately writes dLogits[v] into the same offset (single thread
 * processes its v positions serially, so the read happens-before the
 * write of the same address — no race). Saves seq_len*vocab_size*4 bytes
 * of activation memory (32 MB at S=512, V=16384).
 *
 * If tgt[s] >= vocab_size we treat the row as ignored:
 *   loss[s] = 0; dLogits[s, *] = 0.
 *
 * Norm is the count of valid (non-ignored) rows, computed by JS and
 * passed in `params.x = 1/norm`.
 */

// --- KERNEL: cross_entropy ---
// Dispatch: workgroups = (seq_len, 1, 1). One workgroup per row.
//
// Combines three losses in one pass:
//   1. Cross-entropy: -log p[target]
//   2. Label smoothing (α): target dist becomes (1-α)·δ_t + α/V uniform.
//      Loss term: lse - (1-α)·l_t - α·mean(l)
//      Grad term: (p - (1-α)·one_hot - α/V) / norm
//   3. Z-loss (β): regularizes log Σexp(l) toward 0. Stabilizes training.
//      Loss term: β · lse²        (lse = log Σ exp(l))
//      Grad term: 2β · lse · p / norm
//
// All three are scaled by inv_norm = 1/valid_tokens. β=0 and α=0 reduce to
// plain cross-entropy.
// Single read_write binding for logits — Phase 4 writes dLogits in-place.
@group(0) @binding(0) var<storage, read_write> logits:  array<f32>;
@group(0) @binding(1) var<storage, read>       tgts: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses:  array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>;   // (seq_len, vocab_size, _, _)
@group(0) @binding(4) var<uniform> params: vec4<f32>; // (inv_norm, alpha, beta_zl, inv_V)

@compute @workgroup_size(256, 1, 1)
fn cross_entropy(@builtin(workgroup_id) wgid: vec3<u32>,
                 @builtin(local_invocation_index) tid: u32) {
    let s = wgid.x;
    let seq_len = dims.x;
    let vocab_size = dims.y;
    let inv_norm = params.x;
    let alpha    = params.y;
    let beta_zl  = params.z;
    let inv_V    = params.w;

    if (s >= seq_len) { return; }

    let row_base = s * vocab_size;
    let tgt = tgts[s];

    // Ignore if tgt out of range
    if (tgt >= vocab_size) {
        if (tid == 0u) { losses[s] = 0.0; }
        for (var v = tid; v < vocab_size; v = v + WG) {
            logits[row_base + v] = 0.0;
        }
        return;
    }

    // Phase 1: row max
    var local_max: f32 = NEG_INF;
    for (var v = tid; v < vocab_size; v = v + WG) {
        local_max = max(local_max, logits[row_base + v]);
    }
    let row_max = wg_reduce_max(tid, local_max);

    // Phase 2a: denom = Σ exp(l - row_max)
    var local_sum: f32 = 0.0;
    for (var v = tid; v < vocab_size; v = v + WG) {
        local_sum = local_sum + exp(logits[row_base + v] - row_max);
    }
    let denom = wg_reduce_sum(tid, local_sum);

    // Phase 2b: sum of logits — only needed when label smoothing is on.
    // Skipping when alpha=0 saves a vocab_size-wide loop + reduction per row,
    // which adds up at large vocabs (50k+) where the loss kernel becomes
    // bandwidth-bound. The conditional is uniform (alpha is a uniform), so
    // the workgroupBarrier inside the branch is safe — all threads agree.
    var sum_l: f32 = 0.0;
    if (alpha > 0.0) {
        workgroupBarrier();
        var local_l: f32 = 0.0;
        for (var v = tid; v < vocab_size; v = v + WG) {
            local_l = local_l + logits[row_base + v];
        }
        sum_l = wg_reduce_sum(tid, local_l);
    }

    let inv_denom = 1.0 / max(denom, 1e-30);
    let lse       = row_max + log(max(denom, 1e-30));
    let mean_l    = sum_l * inv_V;

    // Phase 3: loss = lse - (1-α)·l_t - α·mean_l + β·lse²
    if (tid == 0u) {
        let l_t  = logits[row_base + tgt];
        let ce   = lse - (1.0 - alpha) * l_t - alpha * mean_l;
        let zl   = beta_zl * lse * lse;
        losses[s] = ce + zl;
    }

    // Phase 4: dLogits = ((p - tgt_dist) + 2β·lse·p) / norm
    //          tgt_dist = (1-α)·one_hot + α/V
    // In-place write: each thread reads logits[v] then writes dLogits[v] into
    // the same offset (read-before-write per-thread, no cross-thread overlap
    // because thread t owns positions {t, t+WG, t+2*WG, ...}).
    let zl_grad_factor = 2.0 * beta_zl * lse;
    for (var v = tid; v < vocab_size; v = v + WG) {
        let p = exp(logits[row_base + v] - row_max) * inv_denom;
        let one_hot = select(0.0, 1.0, v == tgt);
        let tgt_dist = (1.0 - alpha) * one_hot + alpha * inv_V;
        let gr       = (p - tgt_dist) + zl_grad_factor * p;
        logits[row_base + v] = gr * inv_norm;
    }
}


// --- KERNEL: sum_losses ---
// Reduce per-row losses → single scalar (sum). Reads losses[seq_len] → out_sum[0].
// Dispatch: 1 workgroup of 256.
@group(0) @binding(0) var<storage, read>       losses:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_sum: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256, 1, 1)
fn sum_losses(@builtin(local_invocation_index) tid: u32) {
    var local_sum: f32 = 0.0;
    for (var i = tid; i < n; i = i + WG) {
        local_sum = local_sum + losses[i];
    }
    let total = wg_reduce_sum(tid, local_sum);
    if (tid == 0u) { out_sum[0] = total; }
}
