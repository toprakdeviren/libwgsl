/**
 * rope.wgsl — Rotary Position Embedding, in-place on Q and K.
 *
 * Interleaved-pair convention: pairs are (q[2i], q[2i+1]).
 * For each (seq_pos s, head h, pair i) in head_dim/2 pairs per head:
 *   freq  = base^(-2i / head_dim)
 *   angle = (s + pos_offset) * freq
 *   q0' = q0 * cos - q1 * sin
 *   q1' = q0 * sin + q1 * cos
 *
 * Q layout: [seq_len, n_heads * head_dim]
 * K layout: [seq_len, n_kv_heads * head_dim]   (GQA: kv_dim ≤ d_model)
 */

fn rope_freq(pair_in_head: u32, head_dim: u32, base: f32) -> f32 {
    // exp(-log(base) * 2*i / head_dim)
    return exp(-log(base) * f32(2u * pair_in_head) / f32(head_dim));
}


// --- KERNEL: rope_q ---
// Apply RoPE to Q in-place. One thread per (seq_pos, q_pair).
@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>;   // (seq_len, n_heads, head_dim, pos_offset)
@group(0) @binding(2) var<uniform> params: vec4<f32>; // (rope_base, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn rope_q(@builtin(global_invocation_id) gid: vec3<u32>,
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

    let freq = rope_freq(p, head_dim, base);
    let angle = f32(s + pos_offset) * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let row_base = s * (n_heads * head_dim);
    let head_base = row_base + h * head_dim;
    let i0 = head_base + 2u * p;
    let i1 = i0 + 1u;
    let q0 = Q[i0];
    let q1 = Q[i1];
    Q[i0] = q0 * c - q1 * sn;
    Q[i1] = q0 * sn + q1 * c;
}


// --- KERNEL: rope_k ---
// Same as rope_q but on K, which uses n_kv_heads instead of n_heads.
@group(0) @binding(0) var<storage, read_write> K: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>;   // (seq_len, n_kv_heads, head_dim, pos_offset)
@group(0) @binding(2) var<uniform> params: vec4<f32>; // (rope_base, _, _, _)

@compute @workgroup_size(256, 1, 1)
fn rope_k(@builtin(global_invocation_id) gid: vec3<u32>,
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

    let freq = rope_freq(p, head_dim, base);
    let angle = f32(s + pos_offset) * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let row_base = s * (n_kv * head_dim);
    let head_base = row_base + h * head_dim;
    let i0 = head_base + 2u * p;
    let i1 = i0 + 1u;
    let k0 = K[i0];
    let k1 = K[i1];
    K[i0] = k0 * c - k1 * sn;
    K[i1] = k0 * sn + k1 * c;
}
