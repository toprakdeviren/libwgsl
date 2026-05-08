/**
 * embedding.wgsl — token embedding lookup + scatter-backward.
 *
 * Forward:  out[s, d] = embed_table[tokens[s], d]
 * Backward: grad_table[t, d] += sum over s where tokens[s]==t of grad_out[s, d]
 *
 * The backward uses CAS atomic-add on bit-cast f32 (WebGPU has no native
 * f32 atomicAdd). Contention is low for typical vocab/seq combos.
 */

// CAS-based f32 atomic add; inlined where used because WGSL forbids
// ptr-to-storage parameters in portable mode.


// --- KERNEL: embed_lookup ---
// Forward: one thread per (s, d) element of the output.
// out: [seq_len, d_model]   table: [vocab_size, d_model]   tokens: [seq_len]
@group(0) @binding(0) var<storage, read>       tokens: array<u32>;
@group(0) @binding(1) var<storage, read>       table:  array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // (seq_len, d_model, vocab_size, _)

@compute @workgroup_size(256, 1, 1)
fn embed_lookup(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    let seq_len = dims.x;
    let d_model = dims.y;
    let vocab_size = dims.z;
    let total = seq_len * d_model;
    if (i >= total) { return; }

    let s = i / d_model;
    let d = i % d_model;
    let t = tokens[s];
    if (t >= vocab_size) {
        out[i] = 0.0;
        return;
    }
    out[i] = table[t * d_model + d];
}


// --- KERNEL: embed_backward ---
// Backward: scatter grad_out into grad_table at row tokens[s].
// One thread per (s, d) element of grad_out; CAS-add into grad_table[t, d].
@group(0) @binding(0) var<storage, read>                tokens:     array<u32>;
@group(0) @binding(1) var<storage, read>                grad_out:   array<f32>;
@group(0) @binding(2) var<storage, read_write>          grad_table: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // (seq_len, d_model, vocab_size, _)

@compute @workgroup_size(256, 1, 1)
fn embed_backward(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    let seq_len = dims.x;
    let d_model = dims.y;
    let vocab_size = dims.z;
    let total = seq_len * d_model;
    if (i >= total) { return; }

    let s = i / d_model;
    let d = i % d_model;
    let t = tokens[s];
    if (t >= vocab_size) { return; }

    let val = grad_out[i];
    if (val == 0.0 || !is_finite(val)) { return; }
    let dst_idx = t * d_model + d;
    var old_bits = atomicLoad(&grad_table[dst_idx]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let res = atomicCompareExchangeWeak(&grad_table[dst_idx], old_bits, new_bits);
        if (res.exchanged) { break; }
        old_bits = res.old_value;
    }
}


// ════════════════════════════════════════════════════════════
// Mixed-precision (f16 table) variant
// ════════════════════════════════════════════════════════════
// --- KERNEL: embed_lookup_w16 ---
@group(0) @binding(0) var<storage, read>       tokens_w16: array<u32>;
@group(0) @binding(1) var<storage, read>       table_w16:  array<f16>;
@group(0) @binding(2) var<storage, read_write> out_w16:    array<f32>;
@group(0) @binding(3) var<uniform> dims_w16: vec4<u32>; // (seq_len, d_model, vocab_size, _)

@compute @workgroup_size(256, 1, 1)
fn embed_lookup_w16(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_id(gid, nwg);
    let seq_len = dims_w16.x;
    let d_model = dims_w16.y;
    let vocab_size = dims_w16.z;
    let total = seq_len * d_model;
    if (i >= total) { return; }

    let s = i / d_model;
    let d = i % d_model;
    let t = tokens_w16[s];
    if (t >= vocab_size) {
        out_w16[i] = 0.0;
        return;
    }
    out_w16[i] = f32(table_w16[t * d_model + d]);
}
