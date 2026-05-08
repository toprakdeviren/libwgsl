/**
 * linear.wgsl — tiled matrix multiply (64×64 output tile, 4×4 sub-tile per thread).
 *
 *   matmul:           Y = X @ W                X is [M, K], W is [K, N], Y is [M, N]
 *   matmul_residual:  Y = X @ W + R            (forward fusion: residual added in epilogue)
 *   matmul_t:         Y = X @ W^T              W is [N, K] (rows are the N output features)
 *   matmul_at:        Y[K, N] = X^T @ B        X is [M, K], B is [M, N]   (for dW)
 *   matmul_at_acc:    Y[K, N] += X^T @ B       (accumulating variant for grad accum)
 *   matmul_t_acc:     Y[M, N] += X @ W^T       (accumulating fused matmul_t for QKV/Wk/Wv)
 *
 * Tile structure:
 *   - workgroup_size = (16, 16, 1) → 256 threads
 *   - Each workgroup writes a 64×64 output tile (TM × TN); 16× more output
 *     than the previous 32×32 tile, so 4× fewer dispatches per matmul.
 *   - Each thread holds a 4×4 register accumulator (16 elements; was 2×2).
 *   - K is tiled at TK=16; loads cooperatively bring 64×16 and 16×64 tiles
 *     into workgroup memory (~8.5 KB vs ~4 KB before).
 *
 * Why this beats 32×32 / 2×2:
 *   - Each shared-mem value reused 4× per thread (vs 2×) → ½ the shared-mem
 *     traffic per multiply-accumulate, 2× arithmetic intensity in the inner loop.
 *   - 16 register-resident accumulators per thread vs 4 → bigger inner-loop
 *     hides shared-mem load latency better.
 *   - ¼ the workgroups → less per-WG launch overhead, better SM occupancy
 *     amortization.
 *
 * Coalesced reads on transposed paths: matmul_at swaps the aI mapping so
 * consecutive threads load consecutive K values at fixed M (instead of
 * stride-K jumps); matmul_t / matmul_t_acc do the same for W^T. Without
 * this, the tile load on transposed inputs hits ~32 cache lines per warp;
 * with it, just 2-4. (Apple GPU's L1 happens to absorb the older stride-K
 * pattern fairly well, so the realised gain from coalescing alone is small;
 * the bigger tile is what moves the needle.)
 *
 * fp32 throughout. Mixed-precision (f16 weight storage) variants live in
 * linear_w16.wgsl and mirror the same structure.
 */

const TM: u32 = 64u;       // output rows per workgroup
const TN: u32 = 64u;       // output cols per workgroup
const TK: u32 = 16u;       // K-dim tile width
// tileA padding: stride=16 floats lands threads with different ty on the
// same 32-bank bank in the inner loop. Stride=17 is coprime with 32 so
// no bank conflicts inside the warp. tileB has no such pattern (different
// threads → different N columns) and stays unpadded.
const TK_PAD: u32 = 17u;

var<workgroup> tileA: array<f32, 1088>;  // 64 × 17 (padded for bank conflicts)
var<workgroup> tileB: array<f32, 1024>;  // 16 × 64

// --- KERNEL: matmul ---
// Y[M, N] = X[M, K] @ W[K, N]
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // (M, N, K, _)

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(workgroup_id) wgid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims.x; let N = dims.y; let K = dims.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        // A tile [TM × TK]: each thread loads 4 consecutive K elements of one M row.
        // Threads 0..3 cover (m=0, k=0..15) → 16 contiguous floats = 1 cache line.
        let aI0 = tid * 4u;
        let aIm = aI0 / TK;       // m_local ∈ [0, 64)
        let aIk = aI0 % TK;       // k_local ∈ [0, 16); always a multiple of 4
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let kk = aIk + i;
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + kk] = select(0.0, X[axr * K + axc + i], okay);
        }

        // B tile [TK × TN]: each thread loads 4 consecutive N elements of one K row.
        // Threads 0..15 cover (k=0, n=0..63) → 64 contiguous floats = 4 cache lines.
        let bI0 = tid * 4u;
        let bIk = bI0 / TN;       // k_local ∈ [0, 16)
        let bIn = bI0 % TN;       // n_local ∈ [0, 64); always a multiple of 4
        let bwr = kBase + bIk; let bwc = block_col + bIn;
        let bwr_in = bwr < K;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let nn = bIn + i;
            let okay = bwr_in && (bwc + i) < N;
            tileB[bIk * TN + nn] = select(0.0, W[bwr * N + bwc + i], okay);
        }

        workgroupBarrier();

        // 4×4 sub-tile per thread. Thread (tx, ty) handles output rows
        // [4*ty, 4*ty+3] and cols [4*tx, 4*tx+3]. Inner loop reuses each
        // shared-mem load 4× (4 a-values × 4 b-values → 16 FMAs per k iter).
        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    // Row 0
    if (or + 0u < M && oc + 0u < N) { Y[(or + 0u) * N + (oc + 0u)] = acc00; }
    if (or + 0u < M && oc + 1u < N) { Y[(or + 0u) * N + (oc + 1u)] = acc01; }
    if (or + 0u < M && oc + 2u < N) { Y[(or + 0u) * N + (oc + 2u)] = acc02; }
    if (or + 0u < M && oc + 3u < N) { Y[(or + 0u) * N + (oc + 3u)] = acc03; }
    // Row 1
    if (or + 1u < M && oc + 0u < N) { Y[(or + 1u) * N + (oc + 0u)] = acc10; }
    if (or + 1u < M && oc + 1u < N) { Y[(or + 1u) * N + (oc + 1u)] = acc11; }
    if (or + 1u < M && oc + 2u < N) { Y[(or + 1u) * N + (oc + 2u)] = acc12; }
    if (or + 1u < M && oc + 3u < N) { Y[(or + 1u) * N + (oc + 3u)] = acc13; }
    // Row 2
    if (or + 2u < M && oc + 0u < N) { Y[(or + 2u) * N + (oc + 0u)] = acc20; }
    if (or + 2u < M && oc + 1u < N) { Y[(or + 2u) * N + (oc + 1u)] = acc21; }
    if (or + 2u < M && oc + 2u < N) { Y[(or + 2u) * N + (oc + 2u)] = acc22; }
    if (or + 2u < M && oc + 3u < N) { Y[(or + 2u) * N + (oc + 3u)] = acc23; }
    // Row 3
    if (or + 3u < M && oc + 0u < N) { Y[(or + 3u) * N + (oc + 0u)] = acc30; }
    if (or + 3u < M && oc + 1u < N) { Y[(or + 3u) * N + (oc + 1u)] = acc31; }
    if (or + 3u < M && oc + 2u < N) { Y[(or + 3u) * N + (oc + 2u)] = acc32; }
    if (or + 3u < M && oc + 3u < N) { Y[(or + 3u) * N + (oc + 3u)] = acc33; }
}


// --- KERNEL: matmul_residual ---
// Y[M, N] = X[M, K] @ W[K, N] + R[M, N]
@group(0) @binding(0) var<storage, read>       X_mr: array<f32>;
@group(0) @binding(1) var<storage, read>       W_mr: array<f32>;
@group(0) @binding(2) var<storage, read>       R_mr: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y_mr: array<f32>;
@group(0) @binding(4) var<uniform> dims_mr: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_residual(@builtin(workgroup_id) wgid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_mr.x; let N = dims_mr.y; let K = dims_mr.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_mr[axr * K + axc + i], okay);
        }

        let bI0 = tid * 4u;
        let bIk = bI0 / TN; let bIn = bI0 % TN;
        let bwr = kBase + bIk; let bwc = block_col + bIn;
        let bwr_in = bwr < K;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = bwr_in && (bwc + i) < N;
            tileB[bIk * TN + (bIn + i)] = select(0.0, W_mr[bwr * N + bwc + i], okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    // Fused residual add in epilogue: Y = X@W + R.
    if (or + 0u < M && oc + 0u < N) { Y_mr[(or + 0u) * N + (oc + 0u)] = acc00 + R_mr[(or + 0u) * N + (oc + 0u)]; }
    if (or + 0u < M && oc + 1u < N) { Y_mr[(or + 0u) * N + (oc + 1u)] = acc01 + R_mr[(or + 0u) * N + (oc + 1u)]; }
    if (or + 0u < M && oc + 2u < N) { Y_mr[(or + 0u) * N + (oc + 2u)] = acc02 + R_mr[(or + 0u) * N + (oc + 2u)]; }
    if (or + 0u < M && oc + 3u < N) { Y_mr[(or + 0u) * N + (oc + 3u)] = acc03 + R_mr[(or + 0u) * N + (oc + 3u)]; }
    if (or + 1u < M && oc + 0u < N) { Y_mr[(or + 1u) * N + (oc + 0u)] = acc10 + R_mr[(or + 1u) * N + (oc + 0u)]; }
    if (or + 1u < M && oc + 1u < N) { Y_mr[(or + 1u) * N + (oc + 1u)] = acc11 + R_mr[(or + 1u) * N + (oc + 1u)]; }
    if (or + 1u < M && oc + 2u < N) { Y_mr[(or + 1u) * N + (oc + 2u)] = acc12 + R_mr[(or + 1u) * N + (oc + 2u)]; }
    if (or + 1u < M && oc + 3u < N) { Y_mr[(or + 1u) * N + (oc + 3u)] = acc13 + R_mr[(or + 1u) * N + (oc + 3u)]; }
    if (or + 2u < M && oc + 0u < N) { Y_mr[(or + 2u) * N + (oc + 0u)] = acc20 + R_mr[(or + 2u) * N + (oc + 0u)]; }
    if (or + 2u < M && oc + 1u < N) { Y_mr[(or + 2u) * N + (oc + 1u)] = acc21 + R_mr[(or + 2u) * N + (oc + 1u)]; }
    if (or + 2u < M && oc + 2u < N) { Y_mr[(or + 2u) * N + (oc + 2u)] = acc22 + R_mr[(or + 2u) * N + (oc + 2u)]; }
    if (or + 2u < M && oc + 3u < N) { Y_mr[(or + 2u) * N + (oc + 3u)] = acc23 + R_mr[(or + 2u) * N + (oc + 3u)]; }
    if (or + 3u < M && oc + 0u < N) { Y_mr[(or + 3u) * N + (oc + 0u)] = acc30 + R_mr[(or + 3u) * N + (oc + 0u)]; }
    if (or + 3u < M && oc + 1u < N) { Y_mr[(or + 3u) * N + (oc + 1u)] = acc31 + R_mr[(or + 3u) * N + (oc + 1u)]; }
    if (or + 3u < M && oc + 2u < N) { Y_mr[(or + 3u) * N + (oc + 2u)] = acc32 + R_mr[(or + 3u) * N + (oc + 2u)]; }
    if (or + 3u < M && oc + 3u < N) { Y_mr[(or + 3u) * N + (oc + 3u)] = acc33 + R_mr[(or + 3u) * N + (oc + 3u)]; }
}


// --- KERNEL: matmul_t ---
// Y[M, N] = X[M, K] @ W^T   where W is stored row-major as [N, K].
// B-tile holds W^T: tileB[k_local, n_local] = W[block_col+n_local, kBase+k_local].
// Coalesced W load: each thread fetches 4 adjacent W values (same n,
// consecutive k) instead of stride-K-scattered reads.
@group(0) @binding(0) var<storage, read>       X_mt: array<f32>;
@group(0) @binding(1) var<storage, read>       W_mt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y_mt: array<f32>;
@group(0) @binding(3) var<uniform> dims_mt: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_t(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_mt.x; let N = dims_mt.y; let K = dims_mt.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        // A tile: same as matmul (X is [M, K], natural row-major).
        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_mt[axr * K + axc + i], okay);
        }

        // B tile: tileB[k, n] = W[n_global, k_global]. Coalesced load:
        // each thread reads 4 consecutive K values at fixed N (W is [N, K]
        // row-major, so consecutive K within a row is consecutive memory).
        let bI0 = tid * 4u;
        let bIn = bI0 / TK;       // n_local ∈ [0, 64)
        let bIk = bI0 % TK;       // k_local ∈ [0, 16); always multiple of 4
        let nG = block_col + bIn; let kG = kBase + bIk;
        let n_in = nG < N;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = n_in && (kG + i) < K;
            tileB[(bIk + i) * TN + bIn] = select(0.0, W_mt[nG * K + kG + i], okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < M && oc + 0u < N) { Y_mt[(or + 0u) * N + (oc + 0u)] = acc00; }
    if (or + 0u < M && oc + 1u < N) { Y_mt[(or + 0u) * N + (oc + 1u)] = acc01; }
    if (or + 0u < M && oc + 2u < N) { Y_mt[(or + 0u) * N + (oc + 2u)] = acc02; }
    if (or + 0u < M && oc + 3u < N) { Y_mt[(or + 0u) * N + (oc + 3u)] = acc03; }
    if (or + 1u < M && oc + 0u < N) { Y_mt[(or + 1u) * N + (oc + 0u)] = acc10; }
    if (or + 1u < M && oc + 1u < N) { Y_mt[(or + 1u) * N + (oc + 1u)] = acc11; }
    if (or + 1u < M && oc + 2u < N) { Y_mt[(or + 1u) * N + (oc + 2u)] = acc12; }
    if (or + 1u < M && oc + 3u < N) { Y_mt[(or + 1u) * N + (oc + 3u)] = acc13; }
    if (or + 2u < M && oc + 0u < N) { Y_mt[(or + 2u) * N + (oc + 0u)] = acc20; }
    if (or + 2u < M && oc + 1u < N) { Y_mt[(or + 2u) * N + (oc + 1u)] = acc21; }
    if (or + 2u < M && oc + 2u < N) { Y_mt[(or + 2u) * N + (oc + 2u)] = acc22; }
    if (or + 2u < M && oc + 3u < N) { Y_mt[(or + 2u) * N + (oc + 3u)] = acc23; }
    if (or + 3u < M && oc + 0u < N) { Y_mt[(or + 3u) * N + (oc + 0u)] = acc30; }
    if (or + 3u < M && oc + 1u < N) { Y_mt[(or + 3u) * N + (oc + 1u)] = acc31; }
    if (or + 3u < M && oc + 2u < N) { Y_mt[(or + 3u) * N + (oc + 2u)] = acc32; }
    if (or + 3u < M && oc + 3u < N) { Y_mt[(or + 3u) * N + (oc + 3u)] = acc33; }
}


// --- KERNEL: matmul_t_acc ---
// Y[M, N] += X[M, K] @ W^T   (same as matmul_t but accumulating).
// Used by backward to fold dQ/dK/dV @ W^T contributions into dx_norm
// without an extra axpy + scratch buffer.
@group(0) @binding(0) var<storage, read>       X_mta: array<f32>;
@group(0) @binding(1) var<storage, read>       W_mta: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y_mta: array<f32>;
@group(0) @binding(3) var<uniform> dims_mta: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_t_acc(@builtin(workgroup_id) wgid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_mta.x; let N = dims_mta.y; let K = dims_mta.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_mta[axr * K + axc + i], okay);
        }

        // Coalesced W^T load — see matmul_t for the rationale.
        let bI0 = tid * 4u;
        let bIn = bI0 / TK; let bIk = bI0 % TK;
        let nG = block_col + bIn; let kG = kBase + bIk;
        let n_in = nG < N;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = n_in && (kG + i) < K;
            tileB[(bIk + i) * TN + bIn] = select(0.0, W_mta[nG * K + kG + i], okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    // Accumulating epilogue: Y[i] += acc.
    if (or + 0u < M && oc + 0u < N) { let i = (or + 0u) * N + (oc + 0u); Y_mta[i] = Y_mta[i] + acc00; }
    if (or + 0u < M && oc + 1u < N) { let i = (or + 0u) * N + (oc + 1u); Y_mta[i] = Y_mta[i] + acc01; }
    if (or + 0u < M && oc + 2u < N) { let i = (or + 0u) * N + (oc + 2u); Y_mta[i] = Y_mta[i] + acc02; }
    if (or + 0u < M && oc + 3u < N) { let i = (or + 0u) * N + (oc + 3u); Y_mta[i] = Y_mta[i] + acc03; }
    if (or + 1u < M && oc + 0u < N) { let i = (or + 1u) * N + (oc + 0u); Y_mta[i] = Y_mta[i] + acc10; }
    if (or + 1u < M && oc + 1u < N) { let i = (or + 1u) * N + (oc + 1u); Y_mta[i] = Y_mta[i] + acc11; }
    if (or + 1u < M && oc + 2u < N) { let i = (or + 1u) * N + (oc + 2u); Y_mta[i] = Y_mta[i] + acc12; }
    if (or + 1u < M && oc + 3u < N) { let i = (or + 1u) * N + (oc + 3u); Y_mta[i] = Y_mta[i] + acc13; }
    if (or + 2u < M && oc + 0u < N) { let i = (or + 2u) * N + (oc + 0u); Y_mta[i] = Y_mta[i] + acc20; }
    if (or + 2u < M && oc + 1u < N) { let i = (or + 2u) * N + (oc + 1u); Y_mta[i] = Y_mta[i] + acc21; }
    if (or + 2u < M && oc + 2u < N) { let i = (or + 2u) * N + (oc + 2u); Y_mta[i] = Y_mta[i] + acc22; }
    if (or + 2u < M && oc + 3u < N) { let i = (or + 2u) * N + (oc + 3u); Y_mta[i] = Y_mta[i] + acc23; }
    if (or + 3u < M && oc + 0u < N) { let i = (or + 3u) * N + (oc + 0u); Y_mta[i] = Y_mta[i] + acc30; }
    if (or + 3u < M && oc + 1u < N) { let i = (or + 3u) * N + (oc + 1u); Y_mta[i] = Y_mta[i] + acc31; }
    if (or + 3u < M && oc + 2u < N) { let i = (or + 3u) * N + (oc + 2u); Y_mta[i] = Y_mta[i] + acc32; }
    if (or + 3u < M && oc + 3u < N) { let i = (or + 3u) * N + (oc + 3u); Y_mta[i] = Y_mta[i] + acc33; }
}


// --- KERNEL: matmul_at ---
// Y[K, N] = X^T @ B   where X is [M, K] (so X^T is [K, M]), B is [M, N].
// dims layout: (K, N, M, _).
//
// A-tile holds X^T: tileA[k_local, m_local] = X[m_global, k_global].
// Coalesced X load: each thread fetches 4 adjacent X values (same m,
// consecutive k) so a 32-thread warp covers TM=64 contiguous K elements
// per m-row in two cache lines. Reduction over M.
@group(0) @binding(0) var<storage, read>       X_mat: array<f32>;
@group(0) @binding(1) var<storage, read>       B_mat: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y_mat: array<f32>;
@group(0) @binding(3) var<uniform> dims_mat: vec4<u32>; // (K, N, M, _)

@compute @workgroup_size(16, 16, 1)
fn matmul_at(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let K = dims_mat.x; let N = dims_mat.y; let M = dims_mat.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    // Output is [K, N]: rows in K dim, cols in N dim.
    let block_row = wgid.y * TM;   // K-dim
    let block_col = wgid.x * TN;   // N-dim

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    // Reduction over M (the inner dim).
    let nTiles = (M + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let mBase = t * TK;

        // A tile [TM × TK] holds X^T: tileA[k_local, m_local] = X[m_global, k_global].
        // Coalesced load: aI/TM = m_local, aI%TM = k_local — consecutive
        // threads read consecutive K values (X is row-major in K → adjacent
        // memory), then advance by m row.
        let aI0 = tid * 4u;
        let aIm = aI0 / TM;       // m_local ∈ [0, 16)
        let aIk = aI0 % TM;       // k_local ∈ [0, 64); always multiple of 4
        let kG = block_row + aIk; let mG = mBase + aIm;
        let m_in = mG < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = m_in && (kG + i) < K;
            tileA[(aIk + i) * TK_PAD + aIm] = select(0.0, X_mat[mG * K + kG + i], okay);
        }

        // B tile [TK × TN]: same as matmul tileB (B is [M, N], natural row-major).
        let bI0 = tid * 4u;
        let bIm = bI0 / TN;       // m_local in TK range = [0, 16)
        let bIn = bI0 % TN;       // n_local ∈ [0, 64); always multiple of 4
        let mGB = mBase + bIm; let nG = block_col + bIn;
        let mGB_in = mGB < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = mGB_in && (nG + i) < N;
            tileB[bIm * TN + (bIn + i)] = select(0.0, B_mat[mGB * N + nG + i], okay);
        }

        workgroupBarrier();

        for (var m: u32 = 0u; m < TK; m = m + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + m];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + m];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + m];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + m];
            let b0 = tileB[m * TN + (4u * tx + 0u)];
            let b1 = tileB[m * TN + (4u * tx + 1u)];
            let b2 = tileB[m * TN + (4u * tx + 2u)];
            let b3 = tileB[m * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < K && oc + 0u < N) { Y_mat[(or + 0u) * N + (oc + 0u)] = acc00; }
    if (or + 0u < K && oc + 1u < N) { Y_mat[(or + 0u) * N + (oc + 1u)] = acc01; }
    if (or + 0u < K && oc + 2u < N) { Y_mat[(or + 0u) * N + (oc + 2u)] = acc02; }
    if (or + 0u < K && oc + 3u < N) { Y_mat[(or + 0u) * N + (oc + 3u)] = acc03; }
    if (or + 1u < K && oc + 0u < N) { Y_mat[(or + 1u) * N + (oc + 0u)] = acc10; }
    if (or + 1u < K && oc + 1u < N) { Y_mat[(or + 1u) * N + (oc + 1u)] = acc11; }
    if (or + 1u < K && oc + 2u < N) { Y_mat[(or + 1u) * N + (oc + 2u)] = acc12; }
    if (or + 1u < K && oc + 3u < N) { Y_mat[(or + 1u) * N + (oc + 3u)] = acc13; }
    if (or + 2u < K && oc + 0u < N) { Y_mat[(or + 2u) * N + (oc + 0u)] = acc20; }
    if (or + 2u < K && oc + 1u < N) { Y_mat[(or + 2u) * N + (oc + 1u)] = acc21; }
    if (or + 2u < K && oc + 2u < N) { Y_mat[(or + 2u) * N + (oc + 2u)] = acc22; }
    if (or + 2u < K && oc + 3u < N) { Y_mat[(or + 2u) * N + (oc + 3u)] = acc23; }
    if (or + 3u < K && oc + 0u < N) { Y_mat[(or + 3u) * N + (oc + 0u)] = acc30; }
    if (or + 3u < K && oc + 1u < N) { Y_mat[(or + 3u) * N + (oc + 1u)] = acc31; }
    if (or + 3u < K && oc + 2u < N) { Y_mat[(or + 3u) * N + (oc + 2u)] = acc32; }
    if (or + 3u < K && oc + 3u < N) { Y_mat[(or + 3u) * N + (oc + 3u)] = acc33; }
}


// --- KERNEL: matmul_at_acc ---
// Y[K, N] += X^T @ B   (same as matmul_at but accumulating).
@group(0) @binding(0) var<storage, read>       X_maa: array<f32>;
@group(0) @binding(1) var<storage, read>       B_maa: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y_maa: array<f32>;
@group(0) @binding(3) var<uniform> dims_maa: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_at_acc(@builtin(workgroup_id) wgid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let K = dims_maa.x; let N = dims_maa.y; let M = dims_maa.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (M + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let mBase = t * TK;

        // Coalesced X load — see matmul_at for the rationale.
        let aI0 = tid * 4u;
        let aIm = aI0 / TM; let aIk = aI0 % TM;
        let kG = block_row + aIk; let mG = mBase + aIm;
        let m_in = mG < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = m_in && (kG + i) < K;
            tileA[(aIk + i) * TK_PAD + aIm] = select(0.0, X_maa[mG * K + kG + i], okay);
        }

        let bI0 = tid * 4u;
        let bIm = bI0 / TN; let bIn = bI0 % TN;
        let mGB = mBase + bIm; let nG = block_col + bIn;
        let mGB_in = mGB < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = mGB_in && (nG + i) < N;
            tileB[bIm * TN + (bIn + i)] = select(0.0, B_maa[mGB * N + nG + i], okay);
        }

        workgroupBarrier();

        for (var m: u32 = 0u; m < TK; m = m + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + m];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + m];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + m];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + m];
            let b0 = tileB[m * TN + (4u * tx + 0u)];
            let b1 = tileB[m * TN + (4u * tx + 1u)];
            let b2 = tileB[m * TN + (4u * tx + 2u)];
            let b3 = tileB[m * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < K && oc + 0u < N) { let i = (or + 0u) * N + (oc + 0u); Y_maa[i] = Y_maa[i] + acc00; }
    if (or + 0u < K && oc + 1u < N) { let i = (or + 0u) * N + (oc + 1u); Y_maa[i] = Y_maa[i] + acc01; }
    if (or + 0u < K && oc + 2u < N) { let i = (or + 0u) * N + (oc + 2u); Y_maa[i] = Y_maa[i] + acc02; }
    if (or + 0u < K && oc + 3u < N) { let i = (or + 0u) * N + (oc + 3u); Y_maa[i] = Y_maa[i] + acc03; }
    if (or + 1u < K && oc + 0u < N) { let i = (or + 1u) * N + (oc + 0u); Y_maa[i] = Y_maa[i] + acc10; }
    if (or + 1u < K && oc + 1u < N) { let i = (or + 1u) * N + (oc + 1u); Y_maa[i] = Y_maa[i] + acc11; }
    if (or + 1u < K && oc + 2u < N) { let i = (or + 1u) * N + (oc + 2u); Y_maa[i] = Y_maa[i] + acc12; }
    if (or + 1u < K && oc + 3u < N) { let i = (or + 1u) * N + (oc + 3u); Y_maa[i] = Y_maa[i] + acc13; }
    if (or + 2u < K && oc + 0u < N) { let i = (or + 2u) * N + (oc + 0u); Y_maa[i] = Y_maa[i] + acc20; }
    if (or + 2u < K && oc + 1u < N) { let i = (or + 2u) * N + (oc + 1u); Y_maa[i] = Y_maa[i] + acc21; }
    if (or + 2u < K && oc + 2u < N) { let i = (or + 2u) * N + (oc + 2u); Y_maa[i] = Y_maa[i] + acc22; }
    if (or + 2u < K && oc + 3u < N) { let i = (or + 2u) * N + (oc + 3u); Y_maa[i] = Y_maa[i] + acc23; }
    if (or + 3u < K && oc + 0u < N) { let i = (or + 3u) * N + (oc + 0u); Y_maa[i] = Y_maa[i] + acc30; }
    if (or + 3u < K && oc + 1u < N) { let i = (or + 3u) * N + (oc + 1u); Y_maa[i] = Y_maa[i] + acc31; }
    if (or + 3u < K && oc + 2u < N) { let i = (or + 3u) * N + (oc + 2u); Y_maa[i] = Y_maa[i] + acc32; }
    if (or + 3u < K && oc + 3u < N) { let i = (or + 3u) * N + (oc + 3u); Y_maa[i] = Y_maa[i] + acc33; }
}


// ════════════════════════════════════════════════════════════
// Mixed-precision (f16 weight) variants
// ════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════
// Mixed-precision variants (f16 weight storage, f32 acc + I/O)
// ════════════════════════════════════════════════════════════
//
// These are the same 64×64 / 4×4 sub-tile structure as the fp32 kernels
// but the W binding is `array<f16>` instead of `array<f32>`. Cast to
// f32 happens at load time into the workgroup tile; the inner loop and
// accumulators stay fp32 (training-stable). Output Y is also f32 —
// Phase 2 only quantizes weights. Phase 3 will add fp16 activations.
//
// Compiled only when adapter has `shader-f16` feature (engine prepends
// `enable f16;` to the shared preamble dynamically).


// --- KERNEL: matmul_w16 ---
// Y[M, N] = X[M, K] @ W[K, N]    where W is f16
@group(0) @binding(0) var<storage, read>       X_w16: array<f32>;
@group(0) @binding(1) var<storage, read>       W_w16: array<f16>;
@group(0) @binding(2) var<storage, read_write> Y_w16: array<f32>;
@group(0) @binding(3) var<uniform> dims_w16: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_w16(@builtin(workgroup_id) wgid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_w16.x; let N = dims_w16.y; let K = dims_w16.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_w16[axr * K + axc + i], okay);
        }

        // f16 W → f32 tile (cast at load time, accumulators stay f32).
        let bI0 = tid * 4u;
        let bIk = bI0 / TN; let bIn = bI0 % TN;
        let bwr = kBase + bIk; let bwc = block_col + bIn;
        let bwr_in = bwr < K;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = bwr_in && (bwc + i) < N;
            tileB[bIk * TN + (bIn + i)] = select(0.0, f32(W_w16[bwr * N + bwc + i]), okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < M && oc + 0u < N) { Y_w16[(or + 0u) * N + (oc + 0u)] = acc00; }
    if (or + 0u < M && oc + 1u < N) { Y_w16[(or + 0u) * N + (oc + 1u)] = acc01; }
    if (or + 0u < M && oc + 2u < N) { Y_w16[(or + 0u) * N + (oc + 2u)] = acc02; }
    if (or + 0u < M && oc + 3u < N) { Y_w16[(or + 0u) * N + (oc + 3u)] = acc03; }
    if (or + 1u < M && oc + 0u < N) { Y_w16[(or + 1u) * N + (oc + 0u)] = acc10; }
    if (or + 1u < M && oc + 1u < N) { Y_w16[(or + 1u) * N + (oc + 1u)] = acc11; }
    if (or + 1u < M && oc + 2u < N) { Y_w16[(or + 1u) * N + (oc + 2u)] = acc12; }
    if (or + 1u < M && oc + 3u < N) { Y_w16[(or + 1u) * N + (oc + 3u)] = acc13; }
    if (or + 2u < M && oc + 0u < N) { Y_w16[(or + 2u) * N + (oc + 0u)] = acc20; }
    if (or + 2u < M && oc + 1u < N) { Y_w16[(or + 2u) * N + (oc + 1u)] = acc21; }
    if (or + 2u < M && oc + 2u < N) { Y_w16[(or + 2u) * N + (oc + 2u)] = acc22; }
    if (or + 2u < M && oc + 3u < N) { Y_w16[(or + 2u) * N + (oc + 3u)] = acc23; }
    if (or + 3u < M && oc + 0u < N) { Y_w16[(or + 3u) * N + (oc + 0u)] = acc30; }
    if (or + 3u < M && oc + 1u < N) { Y_w16[(or + 3u) * N + (oc + 1u)] = acc31; }
    if (or + 3u < M && oc + 2u < N) { Y_w16[(or + 3u) * N + (oc + 2u)] = acc32; }
    if (or + 3u < M && oc + 3u < N) { Y_w16[(or + 3u) * N + (oc + 3u)] = acc33; }
}


// --- KERNEL: matmul_residual_w16 ---
// Y[M, N] = X[M, K] @ W[K, N] + R[M, N]   where W is f16
@group(0) @binding(0) var<storage, read>       X_mrw: array<f32>;
@group(0) @binding(1) var<storage, read>       W_mrw: array<f16>;
@group(0) @binding(2) var<storage, read>       R_mrw: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y_mrw: array<f32>;
@group(0) @binding(4) var<uniform> dims_mrw: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_residual_w16(@builtin(workgroup_id) wgid: vec3<u32>,
                       @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_mrw.x; let N = dims_mrw.y; let K = dims_mrw.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_mrw[axr * K + axc + i], okay);
        }

        let bI0 = tid * 4u;
        let bIk = bI0 / TN; let bIn = bI0 % TN;
        let bwr = kBase + bIk; let bwc = block_col + bIn;
        let bwr_in = bwr < K;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = bwr_in && (bwc + i) < N;
            tileB[bIk * TN + (bIn + i)] = select(0.0, f32(W_mrw[bwr * N + bwc + i]), okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < M && oc + 0u < N) { Y_mrw[(or + 0u) * N + (oc + 0u)] = acc00 + R_mrw[(or + 0u) * N + (oc + 0u)]; }
    if (or + 0u < M && oc + 1u < N) { Y_mrw[(or + 0u) * N + (oc + 1u)] = acc01 + R_mrw[(or + 0u) * N + (oc + 1u)]; }
    if (or + 0u < M && oc + 2u < N) { Y_mrw[(or + 0u) * N + (oc + 2u)] = acc02 + R_mrw[(or + 0u) * N + (oc + 2u)]; }
    if (or + 0u < M && oc + 3u < N) { Y_mrw[(or + 0u) * N + (oc + 3u)] = acc03 + R_mrw[(or + 0u) * N + (oc + 3u)]; }
    if (or + 1u < M && oc + 0u < N) { Y_mrw[(or + 1u) * N + (oc + 0u)] = acc10 + R_mrw[(or + 1u) * N + (oc + 0u)]; }
    if (or + 1u < M && oc + 1u < N) { Y_mrw[(or + 1u) * N + (oc + 1u)] = acc11 + R_mrw[(or + 1u) * N + (oc + 1u)]; }
    if (or + 1u < M && oc + 2u < N) { Y_mrw[(or + 1u) * N + (oc + 2u)] = acc12 + R_mrw[(or + 1u) * N + (oc + 2u)]; }
    if (or + 1u < M && oc + 3u < N) { Y_mrw[(or + 1u) * N + (oc + 3u)] = acc13 + R_mrw[(or + 1u) * N + (oc + 3u)]; }
    if (or + 2u < M && oc + 0u < N) { Y_mrw[(or + 2u) * N + (oc + 0u)] = acc20 + R_mrw[(or + 2u) * N + (oc + 0u)]; }
    if (or + 2u < M && oc + 1u < N) { Y_mrw[(or + 2u) * N + (oc + 1u)] = acc21 + R_mrw[(or + 2u) * N + (oc + 1u)]; }
    if (or + 2u < M && oc + 2u < N) { Y_mrw[(or + 2u) * N + (oc + 2u)] = acc22 + R_mrw[(or + 2u) * N + (oc + 2u)]; }
    if (or + 2u < M && oc + 3u < N) { Y_mrw[(or + 2u) * N + (oc + 3u)] = acc23 + R_mrw[(or + 2u) * N + (oc + 3u)]; }
    if (or + 3u < M && oc + 0u < N) { Y_mrw[(or + 3u) * N + (oc + 0u)] = acc30 + R_mrw[(or + 3u) * N + (oc + 0u)]; }
    if (or + 3u < M && oc + 1u < N) { Y_mrw[(or + 3u) * N + (oc + 1u)] = acc31 + R_mrw[(or + 3u) * N + (oc + 1u)]; }
    if (or + 3u < M && oc + 2u < N) { Y_mrw[(or + 3u) * N + (oc + 2u)] = acc32 + R_mrw[(or + 3u) * N + (oc + 2u)]; }
    if (or + 3u < M && oc + 3u < N) { Y_mrw[(or + 3u) * N + (oc + 3u)] = acc33 + R_mrw[(or + 3u) * N + (oc + 3u)]; }
}


// --- KERNEL: matmul_t_w16 ---
// Y[M, N] = X[M, K] @ W^T   where W stored as [N, K] in f16.
// Coalesced W load: each thread fetches 4 adjacent f16 W values (same n,
// consecutive k) so a warp covers TK=16 contiguous K elements per n-row
// instead of stride-K-scattered reads.
@group(0) @binding(0) var<storage, read>       X_mtw: array<f32>;
@group(0) @binding(1) var<storage, read>       W_mtw: array<f16>;
@group(0) @binding(2) var<storage, read_write> Y_mtw: array<f32>;
@group(0) @binding(3) var<uniform> dims_mtw: vec4<u32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_t_w16(@builtin(workgroup_id) wgid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = dims_mtw.x; let N = dims_mtw.y; let K = dims_mtw.z;
    let tx = lid.x; let ty = lid.y;
    let tid = ty * 16u + tx;

    let block_row = wgid.y * TM;
    let block_col = wgid.x * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let nTiles = (K + TK - 1u) / TK;
    for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
        let kBase = t * TK;

        let aI0 = tid * 4u;
        let aIm = aI0 / TK; let aIk = aI0 % TK;
        let axr = block_row + aIm; let axc = kBase + aIk;
        let row_in = axr < M;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = row_in && (axc + i) < K;
            tileA[aIm * TK_PAD + (aIk + i)] = select(0.0, X_mtw[axr * K + axc + i], okay);
        }

        // Coalesced W^T load (f16 → f32 cast at load time).
        let bI0 = tid * 4u;
        let bIn = bI0 / TK; let bIk = bI0 % TK;
        let nG = block_col + bIn; let kG = kBase + bIk;
        let n_in = nG < N;
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let okay = n_in && (kG + i) < K;
            tileB[(bIk + i) * TN + bIn] = select(0.0, f32(W_mtw[nG * K + kG + i]), okay);
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TK; k = k + 1u) {
            let a0 = tileA[(4u * ty + 0u) * TK_PAD + k];
            let a1 = tileA[(4u * ty + 1u) * TK_PAD + k];
            let a2 = tileA[(4u * ty + 2u) * TK_PAD + k];
            let a3 = tileA[(4u * ty + 3u) * TK_PAD + k];
            let b0 = tileB[k * TN + (4u * tx + 0u)];
            let b1 = tileB[k * TN + (4u * tx + 1u)];
            let b2 = tileB[k * TN + (4u * tx + 2u)];
            let b3 = tileB[k * TN + (4u * tx + 3u)];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01); acc02 = fma(a0, b2, acc02); acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11); acc12 = fma(a1, b2, acc12); acc13 = fma(a1, b3, acc13);
            acc20 = fma(a2, b0, acc20); acc21 = fma(a2, b1, acc21); acc22 = fma(a2, b2, acc22); acc23 = fma(a2, b3, acc23);
            acc30 = fma(a3, b0, acc30); acc31 = fma(a3, b1, acc31); acc32 = fma(a3, b2, acc32); acc33 = fma(a3, b3, acc33);
        }
        workgroupBarrier();
    }

    let or = block_row + 4u * ty;
    let oc = block_col + 4u * tx;
    if (or + 0u < M && oc + 0u < N) { Y_mtw[(or + 0u) * N + (oc + 0u)] = acc00; }
    if (or + 0u < M && oc + 1u < N) { Y_mtw[(or + 0u) * N + (oc + 1u)] = acc01; }
    if (or + 0u < M && oc + 2u < N) { Y_mtw[(or + 0u) * N + (oc + 2u)] = acc02; }
    if (or + 0u < M && oc + 3u < N) { Y_mtw[(or + 0u) * N + (oc + 3u)] = acc03; }
    if (or + 1u < M && oc + 0u < N) { Y_mtw[(or + 1u) * N + (oc + 0u)] = acc10; }
    if (or + 1u < M && oc + 1u < N) { Y_mtw[(or + 1u) * N + (oc + 1u)] = acc11; }
    if (or + 1u < M && oc + 2u < N) { Y_mtw[(or + 1u) * N + (oc + 2u)] = acc12; }
    if (or + 1u < M && oc + 3u < N) { Y_mtw[(or + 1u) * N + (oc + 3u)] = acc13; }
    if (or + 2u < M && oc + 0u < N) { Y_mtw[(or + 2u) * N + (oc + 0u)] = acc20; }
    if (or + 2u < M && oc + 1u < N) { Y_mtw[(or + 2u) * N + (oc + 1u)] = acc21; }
    if (or + 2u < M && oc + 2u < N) { Y_mtw[(or + 2u) * N + (oc + 2u)] = acc22; }
    if (or + 2u < M && oc + 3u < N) { Y_mtw[(or + 2u) * N + (oc + 3u)] = acc23; }
    if (or + 3u < M && oc + 0u < N) { Y_mtw[(or + 3u) * N + (oc + 0u)] = acc30; }
    if (or + 3u < M && oc + 1u < N) { Y_mtw[(or + 3u) * N + (oc + 1u)] = acc31; }
    if (or + 3u < M && oc + 2u < N) { Y_mtw[(or + 3u) * N + (oc + 2u)] = acc32; }
    if (or + 3u < M && oc + 3u < N) { Y_mtw[(or + 3u) * N + (oc + 3u)] = acc33; }
}
