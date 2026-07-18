@group(0) @binding(0) var<storage, read_write> B: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) i: u32) {
  B[i] = i + 1u;
}
