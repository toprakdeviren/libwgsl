@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;
@compute @workgroup_size(16)
fn main() {
  atomicAdd(&c, 1u);
}
