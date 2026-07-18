@group(0) @binding(0) var<storage, read_write> x: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  atomicXor(&x, 0xFFu);
  atomicXor(&x, 0x0Fu);
  o[0] = atomicLoad(&x); // 0xF0 = 240
}
