@group(0) @binding(0) var<storage, read_write> m: atomic<i32>;
@group(0) @binding(1) var<storage, read_write> b: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  atomicMax(&m, 3);
  atomicMax(&m, -7);
  atomicMax(&m, 8);
  atomicOr(&b, 0xAu);
  atomicOr(&b, 0x5u);
  o[0] = atomicLoad(&m);        // 8
  o[1] = i32(atomicLoad(&b));   // 0xF = 15
}
