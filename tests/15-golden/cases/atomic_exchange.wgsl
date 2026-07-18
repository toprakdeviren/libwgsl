@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  atomicStore(&c, 9u);
  let old = atomicExchange(&c, 42u);
  o[0] = old;
  o[1] = atomicLoad(&c);
}
