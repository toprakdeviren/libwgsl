@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let r = atomicCompareExchangeWeak(&c, 0u, 5u);
  let e = select(0u, 1u, r.exchanged);
  o[0] = r.old_value + e * 10u + atomicLoad(&c) * 100u;
}
