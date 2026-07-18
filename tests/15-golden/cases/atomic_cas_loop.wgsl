@group(0) @binding(0) var<storage, read_write> c: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main() {
  var done = false;
  var tries = 0u;
  loop {
    if (done) { break; }
    let r = atomicCompareExchangeWeak(&c, 0u, 7u);
    tries = tries + 1u;
    if (r.exchanged) { done = true; }
    if (tries > 4u) { break; }
  }
  o[0] = atomicLoad(&c);
  o[1] = tries;
}
