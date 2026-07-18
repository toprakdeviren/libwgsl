enable f16;
@group(0) @binding(0) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
  let a: f16 = 1.5h;
  let b: f16 = 2.0h;
  o[0] = a * b;
}
