@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  // columns [1,3] [2,4] → det = 1*4 - 2*3 = -2
  let M = mat2x2f(1.0, 3.0, 2.0, 4.0);
  o[0] = determinant(M);
}
