@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  // column-major: mat2x2f(c0x,c0y, c1x,c1y)
  let A = mat2x2f(1.0, 2.0, 3.0, 4.0);
  let B = mat2x2f(5.0, 6.0, 7.0, 8.0);
  let C = A * B;
  o[0] = C[0][0];
  o[1] = C[0][1];
  o[2] = C[1][0];
  o[3] = C[1][1];
}
