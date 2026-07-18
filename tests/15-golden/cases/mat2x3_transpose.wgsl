@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main() {
  // mat2x3 columns [1,2,3] [4,5,6]
  let M = mat2x3f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  let T = transpose(M); // mat3x2 columns [1,4] [2,5] [3,6]
  o[0] = T[0][0];
  o[1] = T[0][1];
  o[2] = T[1][0];
  o[3] = T[1][1];
  o[4] = T[2][0];
  o[5] = T[2][1];
}
