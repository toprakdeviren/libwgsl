var<private> p: i32 = 0;
@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  p = p + 5;
  p = p * 3;
  o[0] = p; // 15
}
