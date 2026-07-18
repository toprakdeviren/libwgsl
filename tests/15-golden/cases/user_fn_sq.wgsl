fn sq(x: i32) -> i32 { return x * x; }
@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = sq(7) + sq(3); // 49+9 = 58
}
