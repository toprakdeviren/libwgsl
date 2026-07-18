@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = min(3, max(1, 2));          // 2
  o[1] = select(100, 200, true);     // 200
  o[2] = select(100, 200, false);    // 100
  o[3] = clamp(15, 0, 10);           // 10
}
