@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  o[0] = (7 + 3) * 4 - 5;       // 35
  o[1] = i32((0xFu << 2u) | 1u); // 61
  o[2] = abs(-12);               // 12
  o[3] = 100 % 7;                // 2
}
