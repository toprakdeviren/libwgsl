@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var acc = 0;
  for (var i = 0; i < 10; i = i + 1) {
    if (i % 2 == 0) { continue; }
    acc = acc + i; // 1+3+5+7+9 = 25
  }
  o[0] = acc;
}
