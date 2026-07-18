@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var acc = 0;
  for (var i = 0; i < 4; i = i + 1) {
    acc = acc + i;
  }
  o[0] = acc;
}
