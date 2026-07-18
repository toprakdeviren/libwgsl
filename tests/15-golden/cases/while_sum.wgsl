@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var acc = 0;
  var i = 0;
  while (i < 10) {
    acc = acc + i;
    i = i + 1;
  }
  o[0] = acc;
}
