@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var x = 3;
  var r = 0;
  if (x > 0) {
    if (x > 2) { r = 100; } else { r = 50; }
  } else {
    r = -1;
  }
  o[0] = r;
}
