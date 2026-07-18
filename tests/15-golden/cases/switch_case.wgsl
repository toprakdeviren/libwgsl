@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var s = 2;
  var r = 0;
  switch (s) {
    case 0: { r = 100; }
    case 2: { r = 222; }
    default: { r = 999; }
  }
  o[0] = r;
}
