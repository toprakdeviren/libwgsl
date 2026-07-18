@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var r = 0;
  if (true) {
    r = 7;
  } else {
    r = 9;
  }
  o[0] = r;
}
