@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@compute @workgroup_size(1)
fn main() {
  var acc = 0;
  var i = 0;
  loop {
    acc = acc + i;
    continuing {
      i = i + 1;
      break if i >= 5;
    }
  }
  o[0] = acc; // 0+1+2+3+4 = 10
  o[1] = i;   // 5
}
