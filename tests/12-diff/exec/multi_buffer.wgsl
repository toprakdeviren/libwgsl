@group(0) @binding(0) var<storage, read_write> a: array<i32>;
@group(0) @binding(1) var<storage, read_write> b: array<i32>;
@compute @workgroup_size(1)
fn main() {
  a[0] = 10;
  a[1] = 20;
  b[0] = a[0] + a[1];
  b[1] = a[0] * a[1];
}
