struct S { a: i32, b: i32, c: i32 }
@group(0) @binding(0) var<storage, read_write> s: S;
@compute @workgroup_size(1)
fn main() {
  s.a = 1;
  s.b = 2;
  s.c = s.a + s.b * 10; // 21
}
