fn f() {
  var i = 0;
  loop {
    if (i >= 3) { break; }
    i++;
    continuing { i = i; }
  }
}
