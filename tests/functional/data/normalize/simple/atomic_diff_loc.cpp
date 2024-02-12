@kernel void f(float a) {
  @atomic a = a + 1;
  [[okl::atomic("")]] a += 1;

  {
    float b;
    @atomic b = a + b;
  }

  int c = 0;
  @atomic {
    a *= 1;
    c += a;
  }
}
