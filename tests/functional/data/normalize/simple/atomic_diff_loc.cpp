@kernel void f(float a) {
  @atomic a= a + 1;
  [[okl::atomic("")]] a+= 1;

  {
    float b;
    b = a+b @atomic;
    @atomic b = a+b;
  }

  {
    a *= 1 @atomic;
  }
}
