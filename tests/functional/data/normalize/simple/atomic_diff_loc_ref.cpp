[[okl::kernel("")]] void f(float a) {
  [[okl::atomic("")]] a= a + 1;
  [[okl::atomic("")]] a+= 1;

  {
    float b;
    [[okl::atomic("")]]b = a+b ;
    [[okl::atomic("")]] b = a+b;
  }

  {
    [[okl::atomic("")]]a *= 1 ;
  }
}
