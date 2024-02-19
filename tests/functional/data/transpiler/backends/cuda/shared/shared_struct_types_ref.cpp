
struct ComplexValueFloat {
  float real;
  float imaginary;
};

//TODO: fix me when @kernel/@outer/@inner will be implementeds
[[okl::kernel("")]] void function1(const int *data) {
  __shared__ ComplexValueFloat arr1[32];
  __shared__ ComplexValueFloat arr2[8][32];
}
