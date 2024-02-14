
struct ComplexValueFloat {
  float real;
  float imaginary;
};

@kernel void function1(const int *data) {
  @shared ComplexValueFloat var1;
  @shared ComplexValueFloat arr1[32];
  @shared ComplexValueFloat arr2[8][32];
}

