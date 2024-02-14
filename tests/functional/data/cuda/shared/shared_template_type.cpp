
template<class T>
struct ComplexType {
  T real;
  T imaginary;
};

@kernel void function1(const int *data) {
  @shared ComplexType<double> var1;
  @shared ComplexType<int> arr1[32];
  @shared ComplexType<float> arr2[8][32];
}

