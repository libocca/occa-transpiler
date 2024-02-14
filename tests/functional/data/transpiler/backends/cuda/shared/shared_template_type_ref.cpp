
template<class T>
struct ComplexType {
  T real;
  T imaginary;
};

//TODO: fix me when @kernel/@outer/@inner will be implementeds
[[okl::kernel("")]] void function1(const int *data) {
  __shared__ ComplexType<int> arr1[32];
  __shared__ ComplexType<float> arr2[8][32];
}

