
template<class T>
struct ComplexType {
  T real;
  T imaginary;
};

//TODO: fix me when @kernel/@outer/@inner will be implementeds
@kernel void function1(const int *data) {
  @shared ComplexType<int> arr1[32];
  @shared ComplexType<float> arr2[8][32];
}
