
template <class T>
struct ComplexType {
    T real;
    T imaginary;
};

@kernel void function1(const int* data) {
    @outer for (int i = 0; i < 64; ++i) {
        @shared ComplexType<int> arr1[32];
        @shared ComplexType<float> arr2[8][32];
        @inner for (int j = 0; j < 64; ++j) {
        }
    }
}
