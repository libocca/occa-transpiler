template <class T>
struct ComplexType {
    T real;
    T imaginary;
};

@kernel void function1(const int* data) {
    for (int i = 0; i < 32; ++i; @outer) {
        @shared ComplexType<int> arr1[32];
        @shared ComplexType<float> arr2[8][32];
        for (int j = 0; j < 8; ++j; @inner) {
            arr1[i].real += int(arr2[j][i].real);
            arr1[i].imaginary += int(arr2[j][i].imaginary);
        }
    }
}
