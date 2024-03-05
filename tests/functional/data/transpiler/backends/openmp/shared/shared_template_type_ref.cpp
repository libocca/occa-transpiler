template <class T>
struct ComplexType {
    T real;
    T imaginary;
};

extern "C" void function1(const int* data) {
#pragma omp parallel for
    for (int i = 0; i < 32; ++i) {
        ComplexType<int> arr1[32];
        ComplexType<float> arr2[8][32];
        for (int j = 0; j < 8; ++j) {
            arr1[i].real += int(arr2[j][i].real);
            arr1[i].imaginary += int(arr2[j][i].imaginary);
        }
    }
}
