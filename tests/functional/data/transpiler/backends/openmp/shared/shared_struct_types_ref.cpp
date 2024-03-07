struct ComplexValueFloat {
    float real;
    float imaginary;
};

extern "C" void function1(const int* data) {
#pragma omp parallel for
    for (int i = 0; i < 32; ++i) {
        ComplexValueFloat arr1[32];
        ComplexValueFloat arr2[8][32];
        for (int j = 0; j < 8; ++j) {
            arr1[i].real += arr2[j][i].real;
            arr1[i].imaginary += arr2[j][i].imaginary;
        }
    }
}
