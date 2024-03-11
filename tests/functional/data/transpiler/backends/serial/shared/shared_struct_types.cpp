struct ComplexValueFloat {
    float real;
    float imaginary;
};

@kernel void function1(const int* data) {
    for (int i = 0; i < 32; ++i; @outer) {
        @shared ComplexValueFloat arr1[32];
        @shared ComplexValueFloat arr2[8][32];
        for (int j = 0; j < 8; ++j; @inner) {
            arr1[i].real += arr2[j][i].real;
            arr1[i].imaginary += arr2[j][i].imaginary;
        }
    }
}
