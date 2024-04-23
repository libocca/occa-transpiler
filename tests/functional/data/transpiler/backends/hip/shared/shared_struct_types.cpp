
struct ComplexValueFloat {
    float real;
    float imaginary;
};

@kernel void function1(const int* data) {
    @outer for (int i = 0; i < 64; ++i) {
        @shared ComplexValueFloat arr2[8][32];
        @shared ComplexValueFloat arr1[32];
        @inner for (int j = 0; j < 64; ++j) {
        }
    }
}
