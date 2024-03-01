
struct ComplexValueFloat {
    float real;
    float imaginary;
};

// TODO: fix me when @kernel/@outer/@inner will be implementeds
@kernel void function1(const int* data) {
    @shared ComplexValueFloat arr1[32];
    @shared ComplexValueFloat arr2[8][32];
}
