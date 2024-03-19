
struct ComplexTypeF32 {
    ComplexTypeF32& operator=(const ComplexTypeF32&) = default;
    float real;
    float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
@kernel void atomic_exch_struct(const ComplexTypeF32* vec, ComplexTypeF32* result) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic* result = vec[0];
        }
    }
}
