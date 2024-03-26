@kernel void atomic_add_builtin(const int* iVec, int* iSum, const float* fVec, float* fSum) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic* iSum += iVec[0];
            @atomic* fSum += fVec[0];
        }
    }
}

struct ComplexTypeF32 {
    float real;
    float imag;
};

@kernel void atomic_add_struct(const ComplexTypeF32* vec, ComplexTypeF32* sum) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic sum->real += vec[0].real;
            @atomic sum->imag += vec[0].imag;
        }
    }
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};

@kernel void atomic_add_template(const ComplexType<float>* vec, ComplexType<float>* sum) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic sum->real += vec[0].real;
            @atomic sum->imag += vec[0].imag;
        }
    }
}
