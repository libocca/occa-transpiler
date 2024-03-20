
@kernel void atomic_exch_builtin(const int* iVec, int* iSum, const float* fVec, float* fSum) {
    @atomic* iSum = iVec[0];
    @atomic* fSum = fVec[0];
}

struct ComplexTypeF32 {
    float real;
    float imag;
};


@kernel void atomic_exch_struct(const ComplexTypeF32* vec, ComplexTypeF32* result) {
    @atomic* result = vec[0];
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};


@kernel void atomic_exch_template(const ComplexType<float>* vec, ComplexType<float>* result) {
    @atomic* result = vec[0];
}
