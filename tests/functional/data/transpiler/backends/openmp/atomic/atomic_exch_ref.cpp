// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_exch_builtin(const int* iVec, int* iSum, const float* fVec, float* fSum) {
#pragma omp atomic
    *iSum = iVec[0];
#pragma omp atomic
    *fSum = fVec[0];
}

struct ComplexTypeF32 {
    float real;
    float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_exch_struct(const ComplexTypeF32* vec, ComplexTypeF32* result) {
#pragma omp atomic
    *result = vec[0];
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_exch_template(const ComplexType<float>* vec, ComplexType<float>* result) {
#pragma omp atomic
    *result = vec[0];
}
