
extern "C" void atomic_sub_builtin(const int* iVec, int* iSum, const float* fVec, float* fSum) {
#pragma omp atomic
    *iSum -= iVec[0];
#pragma omp atomic
    *fSum -= fVec[0];
}

struct ComplexTypeF32 {
    float real;
    float imag;
};


extern "C" void atomic_sub_struct(const ComplexTypeF32* vec, ComplexTypeF32* sum) {
#pragma omp atomic
    sum->real -= vec[0].real;
#pragma omp atomic
    sum->imag -= vec[0].imag;
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};


extern "C" void atomic_sub_template(const ComplexType<float>* vec, ComplexType<float>* sum) {
#pragma omp atomic
    sum->real -= vec[0].real;
#pragma omp atomic
    sum->imag -= vec[0].imag;
}
