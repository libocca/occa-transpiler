// TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_add_builtin(const int* iVec,
                                            int* iSum,
                                            const float* fVec,
                                            float* fSum) {
    atomicAdd(&(*iSum), iVec[0]);
    atomicAdd(&(*fSum), fVec[0]);
}

struct ComplexTypeF32 {
    float real;
    float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_add_struct(const ComplexTypeF32* vec, ComplexTypeF32* sum) {
    atomicAdd(&(sum->real), vec[0].real);
    atomicAdd(&(sum->imag), vec[0].imag);
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_add_template(const ComplexType<float>* vec,
                                             ComplexType<float>* sum) {
    atomicAdd(&(sum->real), vec[0].real);
    atomicAdd(&(sum->imag), vec[0].imag);
}
