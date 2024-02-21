//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_exch_builtin(const int *iVec,
                        int *iSum,
                        const float *fVec,
                        float *fSum)
{
     atomicExch(&(*iSum), iVec[0]);
     atomicExch(&(*fSum), fVec[0]);
}

struct ComplexTypeF32 {
    float real;
    float imag;
};

//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_exch_struct(const ComplexTypeF32 *vec,
                        ComplexTypeF32 *result)
{
     atomicExch(&(*result), vec[0]);
}

template<class T>
struct ComplexType {
    T real;
    T imag;
};

//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_exch_template(const ComplexType<float> *vec,
                                 ComplexType<float> *result)
{
     atomicExch(&(*result), vec[0]);
}
