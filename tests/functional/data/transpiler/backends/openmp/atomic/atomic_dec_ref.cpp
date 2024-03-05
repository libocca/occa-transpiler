// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_dec_builtin(unsigned int* value) {
#pragma omp atomic
    --(*value);
    // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_dec_struct(ComplexMaskType* value) {
#pragma omp atomic
    --value->val1;
#pragma omp atomic
    value->val2--;
}
