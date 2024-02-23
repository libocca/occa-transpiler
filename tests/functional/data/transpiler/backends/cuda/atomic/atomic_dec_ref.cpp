// TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_dec_builtin(unsigned int* value) {
    atomicDec(&((*value)), 1);
    // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_dec_struct(ComplexMaskType* value) {
    atomicDec(&(value->val1), 1);
    atomicDec(&(value->val2), 1);
}
