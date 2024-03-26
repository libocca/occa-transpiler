
extern "C" void atomic_dec_builtin(unsigned int* value) {
    --(*value);
    // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};


extern "C" void atomic_dec_struct(ComplexMaskType* value) {
    --value->val1;
    value->val2--;
}
