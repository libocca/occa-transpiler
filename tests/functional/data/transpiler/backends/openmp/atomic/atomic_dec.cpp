
@kernel void atomic_dec_builtin(unsigned int* value) {
    @atomic--(*value);
    // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};


@kernel void atomic_dec_struct(ComplexMaskType* value) {
    @atomic-- value->val1;
    @atomic value->val2--;
}
