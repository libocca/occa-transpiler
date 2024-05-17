
@kernel void atomic_dec_builtin(unsigned int* value) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic--(*value);
        }
    }
    // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};

@kernel void atomic_dec_struct(ComplexMaskType* value) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic-- value->val1;
            @atomic value->val2--;
        }
    }
}
