
@kernel void atomic_and_builtin(const unsigned int* masks, unsigned int* mask) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic* mask ^= masks[0];
        }
    }
}

struct ComplexMaskType {
    unsigned int mask1;
    unsigned int mask2;
};


@kernel void atomic_and_struct(const ComplexMaskType* masks, ComplexMaskType* mask) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @atomic mask->mask1 ^= masks[0].mask1;
            @atomic mask->mask2 ^= masks[0].mask2;
        }
    }
}
