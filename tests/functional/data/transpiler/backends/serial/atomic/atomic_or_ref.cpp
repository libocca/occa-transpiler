// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_and_builtin(const unsigned int* masks, unsigned int* mask) {
    *mask |= masks[0];
}

struct ComplexMaskType {
    unsigned int mask1;
    unsigned int mask2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_and_struct(const ComplexMaskType* masks, ComplexMaskType* mask) {
    mask->mask1 |= masks[0].mask1;
    mask->mask2 |= masks[0].mask2;
}
