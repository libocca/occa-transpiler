// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_and_builtin(const unsigned int* masks, unsigned int* mask) {
#pragma omp atomic
    *mask |= masks[0];
}

struct ComplexMaskType {
    unsigned int mask1;
    unsigned int mask2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void atomic_and_struct(const ComplexMaskType* masks, ComplexMaskType* mask) {
#pragma omp atomic
    mask->mask1 |= masks[0].mask1;
#pragma omp atomic
    mask->mask2 |= masks[0].mask2;
}
