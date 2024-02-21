//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_and_builtin(const unsigned int *masks,
                        unsigned int *mask)
{
     atomicOr(&(*mask), masks[0]);
}

struct ComplexMaskType {
    unsigned int mask1;
    unsigned int mask2;
};

//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_and_struct(const ComplexMaskType *masks,
                        ComplexMaskType *mask)
{
     atomicOr(&(mask->mask1), masks[0].mask1);
     atomicOr(&(mask->mask2), masks[0].mask2);
}
