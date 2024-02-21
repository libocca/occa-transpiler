//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_inc_builtin(unsigned int *value)
{
     atomicInc(&((*value)), 1);
    // @atomic (*value)++; normalizer issue
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};

//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void atomic_inc_struct(ComplexMaskType *value)
{
     atomicInc(&(value->val1), 1);
     atomicInc(&(value->val2), 1);
}
