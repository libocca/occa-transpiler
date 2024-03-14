#define SYNC_BLOCK(a, b, c) \
    {                       \
        a = a * b + c;      \
        @barrier;           \
    }                       \
    while (0);

#define __okl_inline__ inline

inline  int fastMul(float a) {
    return a * 1.f;
}

[[okl::kernel("")]] void simple_function([[okl::restrict("")]]const float* inputArray ,
                             float* outputArray,
                             float value,
                             int size) {
    [[okl::outer("")]]for (int i = 0; i < size; ++i)   {
        outputArray[i] = inputArray[i] + value;
        {outputArray [i ]=outputArray [i ]*inputArray [i ]+value ;[[okl::barrier("")]] ;}while (0);
    }

    [[okl::tile("(8)")]]for (int i = 0; i < size; ++i)   {
        outputArray[i] = fastMul(inputArray[i]) + value;
    }

}
