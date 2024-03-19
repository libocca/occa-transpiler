#define SYNC_BLOCK(a, b, c) \
    {                       \
        a = a * b + c;      \
        @barrier;           \
    }                       \
    while (0)               \
        ;

#define __okl_inline__ inline
#define __okl_inline__ inline
#define __okl_inline__ inline
#define __okl_inline__ inline

#define USE_THAT 12

__okl_inline__ int fastMul(float a) {
    return a * 1.f;
}

@kernel void simple_function(const float* inputArray @ restrict,
                             float* outputArray,
                             float value,
                             int size) {
    for (int i = 0; i < size; ++i; @outer) {
        outputArray[i] = inputArray[i] + value;
#if USE_THIS
        SYNC_BLOCK(outputArray[i], inputArray[i], value)
#elif USE_THAT
        SYNC_BLOCK(outputArray[i + 2], inputArray[i - 2], value)
#endif
    }

    for (int i = 0; i < size; ++i; @tile(8)) {
        outputArray[i] = fastMul(inputArray[i]) + value;
    }
}
