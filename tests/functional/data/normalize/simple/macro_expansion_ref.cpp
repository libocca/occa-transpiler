

inline int fastMul(float a) {
    return a * 1.f;
}

[[okl_kernel("")]] void simple_function([[okl_restrict("")]] const float* inputArray,
                                         float* outputArray,
                                         float value,
                                         int size) {
    [[okl_outer("")]] for (int i = 0; i < size; ++i) {
        outputArray[i] = inputArray[i] + value;
#if 0
        SYNC_BLOCK(outputArray[i], inputArray[i], value)
#elif 1
        {
            outputArray[i + 2] = outputArray[i + 2] * inputArray[i - 2] + value;
            [[okl_barrier("")]];
        }
        while (0)
            ;
#endif
    }

    [[okl_tile("(8)")]] for (int i = 0; i < size; ++i) {
        outputArray[i] = fastMul(inputArray[i]) + value;
    }
}
