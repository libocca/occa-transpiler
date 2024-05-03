[[okl_kernel("")]] void simple_function([[okl_restrict("")]] const float* inputArray,
                                         float* outputArray,
                                         float value,
                                         int size) {
    [[okl_outer("")]] for (int i = 0; i < size; ++i) { outputArray[i] = inputArray[i] + value; }
}
