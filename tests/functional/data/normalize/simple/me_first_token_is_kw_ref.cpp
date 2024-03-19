template <typename T>
struct TypedStruct {
    T val;
};

[[okl::kernel("")]] void simple_function([[okl::restrict("")]] const float* inputArray,
                                         float* outputArray,
                                         float value,
                                         int size) {
    TypedStruct<decltype(value)> valuename;
    [[okl::outer("")]] for (int i = 0; i < size; ++i) { outputArray[i] = inputArray[i] + value; }
}
