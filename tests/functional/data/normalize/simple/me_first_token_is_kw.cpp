template<typename T> struct TypedStruct {
    T val;
};

#define DECL_TYPED_STRUCT(x) TypedStruct<decltype(x)> x##name;
@kernel void simple_function(const float* inputArray @ restrict,
                             float* outputArray,
                             float value,
                             int size) {
    DECL_TYPED_STRUCT(value)
    for (int i = 0; i < size; ++i; @outer) {
        outputArray[i] = inputArray[i] + value;
    }
}
