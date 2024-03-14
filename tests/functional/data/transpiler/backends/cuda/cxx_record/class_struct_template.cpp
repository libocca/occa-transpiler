template <typename T>
struct ComplexType {
    T v1;
    T v2;
    T calc();
    ComplexType(T in)
        : v1(in),
          v2(in) {}
};

struct ComplexTypeFloat {
    float v1;
    float v2;
    float calc();
};

@kernel void reductionWithSharedMemory(const int entries, const float* vec) {
    @tile(16, @outer, @inner) for (int i = 0; i < entries; ++i) {
        auto tmp = vec[i];
    }
}
