template<int aa, int bb>
class HelloClass;

template<int bb>
class HelloClass<0, bb> {
public:
    static inline void myfn() { }
};

template<int bb>
class HelloClassFull {
public:
    inline void myfn() { }
};


template<>
class HelloClassFull<0> {
public:
    inline void myfn() { }
};

@kernel void reductionWithSharedMemory(const int entries, const float* vec) {
    @tile(16, @outer, @inner) for (int i = 0; i < entries; ++i) {
        auto tmp = vec[i];
    }
}
