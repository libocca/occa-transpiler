
template <class T>
struct Complex {
    T real;
    T imaginary;
};

struct Configs {
    unsigned int size1;
    unsigned long size2;
};

struct Data {
    @restrict float* x;
    @restrict float* y;
    unsigned long size;
};


@kernel void function1(const Complex<float>* vectorData @restrict,
                       unsigned int vectorSize,
                       const Complex<float>** matricesData @restrict,
                       const Configs* matricesSizes @restrict) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
        }
    }
}
