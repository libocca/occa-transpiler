
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
    float* __restrict__ x;
    float* __restrict__ y;
    unsigned long size;
};


extern "C" void function1(const Complex<float>* __restrict__ vectorData,
                          unsigned int& vectorSize,
                          const Complex<float>** __restrict__ matricesData,
                          const Configs* __restrict__ matricesSizes) {}
