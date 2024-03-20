
template <class T>
struct Complex {
    T real;
    T imaginary;
};

struct Configs {
    unsigned int size1;
    unsigned long size2;
};


extern "C" void function1(const Complex<float>* __restrict__ vectorData,
                          unsigned int& vectorSize,
                          const Complex<float>** __restrict__ matricesData,
                          const Configs* __restrict__ matricesSizes) {}
