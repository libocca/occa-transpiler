
template<class T>
struct Complex {
    T real;
    T imaginary;
};

struct Configs {
  unsigned int size1;
  unsigned long size2; 
};

//TODO: fix me when @kernel/@outer/@inner are implemented
@kernel void function1(const Complex<float> *vectorData @restrict,
                       unsigned int vectorSize,
                       const Complex<float> **matricesData @restrict,
                       const Configs *matricesSizes @restrict)
{}

