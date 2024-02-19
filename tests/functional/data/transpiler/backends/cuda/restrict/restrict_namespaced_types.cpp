
namespace A {
template<class T>
struct Complex {
    T real;
    T imaginary;
};

namespace B {
struct Configs {
  unsigned int size1;
  unsigned long size2; 
};
namespace C {
typedef int SIZE_TYPE;
typedef SIZE_TYPE SIZES;
}
}
}

//TODO: fix me when @kernel/@outer/@inner are implemented
@kernel void function1(const A::Complex<float> *vectorData @restrict,
                       unsigned int vectorSize,
                       const A::Complex<float> **matricesData @restrict,
                       const A::B::Configs *matricesSizes @restrict)
{}

//TODO: fix me when @kernel/@outer/@inner are implemented
@kernel void function2(const A::Complex<float> *vectorData @restrict,
                       const A::B::Configs *configs @restrict,
                       A::B::C::SIZES *vectorSize @restrict)
{}
