
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
[[okl::kernel("")]] void function1(const A::Complex<float> * __restrict__ vectorData,
                       unsigned int vectorSize,
                       const A::Complex<float> ** __restrict__ matricesData,
                       const A::B::Configs * __restrict__ matricesSizes)
{}

//TODO: fix me when @kernel/@outer/@inner are implemented
[[okl::kernel("")]] void function2(const A::Complex<float> * __restrict__ vectorData,
                       const A::B::Configs * __restrict__ configs,
                       A::B::C::SIZES * __restrict__ vectorSize)
{}
