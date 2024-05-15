
namespace A {
template <class T>
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
}  // namespace C
}  // namespace B
}  // namespace A

@kernel void function1(const A::Complex<float>* vectorData @ restrict,
                       unsigned int vectorSize,
                       const A::Complex<float>** matricesData @ restrict,
                       const A::B::Configs* matricesSizes @ restrict) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
        }
    }
}

@kernel void function2(const A::Complex<float>* vectorData @ restrict,
                       const A::B::Configs* configs @ restrict,
                       A::B::C::SIZES* vectorSize @ restrict) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
        }
    }
}
