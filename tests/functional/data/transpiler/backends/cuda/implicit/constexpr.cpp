constexpr float f = 13;

class HelloClass {
public:
    static constexpr int a = 2 + 2;
};

@kernel void test() {
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
        }
    }
}