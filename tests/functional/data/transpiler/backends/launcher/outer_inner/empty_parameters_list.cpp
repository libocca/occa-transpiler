@kernel void test() {
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
            int a = 0;
        }
    }
}
