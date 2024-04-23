@kernel void test_kern() {
    for (int i = 0; i < 10; ++i; @outer) {
        @restrict for (int j = 0; j < 10; ++j; @inner) {
        }
    }
}
