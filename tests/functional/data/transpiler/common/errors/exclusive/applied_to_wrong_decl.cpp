@exclusive int get0() {
    return 0;
}

@kernel void test_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @exclusive int kk = 0;
        @inner for (int j = 0; j < 10; ++j) {
            kk = i + j;
        }
    }
}
