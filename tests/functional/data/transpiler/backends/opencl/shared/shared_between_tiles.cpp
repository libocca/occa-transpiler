@kernel void test_kern() {
    @tile(4, @outer) for (int i = 0; i < 10; ++i) {
        @shared int shm[10];
        @tile(4, @inner, @inner) for (int j = 0; j < 10; ++j) {
            shm[j] = j;
        }
    }
}
