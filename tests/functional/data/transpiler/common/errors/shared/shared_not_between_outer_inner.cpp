@kernel void test_kern() {
    @shared int shm[10];
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }
    }
}
