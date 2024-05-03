@kernel void test_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @shared int shm[10];
        @inner for (int j = 0; j < 10; ++j) {
            @shared for (int k = 0; k < 10; ++k) {
            }
            shm[j] = j;
        }
    }
}
