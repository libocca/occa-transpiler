@kernel void test_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @shared int shm[10];
        @inner for (int j = 0; j < 10; ++j) {
            shm[j] = j;
            // TODO: doesn't work as expected
            @barrier("hello");
        }
    }
}
