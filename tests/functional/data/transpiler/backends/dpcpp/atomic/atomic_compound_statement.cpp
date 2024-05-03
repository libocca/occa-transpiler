@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @shared float shm[32];
        @inner for (int j = 0; j < 32; ++j) {
            @atomic {
                shm[i*j]++;
                j += 32;
            }
        }
    }
}