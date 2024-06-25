typedef float sh_float32_t @shared;

@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        sh_float32_t b[32];
        @inner for (int j = 0; j < 32; ++j) {
            b[j] = i+j;
        }
    }
}
