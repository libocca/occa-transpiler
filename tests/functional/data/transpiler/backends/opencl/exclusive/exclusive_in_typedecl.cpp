typedef float ex_float32_t @exclusive;

@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        ex_float32_t d[32];
        @inner for (int j = 0; j < 32; ++j) {
            d[j] = i-j;
        }
    }
}
