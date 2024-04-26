@kernel void test0() {
    @outer for (int x = 0; x < 10; ++x; @simd_length(16)) {
        @inner for (int y = 0; y < 10; ++y) {
            int z = x + y;
        }
    }
}

@kernel void test1() {
    @tile(5, @outer) for (int x = 0; x < 10; ++x; @simd_length(16)) {
        @inner for (int y = 0; y < 10; ++y) {
            int z = x + y;
        }
    }
}

@kernel void test2() {
    @outer for (int x = 0; x < 10; ++x; @simd_length(16)) {
        @inner for (int y = 0; y < 10; ++y) {
            int z = x + y;
        }
    }
    @outer for (int x = 0; x < 10; ++x; @simd_length(16)) {
        @inner for (int y = 0; y < 10; ++y) {
            int z = x + y;
        }
    }
}
