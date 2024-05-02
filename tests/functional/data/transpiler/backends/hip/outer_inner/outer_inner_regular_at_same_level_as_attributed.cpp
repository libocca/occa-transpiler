@kernel void test_kernel() {
    @outer for (int i = 0; i < 10; ++i) {
        @outer for (int i2 = 0; i2 < 10; ++i2) {
            @inner for (int j = 0; j < 10; ++j) {
            }

            for (int ii = 0; ii < 10; ++ii) {
                @inner for (int j = 0; j < 10; ++j) {
                }
                for (int j = 0; j < 10; ++j) {
                }
            }
        }

        for (int ii = 0; ii < 10; ++ii) {
            @outer for (int i = 0; i < 10; ++i) {
                @inner for (int j = 0; j < 10; ++j) {
                }
            }
        }
    }
    for (int ii = 0; ii < 10; ++ii) {
        @outer for (int i = 0; i < 10; ++i) {
            for (int i2 = 0; i2 < 10; ++i2) {
                @outer for (int i2 = 0; i2 < 10; ++i2) {
                    @inner for (int j = 0; j < 10; ++j) {
                    }
                }
            }
        }
    }
}
