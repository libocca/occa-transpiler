@kernel void hello_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @outer for (int j = 0; j < 10; ++j) {
            @inner for (int k = 0; k < 10; ++k) {
                @inner for (int kk = 0; kk < 10; ++kk) {
                }
                @inner for (int kk = 0; kk < 10; ++kk) {
                }
            }
            @inner for (int k = 0; k < 10; ++k) {
                // Commented loop is required for correct structure
                @inner for (int kk = 0; kk < 10; ++kk) {
                }
            }
        }

        // First tile must be @outer for correcst structure
        @tile(4, @inner, @inner, check = false) for (int j = 0; j < 10; ++j) {
            @inner for (int kk = 0; kk < 10; ++kk) {
            }
        }
    }
}
