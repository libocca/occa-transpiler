@kernel void hello_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @inner for (int j = 0; j < 10; ++j) {
            @dimOrder(1, 0) for (int k = 0; k < 10; ++k) {

            }
        }
    }
}
