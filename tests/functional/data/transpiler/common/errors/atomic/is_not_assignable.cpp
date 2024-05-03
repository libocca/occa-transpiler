@kernel void hello_kern() {
    @outer for (int i = 0; i < 10; ++i) {
        @inner for (int j = 0; j < 10; ++j) {
            @atomic 12 *= 12;
        }
    }
}
