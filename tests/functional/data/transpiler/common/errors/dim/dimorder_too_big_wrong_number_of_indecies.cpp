typedef float* mat89_f @dim(8, 9);

@kernel void hello_kern(mat89_f mat @dimOrder(1)) {
    @outer for (int i = 0; i < 10; ++i) {
        @inner for (int j = 0; j < 10; ++j) {
            mat(1, 2) = i + j;
        }
    }
}
