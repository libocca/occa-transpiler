typedef int* mat89 @dim;

@kernel void hello_kern(mat89 mat) {
    @outer for (int i = 0; i < 10; ++i) {
        @inner for (int j = 0; j < 10; ++j) {
            mat(i, j) = 12;
        }
    }
}
