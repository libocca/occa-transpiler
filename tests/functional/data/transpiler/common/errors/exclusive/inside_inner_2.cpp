// Cannot define [@exclusive] variables inside an [@inner] loop
@kernel void hello_kern() {
    @tile(4, @outer, @inner) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}
