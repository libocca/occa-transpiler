@kernel void hello_kern() {
    for (int i = 0; i < 10; ++i; @tile(2, @outer, @inner)) {
    }
}
