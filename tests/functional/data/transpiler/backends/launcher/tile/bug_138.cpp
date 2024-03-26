@kernel void hello_kern() {
    for (int i = 0; i < 1; i++; @tile(1, @outer, @inner)) {
    }
}
