@ restrict float* myfn(float* a) {
    return a + 1;
}

float* myfn2(float* a) {
    return a + 1;
}

@kernel void hello() {
    for (int i = 0; i < 10; i++; @outer) {
        for (int j = 0; j < 10; j++; @inner) {
        }
    }
}
