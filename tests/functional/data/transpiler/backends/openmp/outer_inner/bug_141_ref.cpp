extern "C" void hello_kern() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
        }
    }
}
