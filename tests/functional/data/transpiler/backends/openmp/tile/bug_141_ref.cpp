extern "C" void hello_kern() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 2) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 2); ++i) {
            if (i < 10) {
            }
        }
    }
}
