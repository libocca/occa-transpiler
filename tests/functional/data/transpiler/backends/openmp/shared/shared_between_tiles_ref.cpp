extern "C" void test_kern() {
#pragma omp parallel for
  for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
    for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
      if (i < 10) {
        int shm[10];
        for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
          for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
            if (j < 10) {
              shm[j] = j;
            }
          }
        }
      }
    }
  }
}
