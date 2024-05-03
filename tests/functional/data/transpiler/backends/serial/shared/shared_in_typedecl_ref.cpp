typedef float sh_float32_t;

extern "C" void test_kernel() {
  for (int i = 0; i < 32; ++i) {
    sh_float32_t b[32];
    for (int j = 0; j < 32; ++j) {
      b[j] = i + j;
    }
  }
}
