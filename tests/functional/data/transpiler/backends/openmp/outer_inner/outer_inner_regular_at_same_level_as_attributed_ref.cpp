extern "C" void test_kernel() {
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    for (int i2 = 0; i2 < 10; ++i2) {
      for (int j = 0; j < 10; ++j) {
      }
      for (int ii = 0; ii < 10; ++ii) {
        for (int j = 0; j < 10; ++j) {
        }
        for (int j = 0; j < 10; ++j) {
        }
      }
    }
    for (int ii = 0; ii < 10; ++ii) {
      for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
        }
      }
    }
  }
  for (int ii = 0; ii < 10; ++ii) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
      for (int i2 = 0; i2 < 10; ++i2) {
        for (int i2 = 0; i2 < 10; ++i2) {
          for (int j = 0; j < 10; ++j) {
          }
        }
      }
    }
  }
}
