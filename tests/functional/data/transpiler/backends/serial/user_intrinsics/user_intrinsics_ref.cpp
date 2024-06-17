//INFO: needed for std::isnan
#include <cmath>

bool okl_is_nan(float value) {
  return std::isnan(value);
}

extern "C" void zero_nans(float *vec) {
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      int idx = i * 32 + j;
      float value = vec[idx];
      if (okl_is_nan(value)) {
        vec[idx] = 0.0f;
      }
    }
  }
}

