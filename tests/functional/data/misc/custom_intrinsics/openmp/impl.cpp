// INFO: needed for std::isnan
#include <cmath>

bool okl_is_nan(float value) {
  return std::isnan(value);
}
