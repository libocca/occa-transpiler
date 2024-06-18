//INFO: transpiling to cuda backend already include necessary header

__device__ bool okl_is_nan(float value) {
  return isnan(value) != 0;
}
