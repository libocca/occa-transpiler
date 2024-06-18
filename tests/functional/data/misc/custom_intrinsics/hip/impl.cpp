// INFO: relate to the documentation is must be supported natively
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#single-precision-mathematical-functions

__device__ bool okl_is_nan(float value) {
  return isnan(value);
}

