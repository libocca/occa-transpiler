@kernel void simple_function(const float* inputArray @ restrict,
                             float* outputArray,
                             float value,
                             int size) {
  for (int i = 0; i < size; ++i; @outer) {
    outputArray[i] = inputArray[i] + value;
  }
}
