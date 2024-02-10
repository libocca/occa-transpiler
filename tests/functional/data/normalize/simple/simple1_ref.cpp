[[okl::kernel("")]] void simple_function([[okl::restrict("")]] const float* inputArray,
                                               float* outputArray,
                                               float value,
                                               int size) {
  [[okl::outer("")]] for (int i = 0; i < size; ++i) {
    outputArray[i] = inputArray[i] + value;
  }
}
