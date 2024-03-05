[[okl::kernel("")]] void addVectors(const int entries,
                                    const float* a,  // @dim(16),
                                    const float* b,
                                    float* ab) {
    [[okl::outer("")]] for (int k = entries; k > 0; k -= 2) {
        [[okl::outer("(1)")]] for (int g = 0; g < entries; ++g) {
            [[okl::exclusive("")]] int arr1 = 0;
            [[okl::inner("(0)")]] for (int m = 0; m < 100; ++m) {
                [[okl::inner("(1)")]] for (int n = 0; n < 50; ++n) { ab[k] = a[n] + b[n]; }
            }
        }
    }
}
