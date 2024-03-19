extern "C" void mykern(int& aaa, int& bbb) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            // BODY
        }
    }
}
