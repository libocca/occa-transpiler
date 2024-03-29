extern "C" void hello_kern() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        int shm[10];
        for (int j = 0; j < 10; ++j) {
            shm[j] = j;
        }
        for (int j = 0; j < 10; ++j) {
            shm[j] = j;
        }
        for (int j = 0; j < 10; ++j) {
            shm[j] = j;
        }
        for (int j = 0; j < 10; ++j) {
            shm[j] = j;
        }
    }
}
