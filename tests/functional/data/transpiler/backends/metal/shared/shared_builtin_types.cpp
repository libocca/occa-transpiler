@kernel void function1(const int* data) {
    @outer for (int i = 0; i < 64; ++i) {
        @shared int arr1[32];
        @shared float arr2[8][32];
        @shared double arr3[4 + 4];
        @inner for (int j = 0; j < 64; ++j) {
        }
    }
}

// syncronization between @inner loops:
@kernel void function2() {
    for (int i = 0; i < 10; i++; @outer) {
        @shared int shm[10];

        for (int j = 0; j < 10; j++; @inner) {
            shm[i] = j;
        }
        // sync should be here
        for (int j = 0; j < 10; j++; @inner) {
            shm[i] = j;
        }
        // sync should not be here
    }
}

// Even if loop is last, if it is inside regular loop, syncronization is inserted
@kernel void function3() {
    for (int i = 0; i < 10; i++; @outer) {
        @shared int shm[10];

        for (int q = 0; q < 5; ++q) {
            for (int j = 0; j < 10; j++; @inner) {
                shm[i] = j;
            }
            // sync should be here
        }
    }
}
