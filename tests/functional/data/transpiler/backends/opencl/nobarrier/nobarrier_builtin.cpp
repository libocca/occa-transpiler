@kernel void hello_kern() {
    for (int i = 0; i < 10; ++i; @outer) {
        @shared int shm[10];
        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        @nobarrier for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }
    }
}

@kernel void priority_issue() {
    @outer for (int i = 0; i < 32; ++i) {
        @shared float shm[32];
        @nobarrier for (int j = 0; j < 32; ++j; @inner) {
                shm[i] = i;
        }
        @inner for (int j = 0; j < 32; ++j) {
                @atomic shm[i * j] += 32;
        }
    }
}
