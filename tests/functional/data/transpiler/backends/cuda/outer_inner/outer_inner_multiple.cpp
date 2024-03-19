const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// with shared memory usage (should be automatic sync)
@kernel void addVectors(const int entries, float* a, float* b, float* ab, float* mat @dim(10, 10)) {
    for (int i = 0; i < entries; i += 1; @outer) {
        for (int i2 = 0; i2 < entries; i2 += 1; @outer) {
            @shared int shm[32];
            @shared int shm2[32];
            for (int j = 0; j < entries; j += 1; @inner) {
                shm[j] = 0;  // shared memory usage -> should be barrier after @inner loop
                mat(0, 0) = 12;
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }

                    ab[i] = add(a[i], b[k]);
                }
            }

            for (int j = 0; j < entries; j += 1; @tile(4, @inner, @inner, check = false)) {
                for (int k = 0; k < entries; k += 1; @inner) {
                    // shared memory usage -> should be barrier, since @tile is inner, inner
                    shm[j] = 0;
                }
            }

            for (int j = 0; j < entries; j += 1; @inner) {
                shm[j] = 0;
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }

                    ab[i] = add(a[i], b[k]);
                }

                for (int k = 0; k < entries; k += 1; @tile(4, @inner, @inner, check = false)) {
                    ab[i] = add(a[i], b[k]);
                }
            }
        }
    }
}

// without shared memory usage (should be no automatic sync)
@kernel void addVectors1(const int entries,
                         float* a,
                         float* b,
                         float* ab,
                         float* mat @dim(10, 10)) {
    for (int i = 0; i < entries; i += 1; @outer) {
        for (int i2 = 0; i2 < entries; i2 += 1; @outer) {
            @shared int shm[32];
            @shared int shm2[32];
            for (int j = 0; j < entries; j += 1; @inner) {
                // shm[j] = 0;  // shared memory usage -> should be barrier after @inner loop
                mat(0, 0) = 12;
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }

                    ab[i] = add(a[i], b[k]);
                }
            }

            for (int j = 0; j < entries; j += 1; @tile(4, @inner, @inner, check = false)) {
                for (int k = 0; k < entries; k += 1; @inner) {
                    // shared memory usage -> should be barrier, since @tile is inner, inner
                    // shm[j] = 0;
                }
            }

            for (int j = 0; j < entries; j += 1; @inner) {
                shm[j] = 0;
                for (int k = 0; k < entries; k += 1; @inner) {
                    for (int ii = 0; ii < entries; ii += 1; @inner) {
                        ab[i] = add(a[i], b[k]);
                    }

                    ab[i] = add(a[i], b[k]);
                }

                for (int k = 0; k < entries; k += 1; @tile(4, @inner, @inner, check = false)) {
                    ab[i] = add(a[i], b[k]);
                }
            }
        }
    }
}
