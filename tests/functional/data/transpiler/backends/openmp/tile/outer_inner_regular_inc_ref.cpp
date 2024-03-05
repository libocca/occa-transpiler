const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner ==> regular -> regular
extern "C" void addVectors0(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> regular
extern "C" void addVectors2(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> inner
extern "C" void addVectors3(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> regular
extern "C" void addVectors4(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> inner
extern "C" void addVectors5(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> outer -> inner
extern "C" void addVectors6(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = 0; _occa_tiled_i < entries; _occa_tiled_i += (4 * 1)) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); i += 1) {
            for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += 4) {
                if (i < entries) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}
