const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
extern "C" void addVectors0(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= (4 * 1)) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= 1) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner non 1 increment
extern "C" void addVectors1(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= (4 * 2)) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= 2) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner unary post add
extern "C" void addVectors2(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= 4) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i--) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner unary pre add
extern "C" void addVectors3(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= 4) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); --i) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, check=True
extern "C" void addVectors4(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= (4 * 1)) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= 1) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, complex range
extern "C" void addVectors5(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = (entries + 16); _occa_tiled_i >= (entries - 12 + 4);
         _occa_tiled_i -= (4 * (entries / 16 + 1))) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= (entries / 16 + 1)) {
            if (i >= (entries - 12 + 4)) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, set dimension
extern "C" void addVectors6(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= (4 * 1)) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= 1) {
            if (i >= 0) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner ==> inner -> inner (nested)
extern "C" void addVectors7(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = entries - 1; _occa_tiled_i >= 0; _occa_tiled_i -= (4 * 1)) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= 1) {
            if (i >= 0) {
                for (int _occa_tiled_j = entries - 1; _occa_tiled_j >= 0;
                     _occa_tiled_j -= (4 * 1)) {
                    for (int j = _occa_tiled_j; j > (_occa_tiled_j - 4); j -= 1) {
                        if (j >= 0) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
extern "C" void addVectors8(const int& entries, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int _occa_tiled_i = (entries + 16); _occa_tiled_i >= (entries - 12 + static_cast<int>(*a));
         _occa_tiled_i -= (4 * (entries / 16 + 1))) {
        for (int i = _occa_tiled_i; i > (_occa_tiled_i - 4); i -= (entries / 16 + 1)) {
            if (i >= (entries - 12 + static_cast<int>(*a))) {
                for (unsigned long long int _occa_tiled_j = (entries + 16);
                     _occa_tiled_j >= (entries - 12 + static_cast<int>(*a));
                     _occa_tiled_j -= (4 * (entries / 16 + 1))) {
                    for (unsigned long long int j = _occa_tiled_j; j > (_occa_tiled_j - 4);
                         j -= (entries / 16 + 1)) {
                        if (j >= (entries - 12 + static_cast<int>(*a))) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}
