const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
extern "C" void addVectors0(const int& entries, const float* a, const float* b, float* ab) {
    for (int j = entries - 1; j >= 0; j -= 1) {
        for (int i = entries - 1; i >= 0; i -= 1) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner non 1 increment
extern "C" void addVectors1(const int& entries, const float* a, const float* b, float* ab) {
    for (int j = entries - 1; j >= 0; j -= 2) {
        for (int i = entries - 1; i >= 0; i -= 2) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary post add
extern "C" void addVectors2(const int& entries, const float* a, const float* b, float* ab) {
    for (int j = entries - 1; j >= 0; j--) {
        for (int i = entries; i > 0; i--) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary pre add
extern "C" void addVectors3(const int& entries, const float* a, const float* b, float* ab) {
    for (int j = entries - 1; j >= 0; --j) {
        for (int i = entries - 1; i >= 0; --i) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
extern "C" void addVectors4(const int& entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; --i) {
        for (int j = entries - 1; j >= 0; --j) {
            for (int k = entries - 1; k >= 0; --k) {
                for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
extern "C" void addVectors5(const int& entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; --i) {
        for (int j = entries - 1; j >= 0; --j) {
            for (int k = entries - 1; k >= 0; --k) {
                for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
extern "C" void addVectors6(const int& entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; --i) {
        for (int j = entries - 1; j >= 0; --j) {
            for (int k = entries - 1; k >= 0; --k) {
                for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}
