const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
[[okl::kernel("")]] void addVectors0(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int j = entries - 1; j >= 0; j -= 1) {
        [[okl::inner("")]] for (int i = entries - 1; i >= 0; i -= 1) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner non 1 increment
[[okl::kernel("")]] void addVectors1(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int j = entries - 1; j >= 0; j -= 2) {
        [[okl::inner("")]] for (int i = entries - 1; i >= 0; i -= 2) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary post add
[[okl::kernel("")]] void addVectors2(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int j = entries - 1; j >= 0; j--) {
        [[okl::inner("")]] for (int i = entries; i > 0; i--) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary pre add
[[okl::kernel("")]] void addVectors3(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int j = entries - 1; j >= 0; --j) {
        [[okl::inner("")]] for (int i = entries - 1; i >= 0; --i) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
[[okl::kernel("")]] void addVectors4(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int i = entries - 1; i >= 0; --i) {
        [[okl::outer("")]] for (int j = entries - 1; j >= 0; --j) {
            [[okl::inner("")]] for (int k = entries - 1; k >= 0; --k) {
                [[okl::inner("")]] for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
[[okl::kernel("")]] void addVectors5(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("(1)")]] for (int i = entries - 1; i >= 0; --i) {
        [[okl::outer("(0)")]] for (int j = entries - 1; j >= 0; --j) {
            [[okl::inner("(1)")]] for (int k = entries - 1; k >= 0; --k) {
                [[okl::inner("(0)")]] for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
[[okl::kernel("")]] void addVectors6(const int entries, const float* a, const float* b, float* ab) {
    [[okl::outer("")]] for (int i = entries - 1; i >= 0; --i) {
        [[okl::outer("(0)")]] for (int j = entries - 1; j >= 0; --j) {
            [[okl::inner("")]] for (int k = entries - 1; k >= 0; --k) {
                [[okl::inner("(0)")]] for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}
