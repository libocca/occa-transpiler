const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
@kernel void addVectors0(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = 0; j < entries; j += 1) {
        @inner for (int i = 0; i < entries; i += 1) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner non 1 increment
@kernel void addVectors1(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = 0; j < entries; j += 2) {
        @inner for (int i = 0; i < entries; i += 2) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary post add
@kernel void addVectors2(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = 0; j < entries; j++) {
        @inner for (int i = 0; i <= entries - 1; i++) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary pre add
@kernel void addVectors3(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = 0; j < entries; ++j) {
        @inner for (int i = 0; i < entries; ++i) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
@kernel void addVectors4(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int i = 0; i < entries; ++i) {
        @outer for (int j = 0; j < entries; ++j) {
            @inner for (int k = 0; k < entries; ++k) {
                @inner for (int ii = 0; ii < entries; ++ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
@kernel void addVectors5(const int entries, const float* a, const float* b, float* ab) {
    @outer(1) for (int i = 0; i < entries; ++i) {
        @outer(0) for (int j = 0; j < entries; ++j) {
            @inner(1) for (int k = 0; k < entries; ++k) {
                @inner(0) for (int ii = 0; ii < entries; ++ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
@kernel void addVectors6(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int i = 0; i < entries; ++i) {
        @outer(0) for (int j = 0; j < entries; ++j) {
            @inner for (int k = 0; k < entries; ++k) {
                @inner(0) for (int ii = 0; ii < entries; ++ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}
