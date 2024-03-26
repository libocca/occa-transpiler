const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
@kernel void addVectors0(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = entries - 1; j >= 0; j -= 1) {
        @inner for (int i = entries - 1; i >= 0; i -= 1) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner non 1 increment
@kernel void addVectors1(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = entries - 1; j >= 0; j -= 2) {
        @inner for (int i = entries - 1; i >= 0; i -= 2) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary post add
@kernel void addVectors2(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = entries - 1; j >= 0; j--) {
        @inner for (int i = entries; i > 0; i--) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> inner unary pre add
@kernel void addVectors3(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int j = entries - 1; j >= 0; --j) {
        @inner for (int i = entries - 1; i >= 0; --i) {
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
@kernel void addVectors4(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int i = entries - 1; i >= 0; --i) {
        @outer for (int j = entries - 1; j >= 0; --j) {
            @inner for (int k = entries - 1; k >= 0; --k) {
                @inner for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
@kernel void addVectors5(const int entries, const float* a, const float* b, float* ab) {
    @outer(1) for (int i = entries - 1; i >= 0; --i) {
        @outer(0) for (int j = entries - 1; j >= 0; --j) {
            @inner(1) for (int k = entries - 1; k >= 0; --k) {
                @inner(0) for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
@kernel void addVectors6(const int entries, const float* a, const float* b, float* ab) {
    @outer for (int i = entries - 1; i >= 0; --i) {
        @outer(0) for (int j = entries - 1; j >= 0; --j) {
            @inner for (int k = entries - 1; k >= 0; --k) {
                @inner(0) for (int ii = entries - 1; ii >= 0; --ii) {
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}
