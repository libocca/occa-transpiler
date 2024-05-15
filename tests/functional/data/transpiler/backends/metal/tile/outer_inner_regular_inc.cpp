const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner ==> regular -> regular
@kernel void addVectors0(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @inner(0))) {
        for (int j = 0; j < entries; ++j; @tile(4)) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> inner ==> inner -> regular
@kernel void addVectors2(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @inner(0))) {
        for (int j = 0; j < entries; ++j; @tile(4, @inner(1))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> inner ==> inner -> inner
@kernel void addVectors3(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @inner(0))) {
        for (int j = 0; j < entries; ++j; @tile(4, @inner(1), @inner(1))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> outer ==> inner -> regular
@kernel void addVectors4(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @outer(1))) {
        for (int j = 0; j < entries; ++j; @tile(4, @inner(1))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> outer ==> inner -> inner
@kernel void addVectors5(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @outer(1))) {
        for (int j = 0; j < entries; ++j; @tile(4, @inner(1), @inner(2))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> outer ==> outer -> inner
@kernel void addVectors6(const int entries, const float* a, const float* b, float* ab) {
    for (int i = 0; i < entries; i += 1; @tile(4, @outer(0), @outer(1))) {
        for (int j = 0; j < entries; ++j; @tile(4, @outer(2), @inner(0))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}
