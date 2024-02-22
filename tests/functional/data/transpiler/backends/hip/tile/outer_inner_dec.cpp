const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
@kernel void addVectors0(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i -= 1; @tile(4, @outer, @inner)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner non 1 increment
@kernel void addVectors1(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i -= 2; @tile(4, @outer, @inner)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner unary post add
@kernel void addVectors2(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i--; @tile(4, @outer, @inner)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner unary pre add
@kernel void addVectors3(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; --i; @tile(4, @outer, @inner)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner, check=True
@kernel void addVectors4(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i -= 1; @tile(4, @outer, @inner, check = true)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner, complex range
@kernel void addVectors5(const int entries, const float* a, const float* b, float* ab) {
    for (int i = (entries + 16); i >= (entries - 12 + 4); i -= (entries / 16 + 1); @tile(4, @outer, @inner)) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner, set dimension
@kernel void addVectors6(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i -= 1; @tile(4, @outer(1), @inner(2))) {
        ab[i] = add(a[i], b[i]);
    }
}

// Outer -> inner ==> inner -> inner (nested)
@kernel void addVectors7(const int entries, const float* a, const float* b, float* ab) {
    for (int i = entries - 1; i >= 0; i -= 1; @tile(4, @outer(0), @inner(0))) {
        for (int j = entries - 1; j >= 0 ; j -= 1; @tile(4, @inner(1), @inner(2))) {
            ab[i] = add(a[i], b[j]);
        }
    }
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
@kernel void addVectors8(const int entries, const float* a, const float* b, float* ab) {
    for (int i = (entries + 16); i >= (entries - 12 + static_cast<int>(*a)); i -= (entries / 16 + 1); @tile(4, @outer(0), @inner(0), check=true)) {
        for (unsigned long long j = (entries + 16); j >= (entries - 12 + static_cast<int>(*a)); j -= (entries / 16 + 1); @tile(4, @inner(1), @inner(2), check=true)) {
            ab[i] = add(a[i], b[j]);
        }
    }
}
