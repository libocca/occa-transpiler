static float add1(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}

float add2(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}

// At least one @kern function is requried
@kernel void kern () {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {}
    }
}
