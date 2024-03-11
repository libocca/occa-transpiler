static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

#define BLOCK_SIZE 4

extern "C" void addVectors0(const int& N, const float* a, const float* b, float* ab) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        float s_b[BLOCK_SIZE];
        const float* g_a = a;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            s_b[j] = b[i + j];
            ;

            ab[i + j] = add(g_a, i + j, s_b, j);
        }
    }
}
