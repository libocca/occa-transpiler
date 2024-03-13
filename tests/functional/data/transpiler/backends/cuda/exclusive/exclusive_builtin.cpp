static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

// TODO: fix preprocessor handling and try with define
// #define BLOCK_SIZE 4
const int BLOCK_SIZE = 4;

@kernel void addVectors(const int N, const float* a, const float* b, float* ab) {
    @outer for (int i = 0; i < N; i += BLOCK_SIZE) {
        @shared float s_b[BLOCK_SIZE];
        @exclusive const float* g_a = a;
        @inner for (int j = 0; j < BLOCK_SIZE; ++j) {
            s_b[j] = b[i + j];
            @barrier;

            ab[i + j] = add(g_a, i + j, s_b, j);
        }
    }
}
