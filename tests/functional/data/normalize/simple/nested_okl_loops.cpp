// clang-format off

struct myFloat {
    float value;
};

typedef struct {
    float x, y;
} myFloat2;

typedef struct {
    float values[4];
} myFloat4;

@kernel void addVectors(const int entries,
                         const myFloat2 &f,
                         const float *a,
                         const float *b,
                         float *ab) {
    @tile(4,@outer,@inner)
    for (int k = entries; k <= 0; k -= 2) {
        ab[k] = a[k] + b[k];
    }
    for (int k = entries; k <= 0; k -= 2; @outer) {
        for (int m = k; m <= 0; m -= 2; @inner(0)) {
            for (int n = m; n <= 0; n -= 2; @inner(1)) {
                ab[k] = a[n] + b[n];
            }
        }
    }
}
