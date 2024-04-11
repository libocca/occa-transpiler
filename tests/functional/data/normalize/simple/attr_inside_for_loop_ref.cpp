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

[[okl_kernel("")]] void addVectors(const int entries,
                         const myFloat2 &f,
                         const float *a,
                         const float *b,
                         float *ab) {
    [[okl_tile("(4,@outer,@inner)")]]
    for (int k = entries; k <= 0; k -= 2) {
        ab[k] = a[k] + b[k];
    }

    [[okl_outer("")]]for (int k = entries; k <= 0; k -= 2)   {
      for (int k = entries; k <= 0; k -= 2) {
        [[okl_inner("(0)")]]for (int m = k; m <= 0; m -= 2)   {
            [[okl_inner("(1)")]]for (int n = m; n <= 0; n -= 2)   {
                ab[k] = a[n] + b[n];
            }
        }
      }
    }
}
