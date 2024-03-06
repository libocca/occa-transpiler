// TODO: After multiple @dimOrder are fixed, generate ref and test entry
typedef float* mat89_f @dim(8, 9);
typedef int* mat89_i @dim(8, 9);

// dimOrder inside function argument
@kernel void test_kernel_0(const int entries,
                           float* a,
                           float* b,
                           float* ab,
                           mat89_f mat @dimOrder(1, 0)) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, j);
        }
    }
}

@kernel void test_kernel_1(const int entries,
                           float* a,
                           float* b,
                           float* ab,
                           mat89_f mat @dimOrder(0, 1)) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, j);
        }
    }
}

typedef float* mat98_f @dim(8, 9) @dimOrder(1, 0);

// typeDefs
@kernel void test_kernel_2(const int entries, float* a, float* b, float* ab, mat98_f mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, mat(j, i));
        }
    }
}

// variable declaration
@kernel void test_kernel_3(const int entries, float* a, float* b, float* ab, mat98_f mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, mat(j, i));
            typedef int* mat23 @dim(j, 3);
            mat23 xy;
            mat23 yx @dimOrder(1, 0);
            yx(1, 2) = 0;
            xy(1, 2) = yx(1, 2);
        }
    }
}
