struct Coord {
    float x;
    float y;
};
typedef float* mat89_f @dim(8, 9);
typedef int* mat89_i @dim(8, 9);
typedef Coord* mat89_s @dim(8, 9);
typedef Coord* mat8_s @dim(8);

// float dim
@kernel void test_kernel_0(const int entries, float* a, float* b, float* ab, mat89_f mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, j);
        }
    }
}

// int dim
@kernel void test_kernel_1(const int entries, float* a, float* b, float* ab, mat89_i mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + static_cast<float>(mat(i, j));
        }
    }
}

// struct dim
@kernel void test_kernel_2(const int entries, float* a, float* b, float* ab, mat89_s mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, j).x;
        }
    }
}

// struct + single dim
@kernel void test_kernel_3(const int entries, float* a, float* b, float* ab, mat8_s mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i).x + mat(j).y;
        }
    }
}

// inside attributed loop
@kernel void test_kernel_4(const int entries, float* a, float* b, float* ab, mat89_s mat) {
    for (int i = mat(7, 7); i < entries; i += mat(1, 1); @outer(0)) {
        for (int j = mat(0, 0); j < entries; j += 1; @inner(0)) {
            ab[i] = a[i] + b[j] + mat(i, j).x + mat(j, i).y;
        }
    }
}

// TODO: doesnt work if dim is in condition right now
// @kernel void test_kernel_4_2(const int entries, float* a, float* b, float* ab, mat89_s mat) {
//     for (int i = mat(7, 7); i < mat(0, 0); i += mat(1, 1); @outer(0)) {
//         for (int j = mat(0, 0); j < entries; j += 1; @inner(0)) {
//             ab[i] = a[i] + b[j] + mat(i, j).x + mat(j, i).y;
//         }
//     }
// }

// assignment, comparison, etc.
@kernel void test_kernel_5(const int entries, float* a, float* b, float* ab, mat89_s mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            if (mat(i, j).x <= 0) {
                mat(i, j).x = a[i] + b[j] + mat(i, j).x + mat(j, i).y;
            }
        }
    }
}

// nested mat
@kernel void test_kernel_6(const int entries, float* a, float* b, float* ab, mat89_s mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            mat(i, j) = a[i] + b[j] + mat(i, mat(j, 0));
        }
    }
}

float get1() {
    return 1;
}

// nested mat + complex expressions inside dim
@kernel void test_kernel_7(const int entries, float* a, float* b, float* ab, mat89_s mat) {
    for (int i = 0; i < entries; i += 1; @outer(0)) {
        for (int j = 0; j < entries; j += 1; @inner(0)) {
            mat(i, j + get1() + (i * j / get1())) = a[i] + b[j] + mat(i + 12, mat(j, get1()));
        }
    }
}

void many_dims(const int* B @dim(10, 10)) {
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
    B(1, 1) = 10;
}
