// const array
const int arr_const0[12] = {0};
int const arr_const1[12] = {0};

// Stupid formatting
const int arr_const2[12] = {0};

// Deduced size
const float arr_const3[] = {1., 2., 3., 4., 5., 6.};

// Multidimensional
const float arr_const4[][2] = {{1., 2.}, {3., 4.}, {5., 6.}};
const float arr_const5[][3][2] = {{{1., 2.}, {3., 4.}, {5., 6.}}, {{1., 2.}, {3., 4.}, {5., 6.}}};

// At least one @kern function is requried
@kernel void kern() {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {
        }
    }
}
