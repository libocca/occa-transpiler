

@kernel void function1(const int* i32Data @ restrict,
                       float* fp32Data @ restrict,
                       const double* fp64Data @ restrict) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            @ restrict float* b = &fp32Data[0];
        }
    }
}
