struct S {
    int hello[12];
};
extern const int arr_0[];
extern const float arr_1[];
extern const S arr_2[];

// At least one @kern function is required
@kernel void kern() {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {
        }
    }
}
