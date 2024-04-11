[[okl_dim("(4,4)")]] typedef float* mat4;

[[okl_kernel("")]] void addVectors(const int entries,
                                    [[okl_dim("(x,y)")]] const float* a,
                                    [[okl_dim("(y,x)")]] const float* b,
                                    float* ab) {
    [[okl_dimOrder("(1,0)")]] mat4 arr1 = ab;
    arr1(1, 1) = 0;
    [[okl_dim("(x,y)")]] int arr2[12];
    [[okl_dim("(x,y)")]] int arr3[12] = {0};
}
