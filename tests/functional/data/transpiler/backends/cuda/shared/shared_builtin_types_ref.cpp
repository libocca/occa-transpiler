
// TODO: fix me when @kernel/@outer/@inner will be implementeds
[[okl::kernel("")]] void function1(const int* data) {
    __shared__ int arr1[32];
    __shared__ float arr2[8][32];
    __shared__ double arr3[4 + 4];
}
