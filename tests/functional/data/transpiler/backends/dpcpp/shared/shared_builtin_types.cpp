
// TODO: fix me when @kernel/@outer/@inner will be implementeds
@kernel void function1(const int* data) {
    @shared int arr1[32];
    @shared float arr2[8][32];
    @shared double arr3[4 + 4];
}
