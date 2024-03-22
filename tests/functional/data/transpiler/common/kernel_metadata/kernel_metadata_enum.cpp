enum class MyEn {
    El1=0,
    El2=2
};

typedef struct {
    MyEn val;
} MyS;

typedef struct {
    MyEn val[2][3];
} MySArr;

@kernel void kern(MyS s, MySArr sArr) {
    for (int iz = 0; iz < 3; iz++; @outer) {
        for (int iy = 0; iy < 3; iy++; @inner) {

        }
    }
}
