typedef struct {
    int vals[2][3][4];
} MyS;


@kernel void kern(MyS s) {
    for (int iz = 0; iz < 3; iz++; @outer) {
        for (int iy = 0; iy < 3; iy++; @inner) {

        }
    }
}
