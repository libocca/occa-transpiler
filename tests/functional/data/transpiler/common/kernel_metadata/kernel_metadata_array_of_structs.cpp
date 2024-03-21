typedef struct {
    int vals_2;
} MyS1;

typedef struct {
    MyS1 vals[3];
} MyS;

@kernel void kern(MyS s) {
    for (int iz = 0; iz < 3; iz++; @outer) {
        for (int iy = 0; iy < 3; iy++; @inner) {

        }
    }
}
