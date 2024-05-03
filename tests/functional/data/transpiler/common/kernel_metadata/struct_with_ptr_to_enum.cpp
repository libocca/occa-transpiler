typedef enum {
    El1,
    El2,
    El3
} MyEnum;


typedef struct {
    MyEnum* my_elems;
} MyStruct;


@kernel void kern(MyStruct s) {
    for (int i = 0; i < 3; i++; @outer) {
        for (int j = 0; j < 3; j++; @inner) {

        }
    }
}
