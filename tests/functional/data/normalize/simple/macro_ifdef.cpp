#define MYMACRO 1

#ifndef MYMACRO
void use_some_type(UnknownType hello);
#endif



@kernel void kern() {
    for (int i = 0; i < 10; i++; @outer) {
        for (int j = 0; j < 10; j++; @inner) {

        }
    }
}
