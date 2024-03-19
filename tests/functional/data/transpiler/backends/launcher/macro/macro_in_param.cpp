#define MACRO_1 1

@kernel void mykern(int aaa
#if MACRO_1
                    ,
                    int bbb
#endif
) {
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
            // BODY
        }
    }
}
