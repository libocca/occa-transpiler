@directive("#define MACRO_1 1") @directive("#define __restrict_macro @restrict")

    @kernel void test(__restrict_macro int* aaa
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
