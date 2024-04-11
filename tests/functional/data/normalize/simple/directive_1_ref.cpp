

[[okl_kernel("")]] void test([[okl_restrict("")]] int* aaa
#if 1
                              ,
                              int bbb
#endif
) {
    [[okl_outer("")]] for (int i = 0; i < 10; ++i) {
        [[okl_inner("")]] for (int j = 0; j < 10; ++j) {
            // BODY
        }
    }
}
