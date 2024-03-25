

[[okl::kernel("")]] void test([[okl::restrict("")]] int* aaa
#if 1
                              ,
                              int bbb
#endif
) {
    [[okl::outer("")]] for (int i = 0; i < 10; ++i) {
        [[okl::inner("")]] for (int j = 0; j < 10; ++j) {
            // BODY
        }
    }
}
