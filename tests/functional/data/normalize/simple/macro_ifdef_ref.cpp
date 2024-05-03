

#if 0
void use_some_type(UnknownType hello);
#endif

[[okl_kernel("")]] void kern() {
    [[okl_outer("")]] for (int i = 0; i < 10; i++) {
        [[okl_inner("")]] for (int j = 0; j < 10; j++) {}
    }
}
