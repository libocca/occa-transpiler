

#if 0
void use_some_type(UnknownType hello);
#endif

[[okl::kernel("")]] void kern() {

    [[okl::outer("")]] for (int i = 0; i < 10; i++) {
        [[okl::inner("")]] for (int j = 0; j < 10; j++) {}
    }
}
