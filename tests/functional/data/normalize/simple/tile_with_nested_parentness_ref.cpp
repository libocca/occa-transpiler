// clang format off

[[okl::kernel("")]] void runDiff([[okl::restrict("")]] int* a, [[okl::restrict("")]] int* b) {
    [[okl::tile("(16,@inner(0),@outer(1),check=true)")]] for (int i = 0; i < 100; ++i) {
        b[i] = a[i] - 100;
    }
}
