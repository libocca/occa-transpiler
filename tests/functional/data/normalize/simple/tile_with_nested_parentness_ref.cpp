// clang format off

[[okl_kernel("")]] void runDiff([[okl_restrict("")]] int* a, [[okl_restrict("")]] int* b) {
    [[okl_tile("(16,@inner(0),@outer(1),check=true)")]] for (int i = 0; i < 100; ++i) {
        b[i] = a[i] - 100;
    }
}
