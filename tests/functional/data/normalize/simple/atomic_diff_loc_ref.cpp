[[okl_kernel("")]] void f(float a) {
    [[okl_atomic("")]] a = a + 1;
    [[okl_atomic("")]] a += 1;

    {
        float b;
        [[okl_atomic("")]] b = a + b;
    }

    int c = 0;
    [[okl_atomic("")]] {
        a *= 1;
        c += a;
    }
}
