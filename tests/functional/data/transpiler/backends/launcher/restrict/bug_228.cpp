@ restrict int* okl(int a, int b) {
    static int c = a + b;
    return &c;
}

@kernel void add_kernel(const int n,
                        const int @ restrict* a,
                        const int @ restrict* b,
                        int @ restrict* c) {
    @tile(16, @outer, @inner) for (int i = 0; i < n; i++) {
        @ restrict int* tmp;
        c[i] = *okl(a[i], b[i]) + tmp[i];
    }
}
