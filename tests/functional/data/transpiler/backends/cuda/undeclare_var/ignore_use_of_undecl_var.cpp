@kernel void hello_kern() {
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
            float var = 10.0;
            float res = __exp10f(var);
            auto ok = std::isnan(var);
        }
    }
}
