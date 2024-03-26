
@kernel void test0(const int entries, const float* a, const float* b, float* ab) {
    int before0 = 0;
    @outer for (int x = 0; x < 10; ++x) {
        // int before1 = 1 + before0;
        int before1 = 1;
        @outer for (int y = 0; y < 20; ++y) {
            int before2 = 1 + before1;
            @outer for (int z = 0; z < 30; ++z) {
                int before3 = 1 + before2;
                @inner for (int n = 0; n < 2; ++n) {
                    int after0 = 1 + before3;
                    @inner for (int m = 0; m < 3; ++m) {
                        int after1 = 1 + after0;
                        @inner for (int k = 0; k < 5; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        @inner for (int k = 0; k < 5; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    @inner for (int m = 0; m < 5; ++m) {
                        int after1 = 1 + after0;
                        @inner for (int k = 0; k < 7; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        @inner for (int k = 0; k < 7; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                }
            }
        }
    }
    int before00 = 1 + before0;
    @outer for (int x = 0; x < 10; ++x) {
        // int before1 = 1 + before00;
        int before1 = 1;
        @outer for (int y = 0; y < 20; ++y) {
            int before2 = 1 + before1;
            @outer for (int z = 0; z < 30; ++z) {
                int before3 = 1 + before2;
                @inner for (int n = 0; n < 2; ++n) {
                    int after0 = 1 + before3;
                    @inner for (int m = 0; m < 3; ++m) {
                        int after1 = 1 + after0;
                        @inner for (int k = 0; k < 5; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        @inner for (int k = 0; k < 5; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    @inner for (int m = 0; m < 5; ++m) {
                        int after1 = 1 + after0;
                        @inner for (int k = 0; k < 7; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        @inner for (int k = 0; k < 7; ++k) {
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                }
            }
        }
    }
}
