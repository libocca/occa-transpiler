extern "C" void testA0() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        int _occa_exclusive_index;
        int k[10];
        _occa_exclusive_index = 0;
        for (int j = 0; j < 10; ++j) {
            k[_occa_exclusive_index] = i + j;
            ++_occa_exclusive_index;
        }
    }
}

extern "C" void testA1() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        int _occa_exclusive_index;
        int k[3];
        _occa_exclusive_index = 0;
        for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
            for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                if (j < 10) {
                    k[_occa_exclusive_index] = i + j;
                }
            }
            ++_occa_exclusive_index;
        }
    }
}

extern "C" void testA2() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        int _occa_exclusive_index;
        int k[4];
        for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
            _occa_exclusive_index = 0;
            for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                if (j < 10) {
                    k[_occa_exclusive_index] = i + j;
                }
                ++_occa_exclusive_index;
            }
        }
    }
}

extern "C" void testA3() {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        int _occa_exclusive_index;
        int k[12];
        _occa_exclusive_index = 0;
        for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
            for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                if (j < 10) {
                    k[_occa_exclusive_index] = i + j;
                }
                ++_occa_exclusive_index;
            }
        }
    }
}

extern "C" void testB0() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        int _occa_exclusive_index;
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            if (i < 10) {
                int k[10];
                _occa_exclusive_index = 0;
                for (int j = 0; j < 10; ++j) {
                    k[_occa_exclusive_index] = i + j;
                    ++_occa_exclusive_index;
                }
            }
        }
    }
}

extern "C" void testB1() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        int _occa_exclusive_index;
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            if (i < 10) {
                int k[3];
                _occa_exclusive_index = 0;
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                    }
                    ++_occa_exclusive_index;
                }
            }
        }
    }
}

extern "C" void testB2() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        int _occa_exclusive_index;
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            if (i < 10) {
                int k[4];
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    _occa_exclusive_index = 0;
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                        ++_occa_exclusive_index;
                    }
                }
            }
        }
    }
}

extern "C" void testB3() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        int _occa_exclusive_index;
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            if (i < 10) {
                int k[12];
                _occa_exclusive_index = 0;
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                        ++_occa_exclusive_index;
                    }
                }
            }
        }
    }
}

extern "C" void testC0() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            int _occa_exclusive_index;
            if (i < 10) {
                int k[10];
                _occa_exclusive_index = 0;
                for (int j = 0; j < 10; ++j) {
                    k[_occa_exclusive_index] = i + j;
                    ++_occa_exclusive_index;
                }
            }
        }
    }
}

extern "C" void testC1() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            int _occa_exclusive_index;
            if (i < 10) {
                int k[3];
                _occa_exclusive_index = 0;
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                    }
                    ++_occa_exclusive_index;
                }
            }
        }
    }
}

extern "C" void testC2() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            int _occa_exclusive_index;
            if (i < 10) {
                int k[4];
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    _occa_exclusive_index = 0;
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                        ++_occa_exclusive_index;
                    }
                }
            }
        }
    }
}

extern "C" void testC3() {
#pragma omp parallel for
    for (int _occa_tiled_i = (0); _occa_tiled_i < 10; _occa_tiled_i += 4) {
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + 4); ++i) {
            int _occa_exclusive_index;
            if (i < 10) {
                int k[12];
                _occa_exclusive_index = 0;
                for (int _occa_tiled_j = (0); _occa_tiled_j < 10; _occa_tiled_j += 4) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + 4); ++j) {
                        if (j < 10) {
                            k[_occa_exclusive_index] = i + j;
                        }
                        ++_occa_exclusive_index;
                    }
                }
            }
        }
    }
}
