@kernel void testA0() {
    @outer for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @inner for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testA1() {
    @outer for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testA2() {
    @outer for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @outer, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testA3() {
    @outer for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testB0() {
    @tile(4, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @inner for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testB1() {
    @tile(4, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testB2() {
    @tile(4, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @outer, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testB3() {
    @tile(4, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testC0() {
    @tile(4, @outer, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @inner for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testC1() {
    @tile(4, @outer, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testC2() {
    @tile(4, @outer, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @outer, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}

@kernel void testC3() {
    @tile(4, @outer, @outer) for (int i = 0; i < 10; ++i) {
        @exclusive int k;
        @tile(4, @inner, @inner) for (int j = 0; j < 10; ++j) {
            k = i + j;
        }
    }
}
