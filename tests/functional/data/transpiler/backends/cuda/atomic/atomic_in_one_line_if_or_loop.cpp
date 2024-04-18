@kernel void buggy_kernel() {
    for (int i = 0; i < 100; ++i; @outer) {
        @shared float shared_val[10];
        for (int j = 0; j < 100; ++j; @inner) {
            for (int z = 0; z < 2; ++z )
               @atomic shared_val[z] += j;
        }
    }
    for (int i = 0; i < 100; ++i; @outer) {
        @shared float shared_val[10];
        for (int j = 0; j < 100; ++j; @inner) {
            if( j<100 )
               @atomic shared_val[j] += j;
        }
    }
}
