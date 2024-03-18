#define __restrict__ @ restrict
#define __shared__ @shared

#define BLOCK_SYNC(a, b) \
    a += 1;              \
    b += 1;              \
    @barrier;

typedef struct {
    __restrict__ float* b;
    @ restrict float* c;
} S;
@ restrict float* aa;

@kernel void hello_kern(S* a) {
    for (int i = 0; i < 10; ++i; @outer) {
        __shared__ float buf[100];
        int a, b;
        for (int j = 0; j < 10; ++j; @inner) {
            BLOCK_SYNC(a, b)
        }
    }
}
