#define __restrict__ @restrict
#define __shared__ @shared

typedef struct {
    __restrict__ float* b;
    @restrict float* c;
} S;
@restrict float* aa;

@kernel void hello_kern(S* a) {
    for (int i = 0; i < 10; ++i; @outer) {
        __shared__ float buf[100];
        for (int j = 0; j < 10; ++j; @inner) {
        }
    }
}
