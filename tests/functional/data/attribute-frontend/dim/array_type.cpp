#define W 100
#define H 100

typedef @dim(W, H, "rowMajor") int *Mat2d;

template <class T>
void f(T *a) {

}

@kernel void f0(@ restrict int *a, @ restrict int *b) {
  Mat2d mat[100];
  @barrier;

  {
    @outer(1) for (int i = 0; i < 100; ++i) {
      @inner(1) for (int j = 0; j < 100; ++j) {
        mat[i + j] = 0;
      }
      @inner(1) for (int j = 0; j < 100; ++j) {
        mat[i + j] = 0;
      }
      @inner(1) for (int j = 0; j < 100; ++j) {
        mat[i + j] = 0;
      }
    }

    @outer(1) for (int i = 0; i < 100; ++i) {
      @inner(1) for (int j = 0; j < 100; ++j) {
        mat[i + j] = 0;
      }
    }
  }

  Mat2d c[100];
  f(c);
  c(0, 0) = 0;

  @dim(3,3,3) int d[100];
  d(0, 0, 2) = 0;

  Mat2d cc[100][100];
  cc[0][0](1,2) = 0;
}
