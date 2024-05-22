typedef int* iPtr45 @dim(4, 5);
typedef int iMat455[4*5*5] @dim(4, 5, 5);

struct sMat24 {
  int* a @dim(2, 4);
};

@kernel void test(iPtr45 a @dimOrder(1, 0), sMat24 *b, iMat455 &ab @dimOrder(2, 1, 0), float *ac @dim(4,5) @dimOrder(0, 1)) {
  for (int i = 0; i < 4; ++i; @outer) {
    @shared int cc[5*4] @dim(4, 5) @dimOrder(0, 1);
    for (int j = 0; j < 5; ++j; @inner) {
      cc(j, i) = 0;

      for (int k = 0; k < j; ++k) {
        ab(k, j, i) = a(j, i);
        ab(k, j, i) += b->a(i, j);
      }

      ac(j, i) = a(j, i);
      ac(j, i) += b->a(i, j);
    }
  }
}