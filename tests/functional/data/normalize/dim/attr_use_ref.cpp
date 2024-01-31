typedef [[okl::dim("(4,5)")]] int *iPtr45;
typedef [[okl::dim("(4,5,5)")]] int iMat455[4 * 5 * 5];

struct sMat24 {
  [[okl::dim("(2,4)")]] int *a;
};

[[okl::kernel("(void)")]] void
test([[okl::dimOrder("(1,0)")]] iPtr45 a, sMat24 *b,
     [[okl::dimOrder("(2,1,0)")]] iMat455 &ab,
     [[okl::dim("(4,5)")]] [[okl::dimOrder("(0,1)")]] float *ac) {
  [[okl::outer("(void)")]] for (int i = 0; i < 4; ++i) {
    [[okl::shared("(void)")]] [[okl::dim("(4,5)")]] [[okl::dimOrder("(0,1)")]] int cc[5 * 4];
    [[okl::inner("(void)")]] for (int j = 0; j < 5; ++j) {
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