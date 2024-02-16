// clang format off

@kernel void runDiff(@restrict int* a, @restrict int* b){
  @tile(16, @inner(0), @outer(1), check=true) for (int i=0; i< 100; ++i) {
    b[i] = a[i] - 100;
  }
}
