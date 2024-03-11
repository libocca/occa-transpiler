typedef float* A;

typedef float FLOAT;

struct CC
{
  float ccc;
};

struct C
{
  CC cc;
  float d[3];
};


@kernel void addVectors(const int entries,
                        const A a,
                        const FLOAT* b,
                        const C c,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(4, @outer, @inner)) {
    ab[i] = a[i] + b[i] + c.cc.ccc + c.d[0];
  }
}

@kernel void addVectors2(const int entries,
                        const float *a,
                        const float const *b,
                        float const *c,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(4, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
