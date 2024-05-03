#include "utils.h"


@kernel void function1(const Data data1, const Data data2) {
      @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
        }
    }
}

