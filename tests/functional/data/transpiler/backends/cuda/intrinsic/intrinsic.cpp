#include "okl_intrinsic.h"


@kernel void intrinsic_builtin(const float* fVec, float* fSum) {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            float value = okl_intrinsic_exp10(fVec[i]);
            @atmomic* fSum += value;		    
            //@atomic* iSum += iVec[0];
            //@atomic* fSum += fVec[0];
        }
    }
}

