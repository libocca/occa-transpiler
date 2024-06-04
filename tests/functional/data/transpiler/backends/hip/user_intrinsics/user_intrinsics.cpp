#include "custom_intrinsics/api.h"

@kernel void zero_nans(float* vec) {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {
	    int idx = i * 32 + j;
	    float value = vec[idx];
	    if(okl_is_nan(value)) {
	       vec[idx] = 0.0f;
	    }
        }
    }
}

