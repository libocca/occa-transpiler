#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(10) void _occa_mykern_0(int aaa, int bbb) {
    {
        int i = (0) + blockIdx.x;
        {
            int j = (0) + threadIdx.x;
            // BODY
        }
    }
}
