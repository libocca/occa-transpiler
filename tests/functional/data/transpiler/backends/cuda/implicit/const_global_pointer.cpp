// pointer to const
const int* ptr_const0 = 0;
int const* ptr_const1 = 0;

// const pointer to const
const int* const ptr_const2 = 0;
int const* const ptr_const3 = 0;

// const pointer to non const
int* const ptr_const4 = 0;

// Stupid formatting
const int* ptr_const5 = 0;

// At least one @kern function is requried
@kernel void kern () {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {}
    }
}
