// int const, const int
const int var_const0 = 0;
int const var_const1 = 0;

// volatile qualifier
volatile const int var_const2 = 0;
volatile int const var_const3 = 0;

// Stupid formatting
const int var_const4 = 0;

int const var_const5 = 0;

// At least one @kern function is required
@kernel void kern() {
    @outer for (int i = 0; i < 32; ++i) {
        @inner for (int j = 0; j < 32; ++j) {
        }
    }
}
