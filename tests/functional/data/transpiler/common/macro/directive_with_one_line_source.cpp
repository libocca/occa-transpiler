@directive("#define MYMACRO 0") @kernel void hello_kern() { for (int i = 0; i < 10; ++i; @outer) { for (int j = 0; j < 10; ++j; @inner) {}} }
