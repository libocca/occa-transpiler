#define MYMACRO 1  // my first comment

#if MYMACRO  //  my second comment

#define HELLOMACRO 1

#endif  //  my third comment

@kernel void hello_kern() {
    for (int i = 0; i < 10; ++i; @outer) {
        for (int j = 0; j < 10; ++j; @inner) {
        }
    }
}
