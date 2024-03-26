#define ASSERT_MACRO static_assert(true, "hello");
#define MY_EMPTY_DEFINE(a)

MY_EMPTY_DEFINE(ASSERT_MACRO)

@kernel void mykern() {
    for (int i = 0; i < 10; i++; @outer ) {
        for (int j = 0; j < 10; j++; @inner ) {
        }
    }
}
