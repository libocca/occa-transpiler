# OKL specification

Occa kernel language is an C++ code extension that adds attributes of '@name` format, but also supports clean c++ code with CXX11 attributes.
While OCCA aims to abstract device management, OKL aims to abstract the programming of devices.

## Attributes
There are two attributes styles:
- Original -- format: `@name(arguments)`
- Normalized -- CXX11 attribute, format: `[[name(arguments)]]`

### \@kernel / \[\[okl_kernel("")\]\]
**Description**:
Declares a function as `kernel` function that can run in parallel across multiple compute threads

**Syntax**:
- `@kernel` doesn't take any arguments

**Semantic rules:**
- Applies to functions only
- Function must return void
- It is allowed to have multiple `@kernel` functions in the file, but there has to be at least one
- Each kernel has to have at least one `@outer` and one `@inner` loop

**Example**:
~~~{.cpp}
@kernel void test_kernel() {
    // ...
}
~~~

### \@outer / \[\[okl_outer("")\]\]
**Description**
Can be used only inside `@kernel` decorated functions. Decorates a `for` loop to be run in parallel across multiple compute threads. Declaration can be used to switch between `x`, `y` and `z` indexed synchronized compute threads on targets that support it, otherwise it has no effect. `@outer` loops corresponds to parallelization over `block` in CUDA and `workgroup` in OpenCL.

**Syntax**
- `@outer(<number>=0)`
- Number is axis and can be 0, 1 or 2 corresponding to x, y and z.
- Number is optional. If it's not specified, it is calculated automatically: starting with zero from the lowest `@inner` loop

**Semantic rules:**
- Applies to for loop only
- Please refer to [kernel structure](#kernel-structure) section to find out about other `@outer` constraints/rules introduced by OKL semantics.

~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @outer for (int i2 = 0; i2 < 10; ++i2) {
            /// ...
        }
    }
    @outer(1) for (int i = 0; i < 32; ++i) {
        for (int i2 = 0; i2 < 10; ++i2) {
            for (int i2 = 0; i2 < 10; ++i2; @outer(1)) {
                // ...
            }
        }
    }
}
~~~

### \@inner / \[\[okl_inner("")\]\]
**Description**
Can only be used inside `@outer` decorated loops. Decorates a `for` loop to be run in parallel across multiple compute threads for targets that support parallelizing inner loops. Declaration can be used to switch between `x`, `y` and `z` indexed synchronized compute threads on targets that support it, otherwise it has no effect. `@inner` loops corresponds to parallelization over `thread` in CUDA and `workitem` in OpenCL.

**Syntax**
- `@inner(<number>)`
- Number is axis and can be 0, 1 or 2 corresponding to x, y and z.
- Number is optional. If it's not specified, it is calculated automatically: starting with zero from the lowest `@inner` loop

**Semantic rules:**
- Applies to for loop only
- Please refer to [kernel structure](#kernel-structure) section to find out about other `@outer` constraints introduced by OKL semantics.

**Example:**
~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            @inner for (int j = 0; j < 32; ++j) {
                // ...
            }
            for (int k = 0;  k < 32; ++k) {
                @inner for (int j = 0; j < 32; ++j) {
                    // ...
                }
            }
        }
    }
}
~~~

## Kernel structure
