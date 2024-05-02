# OKL specification

Occa kernel language is an C++ code extension that adds attributes of `@name` format, but also supports clean c++ code with CXX11 attributes.
While OCCA aims to abstract device management, OKL aims to abstract the programming of devices.

## Attributes
There are two attributes styles:
- Original -- format: `@name(arguments)`
- Normalized -- CXX11 attribute, format: `[[name(arguments)]]`

### \@kernel / \[\[okl_kernel("")\]\]
**Description:**
Declares a function as `kernel` function that can run in parallel across multiple compute threads

**Syntax**:
- `@kernel` doesn't take any arguments

**Semantic rules**
- Applies to functions only
- Function must return void
- It is allowed to have multiple `@kernel` functions in the file, but there has to be at least one
- Each kernel has to have at least one `@outer` and one `@inner` loop

**Example**
~~~{.cpp}
@kernel void test_kernel() {
    // ...
}
~~~

### \@outer / \[\[okl_outer("")\]\]
**Description:**
Can be used only inside `@kernel` decorated functions. Decorates a `for` loop to be run in parallel across multiple compute threads. Declaration can be used to switch between `x`, `y` and `z` indexed synchronized compute threads on targets that support it, otherwise it has no effect. `@outer` loops corresponds to parallelization over `block` in CUDA and `workgroup` in OpenCL.

**Syntax**
- `@outer(<number>=0)`
- Number is axis and can be 0, 1 or 2 corresponding to x, y and z.
- Number is optional. If it's not specified, it is calculated automatically: starting with zero from the deepest `@inner` loop

**Semantic rules**
- Applies to `for` loop only
- There can't be more that 3 (x,y,z) nested `@outer` loops.
- Please refer to [kernel structure](#kernel-structure) section to find out about other `@outer` constraints/rules introduced by OKL semantics.

~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @outer for (int i2 = 0; i2 < 10; ++i2) {
            // ...
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
**Description:**
Can only be used inside `@outer` decorated loops. Decorates a `for` loop to be run in parallel across multiple compute threads for targets that support parallelizing inner loops. Declaration can be used to switch between `x`, `y` and `z` indexed synchronized compute threads on targets that support it, otherwise it has no effect. `@inner` loops corresponds to parallelization over `thread` in CUDA and `workitem` in OpenCL.

**Syntax**
- `@inner(<number>)`
- Number is axis and can be 0, 1 or 2 corresponding to x, y and z.
- Number is optional. If it's not specified, it is calculated automatically: starting with zero from the deepest `@inner` loop

**Semantic rules**
- Applies to `for` loop only
- There can't be more that 3 (x,y,z) nested `@inner` loops.
- Please refer to [kernel structure](#kernel-structure) section to find out about other `@outer` constraints introduced by OKL semantics.

**Example**
~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            for (int j = 0; j < 32; ++j; @inner) {
                // ...
            }
            for (int k = 0;  k < 32; ++k) {
                @inner(0) for (int j = 0; j < 32; ++j) {
                    // ...
                }
            }
        }
    }
}
~~~

### \@tile / \[\[okl_tile("")\]\]
**Description:**
Can be used only inside `@kernel` decorated functions. Decorates a `for` loop to be run in parallel across multiple compute threads in groups `<number>` sized. Optional arguments `<kword>` can only be @outer and @inner explaining how to parallelize loop across multiple compute threads. Last optional argument enables/disables a check for the tiles loops that prevents them from going over the loop scope. Check is enabled by default for all tiled loops.

Code below:
~~~{.cpp}
for (int i = 0; i < N; ++i; @tile(16, @outer, @inner, check=false)) {
  // work
}
~~~
Is equivalent to
~~~{.cpp}
for (int iTile = 0; iTile < N; iTile += 16; @outer) {
  for (int i = iTile; i < (iTile + 16); ++i; @inner) {
    if (i < N) {
      // work
    }
  }
}
~~~

**Syntax**
- `@tile(<number>, [<kword>], [<kword>], [check = bool])`
- First argument is tile size,
- Second and third argument is parallelization method of first (tile index) and second (0..tile_size) loops. If skipped, then no attribute is applies to the loop
- check=false/true. If true, then adds boundary check for index. If you know that loop size is divisible by tile size, you can skip this check. Default: true

**Semantic rules**
- Applies to `for` loops only
- Note that if you use @outer-@inner types, then you can't put any `@shared` or `@exclusive` variables that must be defined between `@outer` and `@inner` loops

**Example**
~~~{.cpp}
@kernel void test_kernel() {
    @tile(16, @outer, @outer, check=false) for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j; @inner) {
            // ...
            @inner(0) for (int jj = 0; jj < 32; ++jj) {
                // ...
            }
        }
        for (int k = 0; k < 32; ++k) {
            @tile(4, @inner(1), @inner(0), check=true) for (int j = 0; j < 32; ++j) {
                // ...
            }
        }
    }
}
~~~

### \@max_inner_dims / \[\[okl_max_inner_dims("")\]\]
**Description:**
For backends that require launcher code, we must know sizes of @inner loops (sizes of x,y,z loops). Usually it is calculated from the code, but user can overwrite these values.

**Syntax**
- `@max_inner_dims(<number>, <number>, <number>)`
- Each number is a size of x, y and z axis, defauling to 1.

**Semantic rules**
- Applies to highest `@outer` `for` loop only

**Example**
~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i; @max_inner_dims(32, 32, 64)) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            @inner for (int j = 0; j < 32; ++j) {
                // ...
            }
            for (int k = 0; k < 32; ++k) {
                @inner for (int j = 0; j < 32; ++j) {
                    // ...
                }
            }
        }
    }

    @outer for (int i = 0; i < 32; ++i; @max_inner_dims(10, 10)) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            @inner for (int j = 0; j < 10; ++j) {
                // ...
            }
            for (int k = 0; k < 32; ++k) {
                @inner for (int j = 0; j < 10; ++j) {
                    // ...
                }
            }
        }
    }
}

~~~

### \@dim / \[\[okl_dim("")\]\]
**Description:**
Transparently transforms variable or type declaration into a multi-indexed array. Variable of type attributed by `@dim` can be indexed with virtual operator() (see example)

**Syntax**
- `@dim(<expr>, <expr>...)`

**Semantic rules**
- Applies to type declaration or variable declaration.
- Value can be indexed with comma separated expressions in parenthesis: `int* mat34 @dim(3, 4)` -> `mat(0, 1+1) = 12`.
- Number of indecies when indecing must be the same as number of dimensions.

**Example**
~~~{.cpp}
typedef float* fmat10_10_t @dim(10, 10);

@kernel void test_kernel(int* mat34 @dim(3, 4), fmat10_10_t mat10_10) {
    @outer for (int i = 0; i < 32; ++i) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            @inner for (int j = 0; j < 32; ++j) {
                mat34(i, i2) = j;
                mat10_10(i, j) = 1.0f / (i + j + 1);
            }
        }
    }
}
~~~


### \@dimOrder / \[\[okl_dimOrder("")\]\]
**Description:** Changes order of dimensions accepted by `@dim` decorated types and variables. Accepts at least one argument, and each parameter represents dimension index starting from 0 and must be unique.

**Syntax**
- `@dimOrder(<number>, <number>...)`

**Semantic rules**
- Applies to type or variable declarations. Type or variable must be attributed with `@dim`
- Number of arguments must be the same as number of dimensions in `@dim`

**Example**
~~~{.cpp}
typedef float* fmat10_10_10_t @dim(10, 10, 10);

@kernel void test_kernel(int* mat34 @dim(3, 4) @dimOrder(1, 0), fmat10_10_10_t mat10_10_10 @dimOrder(2, 0, 1)) {
    for (int i = 0; i < 32; ++i; @outer) {
        @outer for (int i2 = 0; i2 < 32; ++i2) {
            @inner for (int j = 0; j < 32; ++j) {
                mat34(i, i2) = j;
                mat10_10_10(i, j, i2) = 1.0f / (i + j + 1);
                state.velocity(1) = mat10_10_10(i, j, i2);
            }
        }
    }
}
~~~

### \@shared / \[\[okl_shared("")\]\]
**Description:**
The concept of shared memory is taken from the GPU programming model, where parallel threads/workitems can share data.
Adding the @shared attribute when declaring a variable type will allow the data to be shared across inner loop iterations.

**Syntax**
- `@shared` doesn't take any arguments

**Semantic rules**
- Applies to type declaration or variable declaration
- Variable with `@shared` attribute must be declared between `@outer` and `@inner` loops
- Variable must be a constant size array

**Example**
~~~{.cpp}
@kernel void test_kernel() {
    @outer for (int i = 0; i < 32; ++i) {
        @shared int a[32];
        @inner for (int j = 0; j < 32; ++j) {
            a[j] = 12;
        }
    }
}
~~~


### \@exclusive / \[\[okl_exclusive("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


### \@atomic / \[\[okl_atomic("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


### \@restrict / \[\[okl_restrict("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


### \@barrier / \[\[okl_barrier("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


### \@nobarrier / \[\[okl_nobarrier("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


### \@simd_length / \[\[okl_simd_length("")\]\]
**Description:**

**Syntax**

**Semantic rules**

**Example**


## Kernel structure
### Loops tree structure
- There can't be more than 3 nested `@outer` and `@inner` loops (x,y,z axis)
- Tree from attributed loops has the following restrictions (These restrictions come from usual scheme of parallelization for heterogeneous systems (like in CUDA, HIP, OpenCL, etc.)):
  - All nodes on the same level have the same attribute
  - All tree leaves must be on the same level
  - Note: attributed loop tree consists only from attributed loops, meaning that regular loops can be everywhere, as long as attributed loops structure remains valid.
- Between top-level `@inner` loops there is an implicit synchronization (`@barrier`). To turn it off, use `@nobarrier`