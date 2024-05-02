# OKLT specification

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
Can be used only inside `@kernel` decorated functions. Decorates a `for` loop to be run in parallel across multiple compute threads. Declaration can be used to switch between `x`, `y` and `z` indexed synchronized compute threads on targets that support it, otherwise it has no effect.

**Arguments**
Syntax: `@outer(<number>=0)`

**Semantic rules:**
- Applies to for loop only
