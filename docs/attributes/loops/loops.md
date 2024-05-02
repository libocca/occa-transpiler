# Loops attribute descriptions (@outer, @inner, @tile)
- [@outer](./outer.md)
- [@inner](./inner.md)
- [@tile](./tile.md)

## Dimensions

- You can specify dimensions inside @inner, @outer, - 0, 1 or 2, corresponding to x, y, z
Example:
```C++
@tile(8, @outer(1), @inner(2), check=false) for (int i = 0; i < 64; i+=2) { ... }
```
-> ->
```C++
int _occa_tiled_i = (0) + ((8 * 2) * blockIdx.y);
{
    int i = _occa_tiled_i + ((2) * threadIdx.z);
    { ... }
}
```

- If dimensions are not specifies, they are calculated automatically, starting from x for the deepest nested @inner or @outer loop

## Loops constraints
- Loop should create integer variable (init) with some initial value (can be a result of an expression)
- Condition should be simple comparison to a value (or expression).
- Third part of for loop should be a simple unary increment/decrement or add/sub assign.
- In other words, loop must be a simple iteration over range of values in increasing or decreasing order

### Nested Loops constraints

- It is incorrect to have more than 3 nested inner or outer loops, since there are only 3 dimensions
- It is incorrect to have @outer loop after/inside @inner loop.
- Two or more inner loops on the same level must have the same size.