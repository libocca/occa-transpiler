# Tile attribute description

## Syntax
```
@tile(tile_size,[@inner[([0|1|2])]/@outer[([0|1|2])],[@inner[([0|1|2])]/outer[([0|1|2])]],[check=true/false])
```
- Can be applied only to a for loop

## Idea
split loop into two loops so that first loop iterates overa range with step of tile size and second loop iterates over tile size with step `inc`:
```C++
@tile(tile_size) for (int i = start; i < end; i+=inc) { ... }
```
->

```C++
for (int _occa_tiled_i = start; _occa_tiled_i < end; _occa_tiled_i += tile_size) {
    for (int i = _occa_tiled_i; i < (_occa_tiled_i + tile_size); i += inc) {...}
}
```

## Loops types
- You can specify type of first and second loops (`@outer`/`@inner`). If not specified, regular loop (no attributes) is used.
- `@outer` calculates first loop index with blockIdx (cuda/hip)
- `@inner` calculates second loop index with threadIdx (cuda/hip)

Example:
```C++
@tile(8, @outer, @inner, check=false) for (int i = 0; i < 64; i+=2) { ... }
```
->

```C++
@outer for (int _occa_tiled_i = 0; _occa_tiled_i < 64; _occa_tiled_i += 8) {
    @inner for (int i = _occa_tiled_i; i < (_occa_tiled_i + 8); i+=2) {...}
}
```

->
```C++
int _occa_tiled_i = (0) + ((8 * 2) * blockIdx.x);
{
    int i = _occa_tiled_i + ((2) * threadIdx.x);
    { ... }
}
```

Please check out [@inner](./outer.md) and [@outer](./inner.md) if you haven't checked yet

## Bound check can be set or unset (default check=true)
Example:
```C++
for (int i = 0; i < 64; i += 2; @tile(4, @outer(0), @inner(0), check = false)) {
    for (int j = 0; j < 64; j += 2; @tile(4, @inner(1), check = true)) { ... }
}
```
->

```C++
int _occa_tiled_i = (0) + (((4) * 2) * blockIdx.x);
{
    int i = _occa_tiled_i + ((2) * threadIdx.x);
    int _occa_tiled_j = (0) + (((4) * 2) * threadIdx.y);
    for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); j += 2)
        if (j < 64) { ... }
}
```
