# vector_add

Elementwise `C = A + B` with tensor descriptors. Exercises the **one
program per core** idiom: each of the 32 Spyre cores runs one program that
loops over its share of the sequence.

## Variants

### 1D


- **default** (`vector_add`) — `n_elements` is a `constexpr`, baked into
  the TTIR as a literal. Descriptor shapes are fully static (`memref<Nxf32>`).
- **dynamic** (`vector_add__dynamic`) — `n_elements` is a runtime `i32`.
  Descriptor shapes are dynamic (`memref<?xf32>`). Exercises the
  dynamic-shape path through `LowerDescriptorMemory`.

### 2D

- **2d** (`vector_add__2d`) — `shape=[M, N]` with all dimensions as
  `constexpr`. Nested M × N tile loops with `cdiv` block counts and
  bounds-clamping across both dimensions. Static descriptor shapes
  (`memref<MxNxf32>`).
- **2d_dynamic** (`vector_add__2d_dynamic`) — Same kernel with `M` and `N`
  as runtime `i32`. Dynamic descriptor shapes (`memref<?x?xf32>`).

### 3D

- **3d** (`vector_add__3d`) — `shape=[M, N, P]` with all dimensions as
  `constexpr`. Nested M × N × P tile loops with explicit stride computation
  (`stride_m = N * P`, `stride_n = P`). Static descriptor shapes
  (`memref<MxNxPxf32>`).
- **3d_dynamic** (`vector_add__3d_dynamic`) — Same kernel with `M`, `N`,
  and `P` as runtime `i32`. Dynamic descriptor shapes (`memref<?x?x?xf32>`).
