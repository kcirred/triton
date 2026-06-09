"""Row-wise softmax: three implementations sharing the same math.

All three compute per-row softmax over a 2D ``[M, N]`` input tensor in
f16, distributing rows across 32 Spyre cores via an explicit outer loop.
They differ in how N is handled:

- :func:`softmax_single_tile` — N fits entirely in one
  ``BLOCK_SIZE``-wide tile (no N-loop; ``tt.reduce`` covers max + sum
  within the tile).
- :func:`softmax_multi_tile` — N spans multiple ``BLOCK_N``-wide tiles
  (3-pass algorithm: row-max, denominator, normalize).
- :func:`softmax_2pass` — Milakov & Gimelshein (2018) online softmax.
  Pass 1 fuses max and denominator in one N-tile sweep with a running
  correction; pass 2 normalizes. Reads the input 2× per row vs 3× for
  the multi-tile variant.

All three variants partition rows across cores via
``tl.num_programs(0)``, which DistributeWork folds to a compile-time
constant against ``SpyreOptions.grid``.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_single_tile(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    in_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_SIZE],
    )

    rows_per_core = tl.cdiv(M, num_cores)
    start = pid * rows_per_core
    end = tl.minimum(start + rows_per_core, M)

    for row_idx in range(start, end):
        row = in_desc.load([row_idx, 0])
        row_max = tl.max(row, axis=1)
        row_shifted = row - row_max
        numerator = tl.exp(row_shifted)
        denominator = tl.sum(numerator, axis=1)
        softmax_out = numerator / denominator
        out_desc.store([row_idx, 0], softmax_out)


@triton.jit
def softmax_multi_tile(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    in_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_N],
    )

    rows_per_core = tl.cdiv(M, num_cores)
    start = pid * rows_per_core
    end = tl.minimum(start + rows_per_core, M)
    n_tiles = tl.cdiv(N, BLOCK_N)

    for row_idx in range(start, end):
        # Pass 1: find row max.
        row_max = tl.full([1], value=float('-inf'), dtype=tl.float32)
        for n in range(n_tiles):
            col_offset = n * BLOCK_N
            tile = in_desc.load([row_idx, col_offset])
            tile_max = tl.max(tile, axis=1)
            row_max = tl.maximum(row_max, tile_max)

        # Pass 2: compute denominator.
        denom = tl.full([1], value=0.0, dtype=tl.float32)
        for n in range(n_tiles):
            col_offset = n * BLOCK_N
            tile = in_desc.load([row_idx, col_offset])
            tile_shifted = tile - row_max
            denom += tl.sum(tl.exp(tile_shifted), axis=1)

        # Pass 3: normalize and store.
        for n in range(n_tiles):
            col_offset = n * BLOCK_N
            tile = in_desc.load([row_idx, col_offset])
            tile_shifted = tile - row_max
            softmax_out = tl.exp(tile_shifted) / denom
            out_desc.store([row_idx, col_offset], softmax_out)


@triton.jit
def softmax_2pass(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Online softmax (Milakov & Gimelshein 2018), BLOCK_M × BLOCK_N tiled.

    Online update per N-tile in pass 1:
        new_max = max(row_max, tile_max)
        denom   = denom * exp(row_max - new_max)
                + sum(exp(tile - new_max))
        row_max = new_max

    Pass 2 normalizes with the final (row_max, denom).

    Constraint: ``N`` must be divisible by ``BLOCK_N``. The descriptor
    pads OOB loads with zero, which would corrupt the max-reduction in
    pass 1 (zeros look like valid values). Masking is not available on
    descriptor loads, so we require exact divisibility.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    in_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    rows_per_core = tl.cdiv(M, num_cores)
    row_chunk_start = pid * rows_per_core
    row_chunk_end = tl.minimum(row_chunk_start + rows_per_core, M)

    n_tiles = tl.cdiv(N, BLOCK_N)
    m_tiles = tl.cdiv(row_chunk_end - row_chunk_start, BLOCK_M)

    for m in range(m_tiles):
        row_start = row_chunk_start + m * BLOCK_M

        # --- Pass 1: fused online max + denominator ---
        row_max = tl.full([BLOCK_M, 1], value=float('-inf'), dtype=tl.float32)
        denom = tl.full([BLOCK_M, 1], value=0.0, dtype=tl.float32)

        for n in range(n_tiles):
            col_offset = n * BLOCK_N
            tile = in_desc.load([row_start, col_offset])
            tile_f32 = tile.to(tl.float32)

            tile_max = tl.max(tile_f32, axis=1, keep_dims=True)  # [BLOCK_M, 1]
            new_max = tl.maximum(row_max, tile_max)

            # Rescale accumulated denom, add this tile's contribution.
            denom = denom * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(tile_f32 - new_max), axis=1, keep_dims=True
            )
            row_max = new_max

        # --- Pass 2: normalize and store ---
        for n in range(n_tiles):
            col_offset = n * BLOCK_N
            tile = in_desc.load([row_start, col_offset])
            tile_f32 = tile.to(tl.float32)

            softmax_out = tl.exp(tile_f32 - row_max) / denom
            out_desc.store(
                [row_start, col_offset], softmax_out.to(tl.float16)
            )
