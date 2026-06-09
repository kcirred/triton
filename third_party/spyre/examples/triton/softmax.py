"""Row-wise softmax kernel using tensor descriptors (f16)."""

import triton
import triton.language as tl


@triton.jit
def softmax_kernel_desc(
    output_ptr, input_ptr, M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

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

    row = in_desc.load([row_idx, 0])
    row_max = tl.max(row, axis=1)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)
    denominator = tl.sum(numerator, axis=1)
    softmax_out = numerator / denominator
    out_desc.store([row_idx, 0], softmax_out)


SIGNATURE = {
    "output_ptr": "*fp16",
    "input_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
}
CONSTEXPRS = {"BLOCK_SIZE": 1024}
