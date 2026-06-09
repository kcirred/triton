"""Vector-add kernel using tensor descriptors."""

import triton
import triton.language as tl


@triton.jit
def add_kernel_desc(x_ptr, y_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )

    x = x_desc.load([offset])
    y = y_desc.load([offset])
    output = x + y
    out_desc.store([offset], output)


SIGNATURE = {
    "x_ptr": "*fp16",
    "y_ptr": "*fp16",
    "output_ptr": "*fp16",
}
CONSTEXPRS = {"BLOCK_SIZE": 1024, "n_elements": 98304}
