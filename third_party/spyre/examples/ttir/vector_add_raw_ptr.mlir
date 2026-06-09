// Vector-add TTIR using raw pointer arithmetic (non-descriptor path).
//
// This is the classic Triton pattern using tt.load/tt.store with pointer
// splat + addptr.  Kept as a reference — the descriptor-based version
// (vector_add.mlir) is the preferred path for Spyre.

module {
  tt.func public @add_kernel(
      %x_ptr: !tt.ptr<f16>,
      %y_ptr: !tt.ptr<f16>,
      %output_ptr: !tt.ptr<f16>,
      %n_elements: i32
  ) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32

    %block_size = arith.constant 1024 : i32
    %pid_i64 = arith.extsi %pid : i32 to i64
    %block_size_i64 = arith.extsi %block_size : i32 to i64
    %block_start_i64 = arith.muli %pid_i64, %block_size_i64 : i64
    %block_start = arith.muli %pid, %block_size : i32

    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %block_start_splat = tt.splat %block_start : i32 -> tensor<1024xi32>
    %abs_offsets = arith.addi %block_start_splat, %offsets : tensor<1024xi32>

    // Mask: only process elements within bounds
    %n_splat = tt.splat %n_elements : i32 -> tensor<1024xi32>
    %mask = arith.cmpi slt, %abs_offsets, %n_splat : tensor<1024xi32>

    // Load x[offsets]
    %x_base = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %x_ptrs = tt.addptr %x_base, %abs_offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %x = tt.load %x_ptrs, %mask : tensor<1024x!tt.ptr<f16>>

    // Load y[offsets]
    %y_base = tt.splat %y_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %y_ptrs = tt.addptr %y_base, %abs_offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %y = tt.load %y_ptrs, %mask : tensor<1024x!tt.ptr<f16>>

    // Compute and store
    %output = arith.addf %x, %y : tensor<1024xf16>
    %out_base = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %out_ptrs = tt.addptr %out_base, %abs_offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %out_ptrs, %output, %mask : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}
