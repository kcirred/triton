// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// Physicalizing `linalg.broadcast` (issue #91).
//
// A `tt.spyre_tensor_layout` annotation asks for the Spyre "stick" layout: one
// logical axis splits into a stick index (floordiv) and a lane (mod), so a
// logical <64x128> becomes a physical <64x2x64> and the rank goes 2 -> 3.
// phys_src says which logical axis each physical dim comes from, phys_op is
// 0=identity / 1=floordiv / 2=mod, and phys_arg is the stick WIDTH (64), not a
// dim index.
//
// WHAT THESE TESTS ASK OF THE PASS: when the layout splits an axis the reduce
// CONSUMES, every axis the broadcast CARRIES stays whole, and the broadcast must
// follow its operands to the physical shape -- its `dimensions` list renumbered
// to name the new physical dims, its `outs` rebuilt at the new rank.
//
// The second case asks the opposite: with no reduce in front of it, nothing
// requires the broadcast to be physical, and it must be LEFT ALONE.

// -----

// A reduce feeding a broadcast feeding elementwise arithmetic -- the softmax
// shape, and the case this whole file exists for.
//
// Axis 1 (the columns, length 128) is split at 64 and is also the axis the
// reduce consumes. Required outcome: the reduce absorbs BOTH physical dims of
// that axis, the broadcast re-adds the same two, and the subtraction downstream
// ends up with operands of one shape.

// CHECK-LABEL: func @softmax_broadcast_split_reduced_axis
// The load is physical: rank 3, the split axis became 2 sticks of 64.
// CHECK: ktdp.load {{.*}} -> tensor<64x2x64xf32>
// The reduce absorbs BOTH physical dims of the split axis, so dimensions = [1, 2]
// and the result is rank 1.
// CHECK: linalg.reduce
// CHECK-SAME: outs(%{{.*}} : tensor<64xf32>)
// CHECK-SAME: dimensions = [1, 2]
// The broadcast re-adds those same two dims.
// CHECK: linalg.broadcast
// CHECK-SAME: outs(%{{.*}} : tensor<64x2x64xf32>)
// CHECK-SAME: dimensions = [1, 2]
// With the broadcast physical, the subtraction's operands finally agree.
// CHECK: arith.subf %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
// CHECK: math.exp %{{.*}} : tensor<64x2x64xf32>
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
module {
tt.func @softmax_broadcast_split_reduced_axis(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %m = "tt.reduce"(%a) ({ ^bb0(%x: f32, %y: f32): %mx = arith.maximumf %x, %y : f32
    tt.reduce.return %mx : f32 }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>
  // tt.expand_dims then tt.broadcast lower to a size-1 reshape plus a
  // linalg.broadcast; the reshape is a cancelling pair the analysis must cross.
  %me = tt.expand_dims %m {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
  %bc = tt.broadcast %me : tensor<64x1xf32> -> tensor<64x128xf32>
  %d = arith.subf %a, %bc : tensor<64x128xf32>
  %e = math.exp %d : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %e : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}

// -----

// The same broadcast with NO reduce in front of it. Nothing upstream of the
// broadcast reads the split axis, so the backward analysis -- the pass walking
// from each store toward its producers, asking what layout each value must
// have -- never asks the broadcast for one, and it stays LOGICAL. The pass instead bridges at the
// store: an scf.for slices the logical <64x128> into 2 sticks of 64 and
// assembles the physical <64x2x64>. Both outcomes are correct; which one you
// get depends on whether a physical value reaches the broadcast, so this test
// pins the no-reduce half of that choice.

// CHECK-LABEL: func @broadcast_only_split_added_axis
// The broadcast is left logical -- rank 2, dimensions = [1].
// CHECK: linalg.broadcast
// CHECK-SAME: outs(%{{.*}} : tensor<64x128xf32>)
// CHECK-SAME: dimensions = [1]
// CHECK: math.exp %{{.*}} : tensor<64x128xf32>
// The store bridge: one trip per stick (128/64 = 2), each copying a 64-wide slice.
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<64x2x64xf32>
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[EMPTY]])
// CHECK: tensor.extract_slice %{{.*}}[0, %{{.*}}] [64, 64] [1, 1] : tensor<64x128xf32> to tensor<64x64xf32>
// CHECK: tensor.insert_slice %{{.*}} into %[[ACC]][0, %[[IV]], 0] [64, 1, 64] [1, 1, 1]
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<64x2x64xf32>
module {
tt.func @broadcast_only_split_added_axis(%a_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c64 = arith.constant 64 : i32
  %c128 = arith.constant 128 : i32
  %s128 = arith.constant 128 : i64
  %s1 = arith.constant 1 : i64
  // Load a single column so the value fanned out is genuinely rank 1 (length 64)
  // and the broadcast ADDS the split axis rather than carrying it.
  %ad = tt.make_tensor_descriptor %a_ptr, [%c64, %c1], [%s1, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x1xf32>
  %a = tt.descriptor_load %ad[%c0, %c0] : !tt.tensordesc<64x1xf32> -> tensor<64x1xf32>
  %bc = tt.broadcast %a : tensor<64x1xf32> -> tensor<64x128xf32>
  %e = math.exp %bc : tensor<64x128xf32>
  %od = tt.make_tensor_descriptor %o_ptr, [%c64, %c128], [%s128, %s1] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 1, 2>, phys_arg = array<i64: 0, 64, 64>} : !tt.tensordesc<64x128xf32>
  tt.descriptor_store %od[%c0, %c0], %e : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
