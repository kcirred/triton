//===- Utility.h - Shared transform utilities for KTDP passes -------------===//

#ifndef KTDP_TRANSFORMS_UTILITY_H
#define KTDP_TRANSFORMS_UTILITY_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::ktdp {

/// Erase trivially dead ops in reverse walk order.
/// An op is erased only if BOTH conditions hold:
///   1. predicate(op) is true (or predicate is null — matches all ops)
///   2. isOpTriviallyDead(op) — terminators, symbols, and side-effecting ops
///      are never considered dead regardless of the predicate.
void cleanupDeadOps(ModuleOp module,
                    llvm::function_ref<bool(Operation *)> predicate = nullptr);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_UTILITY_H
