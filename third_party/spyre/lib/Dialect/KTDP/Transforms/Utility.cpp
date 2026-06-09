//===- Utility.cpp - Shared transform utilities for KTDP passes -----------===//

#include "Dialect/KTDP/Transforms/Utility.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::ktdp {

void cleanupDeadOps(ModuleOp module,
                    llvm::function_ref<bool(Operation *)> predicate) {
  module.walk([&](Block *block) {
    for (auto it = block->rbegin(); it != block->rend();) {
      Operation &op = *it++;
      if ((!predicate || predicate(&op)) && isOpTriviallyDead(&op))
        op.erase();
    }
  });
}

} // namespace mlir::triton::ktdp
