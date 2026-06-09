//===- ConvertFunctions.cpp - Convert tt.func/return to func dialect ------===//
//
// Converts Triton function-level ops to standard func dialect and finalizes
// !tt.ptr function arguments to index type.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_CONVERTFUNCTIONS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// tt.return -> func.return
struct ConvertTTReturn : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ConvertFunctionsPass
    : public mlir::triton::ktdp::impl::ConvertFunctionsBase<
          ConvertFunctionsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    convertFunctions(module);

    ConversionTarget target(*ctx);
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<triton::ReturnOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertTTReturn>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("ConvertFunctions: failed to convert tt.return ops");
      signalPassFailure();
      return;
    }

    finalizeFunctionSignatures(module);
  }

private:
  void convertFunctions(ModuleOp module) {
    OpBuilder builder(module.getContext());

    SmallVector<triton::FuncOp> ttFuncs;
    module.walk([&](triton::FuncOp funcOp) { ttFuncs.push_back(funcOp); });

    for (auto ttFunc : ttFuncs) {
      auto funcType = ttFunc.getFunctionType();

      builder.setInsertionPoint(ttFunc);
      auto funcOp = func::FuncOp::create(builder, ttFunc.getLoc(),
                                          ttFunc.getName(), funcType);

      if (ttFunc.isPublic())
        funcOp.setVisibility(SymbolTable::Visibility::Public);

      Region &oldRegion = ttFunc.getBody();
      Region &newRegion = funcOp.getBody();

      Block *newEntry = new Block();
      newRegion.push_back(newEntry);
      for (Type t : funcType.getInputs())
        newEntry->addArgument(t, ttFunc.getLoc());

      IRMapping mapping;
      Block &oldEntry = oldRegion.front();
      for (unsigned i = 0; i < oldEntry.getNumArguments(); ++i)
        mapping.map(oldEntry.getArgument(i), newEntry->getArgument(i));

      builder.setInsertionPointToStart(newEntry);
      for (auto &op : oldEntry.getOperations())
        builder.clone(op, mapping);

      ttFunc.erase();
    }
  }

  void finalizeFunctionSignatures(ModuleOp module) {
    module.walk([&](func::FuncOp funcOp) {
      Block &entry = funcOp.getBody().front();
      OpBuilder builder(funcOp.getContext());

      bool changed = false;
      SmallVector<Type> newArgTypes;

      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (isa<triton::PointerType>(arg.getType())) {
          newArgTypes.push_back(builder.getIndexType());
          changed = true;
        } else {
          newArgTypes.push_back(arg.getType());
        }
      }

      if (!changed)
        return;

      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (!isa<triton::PointerType>(arg.getType()))
          continue;

        SmallVector<UnrealizedConversionCastOp> casts;
        for (auto *user : arg.getUsers())
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            casts.push_back(cast);

        arg.setType(builder.getIndexType());

        for (auto cast : casts) {
          cast.getResult(0).replaceAllUsesWith(arg);
          cast.erase();
        }
      }

      auto newFuncType = FunctionType::get(
          funcOp.getContext(), newArgTypes,
          funcOp.getFunctionType().getResults());
      funcOp.setType(newFuncType);
    });
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass() {
  return std::make_unique<ConvertFunctionsPass>();
}
} // namespace mlir::triton::ktdp
