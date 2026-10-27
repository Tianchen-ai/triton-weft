#ifndef WEFT_TARGET_X86COMPILER_H
#define WEFT_TARGET_X86COMPILER_H
#include "Weft/Target/X86/X86Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Pass/Pass.h"

namespace weft {
void loadX86Dialects(mlir::MLIRContext &context);
void registerX86Dialects(mlir::DialectRegistry &registry);
void registerX86Passes();
std::unique_ptr<mlir::Pass> createX86MathPass();
std::unique_ptr<mlir::Pass> createX86HalfCarrierPass();
std::unique_ptr<mlir::Pass> createX86VectorProjectionPass();
std::unique_ptr<mlir::Pass> createX86ProjectionLoopPass();
std::unique_ptr<mlir::Pass> createX86IntegerReductionPass();
std::unique_ptr<mlir::Pass> createX86IntegerProductPass();
std::unique_ptr<mlir::Pass> createX86SnapshotLoadPass();
std::unique_ptr<mlir::Pass> createX86SubbyteProjectionPass(bool selectedConsumersOnly = false);
std::unique_ptr<mlir::Pass> createX86IntegerProjectionPass(bool selectedConsumersOnly = false);
std::unique_ptr<mlir::Pass> createX86LookupProjectionPass();
std::unique_ptr<mlir::Pass> createX86IntegerPacketMergePass();
std::unique_ptr<mlir::Pass> createX86FullTilePass();
std::unique_ptr<mlir::Pass> createX86VectorCarryPass();
std::unique_ptr<mlir::Pass> createX86ArrayPacketPass();
std::unique_ptr<mlir::Pass> createX86ArrayForwardingPass();
std::unique_ptr<mlir::Pass> createX86ReductionSchedulePass();
std::unique_ptr<mlir::Pass> createX86UniformValidityPass();
std::unique_ptr<mlir::Pass> createX86IndexedValuePass();
std::unique_ptr<mlir::Pass> createX86IndexedPacketPass();
std::unique_ptr<mlir::Pass> createX86ContractionSchedulePass();
std::unique_ptr<mlir::Pass> createX86ReductionTreePass();
std::unique_ptr<mlir::Pass> createX86LocalStoragePass();
std::unique_ptr<mlir::Pass> createX86StreamingStorePass();
std::unique_ptr<mlir::Pass> createX86StreamPrefetchPass();
std::unique_ptr<mlir::Pass> createX86PipelineSchedulePass(unsigned depth = 1);
std::unique_ptr<mlir::Pass> createX86PipelinePass();
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> lowerX86Module(
    mlir::ModuleOp source, const X86Target &target,
    const llvm::StringMap<int64_t> &bindings, llvm::StringRef tuneCPU = {}, unsigned pipelineDepth = 1);
mlir::FailureOr<std::string> translateX86Module(mlir::ModuleOp physical);
} // namespace weft
#endif
