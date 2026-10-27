#include "Weft/Target/X86/X86Compiler.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;
namespace wk = weft::kernel;
namespace {
// The SDK's masked LLVM transfers do not expose MemoryEffectOpInterface. Their
// read/write effects must stay visible when array ownership is analyzed beside
// fixed SIMD operations; neither operation allocates or releases a local buffer.
template <typename Op, bool Read>
struct TransferEffects : MemoryEffectOpInterface::ExternalModel<TransferEffects<Op, Read>, Op> {
  void getEffects(Operation *, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    if constexpr (Read) effects.emplace_back(MemoryEffects::Read::get());
    else effects.emplace_back(MemoryEffects::Write::get());
  }
};

}

void weft::registerX86Dialects(DialectRegistry &registry) {
  registry.insert<wk::WEFTKernelDialect, arith::ArithDialect, func::FuncDialect,
                  scf::SCFDialect, cf::ControlFlowDialect, vector::VectorDialect,
                  math::MathDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  bufferization::BufferizationDialect, ub::UBDialect>();
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::MaskedLoadOp::attachInterface<TransferEffects<LLVM::MaskedLoadOp, true>>(*context);
    LLVM::masked_gather::attachInterface<TransferEffects<LLVM::masked_gather, true>>(*context);
    LLVM::MaskedStoreOp::attachInterface<TransferEffects<LLVM::MaskedStoreOp, false>>(*context);
    LLVM::masked_scatter::attachInterface<TransferEffects<LLVM::masked_scatter, false>>(*context);
  });
}

void weft::loadX86Dialects(MLIRContext &context) {
  DialectRegistry registry;
  registerX86Dialects(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<wk::WEFTKernelDialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect, cf::ControlFlowDialect, vector::VectorDialect,
                      math::MathDialect, LLVM::LLVMDialect, memref::MemRefDialect, bufferization::BufferizationDialect>();
  registerLLVMDialectTranslation(context);
}
