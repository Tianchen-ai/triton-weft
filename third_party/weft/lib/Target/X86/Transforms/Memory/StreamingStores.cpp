#include "Weft/Target/X86/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace {
BlockArgument entryPointer(Value value, func::FuncOp function) {
  while (auto gep = value.getDefiningOp<LLVM::GEPOp>()) value = gep.getBase();
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.getBody().front() ? argument : BlockArgument();
}

bool closedOutput(func::FuncOp function, BlockArgument output) {
  auto bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
  if (!bits) return false;
  SmallVector<Value> pending{output};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!pending.empty()) {
    Value pointer = pending.pop_back_val();
    for (Operation *user : pointer.getUsers()) {
      if (!visited.insert(user).second) continue;
      if (auto gep = dyn_cast<LLVM::GEPOp>(user); gep && gep.getBase() == pointer) {
        pending.push_back(gep.getResult());
        continue;
      }
      auto store = dyn_cast<LLVM::MaskedStoreOp>(user);
      if (!store || store.getData() != pointer) return false;
      auto type = dyn_cast<VectorType>(store.getValue().getType());
      if (!store->hasAttr("weft.x86.streaming") || !type || type.getRank() != 1 || type.isScalable() ||
          !type.getElementType().isF32() || type.getNumElements() < 4 || type.getNumElements() % 4) return false;
    }
  }
  return true;
}

class StreamingStores : public PassWrapper<StreamingStores, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamingStores)
  StringRef getArgument() const override { return "weft-x86-select-streaming-stores"; }
  StringRef getDescription() const override {
    return "Select aligned native streaming packets for disjoint write-only output hints";
  }
  void runOnOperation() override {
    SmallVector<LLVM::MaskedStoreOp> stores;
    getOperation().walk([&](LLVM::MaskedStoreOp store) {
      if (store->hasAttr("weft.x86.streaming")) stores.push_back(store);
    });
    llvm::SmallPtrSet<Operation *, 8> fenced;
    DenseMap<Value, bool> eligibleOutputs;
    int64_t selected = 0;
    for (auto store : stores) {
      auto function = store->getParentOfType<func::FuncOp>();
      auto argument = entryPointer(store.getData(), function);
      // The intrinsic calls its address `data`; operand zero is the payload.
      if (!argument || !function.getArgAttr(argument.getArgNumber(), "weft.x86.write_only") ||
          !function.getArgAttr(argument.getArgNumber(), LLVM::LLVMDialect::getNoAliasAttrName())) continue;
      auto found = eligibleOutputs.find(argument);
      if (found == eligibleOutputs.end()) found = eligibleOutputs.try_emplace(argument, closedOutput(function, argument)).first;
      if (!found->second) continue;
      auto type = dyn_cast<VectorType>(store.getValue().getType());
      auto bits = function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits");
      if (!type || !bits || type.getRank() != 1 || type.isScalable() || !type.getElementType().isF32()) continue;
      if (type.getNumElements() < 4 || type.getNumElements() % 4) continue;
      OpBuilder b(store);
      Location loc = store.getLoc();
      Value address = store.getData();
      Value integer = LLVM::PtrToIntOp::create(b, loc, b.getI64Type(), address);
      Value active = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, store.getMask());
      // A checked F32 View promises element alignment, not AVX-512 alignment.
      // Select the widest complete native packet that its actual address admits.
      for (int64_t lanes = bits.getInt() / 32; lanes >= 4; lanes /= 2) {
        if (type.getNumElements() % lanes) continue;
        int64_t alignment = lanes * 4;
        Value low = arith::AndIOp::create(b, loc, integer, arith::ConstantIntOp::create(b, loc, alignment-1, 64));
        Value aligned = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, low,
            arith::ConstantIntOp::create(b, loc, 0, 64));
        auto branch = scf::IfOp::create(b, loc, TypeRange{}, arith::AndIOp::create(b, loc, aligned, active), true);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        for (int64_t offset = 0; offset < type.getNumElements(); offset += lanes) {
          Value packet = vector::ExtractStridedSliceOp::create(b, loc, store.getValue(),
              ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{lanes}, ArrayRef<int64_t>{1});
          Value pointer = LLVM::GEPOp::create(b, loc, address.getType(), b.getF32Type(), address,
              ArrayRef<LLVM::GEPArg>{offset});
          auto write = LLVM::StoreOp::create(b, loc, packet, pointer);
          write.setAlignment(alignment);
          write.setNontemporal(true);
        }
        b.setInsertionPointToStart(&branch.getElseRegion().front());
      }
      // A masked or unaligned tail may overlap an earlier complete write.
      // Order pending streaming packets before using ordinary cacheable stores.
      func::CallOp::create(b, loc, "llvm.x86.sse.sfence", TypeRange{}, ValueRange{});
      Operation *ordinary = b.clone(*store);
      ordinary->removeAttr("weft.x86.streaming");
      store.erase();
      fenced.insert(function);
      ++selected;
    }
    if (fenced.empty()) return;
    OpBuilder b(getOperation().getContext());
    b.setInsertionPointToEnd(getOperation().getBody());
    auto fence = getOperation().lookupSymbol<func::FuncOp>("llvm.x86.sse.sfence");
    if (!fence) {
      fence = func::FuncOp::create(b, getOperation().getLoc(), "llvm.x86.sse.sfence", b.getFunctionType({}, {}));
      fence.setPrivate();
    }
    for (Operation *function : fenced)
      function->walk([&](func::ReturnOp ret) {
        b.setInsertionPoint(ret);
        func::CallOp::create(b, ret.getLoc(), fence, ValueRange{});
      });
    getOperation()->setAttr("weft.x86.streaming_stores", b.getI64IntegerAttr(selected));
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86StreamingStorePass() { return std::make_unique<StreamingStores>(); }
