#include "Weft/Target/X86/X86Compiler.h"
#include "Analysis/Resources.h"
#include "Analysis/StorageAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace {

struct LaneUpdate {
  LLVM::InsertElementOp insert;
  scf::YieldOp commit;
};

// A loop iteration either carries the old vector or changes a single lane.
// Commit the store at the yield, after all reads of the original SSA value.
// Other consumers of the updated vector require an independent snapshot.
bool matchLeaf(Value next, Value carry, scf::YieldOp yield,
               SmallVectorImpl<LaneUpdate> &updates) {
  if (next == carry) return true;
  auto insert = next.getDefiningOp<LLVM::InsertElementOp>();
  if (!insert || insert.getVector() != carry || !next.hasOneUse() ||
      insert->getBlock() != yield->getBlock())
    return false;
  updates.push_back({insert, yield});
  return true;
}

bool materialize(scf::ForOp loop, unsigned position, IRRewriter &b,
                 uint64_t &remainingBytes) {
  Value carry = loop.getRegionIterArgs()[position];
  auto type = dyn_cast<VectorType>(carry.getType());
  if (!type || type.isScalable() || type.getRank() != 1 ||
      !type.getElementType().isIntOrFloat()) return false;
  for (Operation *parent = loop->getParentOp(); !isa<func::FuncOp>(parent);
       parent = parent->getParentOp())
    if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent)) return false;
  unsigned bits = type.getElementTypeBitWidth();
  if (bits != 1 && (bits < 8 || bits % 8)) return false;
  uint64_t bytes = type.getNumElements() * uint64_t(std::max(1u, bits / 8));
  if (bytes > remainingBytes || bytes * 8 <= weft::x86::ResourceModel::get(loop).vectorBits)
    return false;

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  Value next = yield.getOperand(position);
  SmallVector<LaneUpdate> updates;
  Operation *commit = yield;
  SmallVector<OpOperand *> unchanged;
  if (auto branch = next.getDefiningOp<scf::IfOp>()) {
    if (branch->getBlock() != loop.getBody() || !next.hasOneUse()) return false;
    unsigned index = cast<OpResult>(next).getResultNumber();
    for (auto arm : {branch.thenYield(), branch.elseYield()}) {
      if (!matchLeaf(arm.getOperand(index), carry, arm, updates)) return false;
      if (arm.getOperand(index) == carry) unchanged.push_back(&arm->getOpOperand(index));
    }
    commit = branch;
  } else if (!matchLeaf(next, carry, yield, updates)) return false;
  if (!llvm::any_of(updates, [](const LaneUpdate &update) {
        return !matchPattern(update.insert->getOperand(2), m_Constant());
      })) return false;

  SmallVector<LLVM::ExtractElementOp> reads;
  for (OpOperand &use : carry.getUses()) {
    if (llvm::is_contained(unchanged, &use)) continue;
    if (llvm::any_of(updates, [&](const LaneUpdate &update) {
          return update.insert == use.getOwner() && use.getOperandNumber() == 0;
        })) continue;
    auto read = dyn_cast<LLVM::ExtractElementOp>(use.getOwner());
    if (!read) return false;
    Operation *enclosing = loop.getBody()->findAncestorOpInBlock(*read);
    if (!enclosing || (enclosing != commit && !enclosing->isBeforeInBlock(commit))) return false;
    reads.push_back(read);
  }

  Location loc = loop.getLoc();
  Type storedElement = bits == 1 ? Type(b.getI8Type()) : type.getElementType();
  auto storedVector = type.clone(storedElement);
  b.setInsertionPointToStart(&loop->getParentOfType<func::FuncOp>().getBody().front());
  auto buffer = memref::AllocaOp::create(b, loc, MemRefType::get(type.getShape(), storedElement));
  remainingBytes -= bytes;
  b.setInsertionPoint(loop);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value initial = loop.getInitArgs()[position];
  if (bits == 1) initial = arith::ExtUIOp::create(b, loc, storedVector, initial);
  vector::StoreOp::create(b, loc, initial, buffer, ValueRange{zero});

  // Out-of-range LLVM lane operations yield poison. Keep the private memory
  // access in bounds even when computing a value that is subsequently masked.
  auto safeIndex = [&](Value lane) -> Value {
    Value index = arith::IndexCastUIOp::create(b, loc, b.getIndexType(), lane);
    Value extent = arith::ConstantIndexOp::create(b, loc, type.getNumElements());
    Value inBounds = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, index, extent);
    return arith::SelectOp::create(b, loc, inBounds, index, zero);
  };
  for (auto read : reads) {
    b.setInsertionPoint(read);
    Value scalar = memref::LoadOp::create(b, loc, buffer, safeIndex(read.getPosition()));
    if (bits == 1) scalar = arith::TruncIOp::create(b, loc, b.getI1Type(), scalar);
    b.replaceOp(read, scalar);
  }
  for (auto [insert, edge] : updates) {
    b.setInsertionPoint(edge);
    Value scalar = insert.getValue();
    if (bits == 1) scalar = arith::ExtUIOp::create(b, loc, b.getI8Type(), scalar);
    memref::StoreOp::create(b, loc, scalar, buffer, safeIndex(insert.getPosition()));
    b.replaceOp(insert, carry);
  }
  yield->setOperand(position, carry);
  b.setInsertionPointAfter(loop);
  Value final = vector::LoadOp::create(b, loc, storedVector, buffer, ValueRange{zero});
  if (bits == 1) final = arith::TruncIOp::create(b, loc, type, final);
  loop.getResult(position).replaceAllUsesWith(final);
  return true;
}

class IndexedLoopState : public PassWrapper<IndexedLoopState, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IndexedLoopState)
  StringRef getArgument() const override { return "weft-x86-materialize-indexed-loop-state"; }
  StringRef getDescription() const override {
    return "Carry wide, individually updated lanes in bounded private storage";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      uint64_t remaining = weft::x86::privateStorageBudget;
      function.walk([&](scf::ForOp loop) {
        for (unsigned i = 0; i < loop.getNumResults(); ++i) materialize(loop, i, b, remaining);
      });
    }
  }
};
}

std::unique_ptr<Pass> weft::createX86IndexedLoopStatePass() {
  return std::make_unique<IndexedLoopState>();
}
