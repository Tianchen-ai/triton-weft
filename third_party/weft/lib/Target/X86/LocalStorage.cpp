#include "Weft/Target/X86Compiler.h"
#include "StorageAnalysis.h"
#include "Weft/Analysis/IntegerFacts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {
// This is a compiler allocation policy, not a claim about hardware capacity.
// Sum the reserved slots, rather than treating every allocation independently.
constexpr uint64_t stackBudget = weft::x86::privateStorageBudget;
constexpr uint64_t slotAlignment = 64;

Type storageBits(Type type) {
  if (type.isBF16()) return IntegerType::get(type.getContext(), 16);
  if (auto integer = dyn_cast<IntegerType>(type)) {
    unsigned width = llvm::PowerOf2Ceil(std::max<unsigned>(8, integer.getWidth()));
    if (width != integer.getWidth()) return IntegerType::get(type.getContext(), width);
  }
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.clone(storageBits(vector.getElementType()));
  return type;
}

Type bufferBits(Type type) {
  if (auto memref = dyn_cast<MemRefType>(type); memref && storageBits(memref.getElementType()) != memref.getElementType())
    return MemRefType::get(memref.getShape(), storageBits(memref.getElementType()),
                          memref.getLayout(), memref.getMemorySpace());
  return type;
}

// Array descriptors address individual native storage elements. A vector load
// from memref<i1/i4> must not pack several of those elements into one byte.
// Keep their exact bits in native integer carriers and restore the numerical
// width at uses. BF16 similarly uses I16 to avoid numeric conversion libcalls.
void legalizePayloadStorage(ModuleOp module, IRRewriter &b) {
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) { operations.push_back(op); });
  for (Operation *op : operations) {
    for (Value result : op->getResults()) result.setType(bufferBits(result.getType()));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments()) arg.setType(bufferBits(arg.getType()));
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      SmallVector<Type> inputs, results;
      for (Type type : function.getFunctionType().getInputs()) inputs.push_back(bufferBits(type));
      for (Type type : function.getFunctionType().getResults()) results.push_back(bufferBits(type));
      function.setType(b.getFunctionType(inputs, results));
    }
  }
  for (Operation *op : operations) {
    bool read = isa<memref::LoadOp, vector::LoadOp, vector::GatherOp, vector::MaskedLoadOp>(op);
    bool write = isa<memref::StoreOp, vector::StoreOp, vector::ScatterOp, vector::MaskedStoreOp>(op);
    if (!read && !write) continue;
    b.setInsertionPoint(op);
    // Only payload operands change representation. In particular an i1 mask
    // on a boolean load/store is still a predicate, not stored byte data.
    OpOperand *payload = nullptr;
    if (isa<memref::StoreOp, vector::StoreOp>(op)) payload = &op->getOpOperand(0);
    else if (isa<vector::MaskedLoadOp, vector::GatherOp,
                 vector::MaskedStoreOp, vector::ScatterOp>(op))
      payload = &op->getOpOperand(op->getNumOperands() - 1);
    if (payload) {
      Type original = payload->get().getType(), bits = storageBits(original);
      if (original != bits) {
        Type element = isa<VectorType>(original) ? cast<VectorType>(original).getElementType() : original;
        Value converted = element.isBF16() ? Value(arith::BitcastOp::create(b, op->getLoc(), bits, payload->get())) :
                                            Value(arith::ExtUIOp::create(b, op->getLoc(), bits, payload->get()));
        payload->set(converted);
      }
    }
    if (!read) continue;
    Value result = op->getResult(0);
    Type original = result.getType(), bits = storageBits(original);
    if (original == bits) continue;
    result.setType(bits);
    b.setInsertionPointAfter(op);
    Type element = isa<VectorType>(original) ? cast<VectorType>(original).getElementType() : original;
    Value restored = element.isBF16() ? Value(arith::BitcastOp::create(b, op->getLoc(), original, result)) :
                                      Value(arith::TruncIOp::create(b, op->getLoc(), original, result));
    result.replaceAllUsesExcept(restored, restored.getDefiningOp());
  }
}

std::optional<uint64_t> upperBound(Value value) {
  auto type = value.getType().isIndex() ?
      IntegerType::get(value.getContext(), 64, IntegerType::Unsigned) :
      weft::analysis::integerElementType(value.getType());
  if (!type) return {};
  if (type.isSignless()) type = IntegerType::get(type.getContext(), type.getWidth(), IntegerType::Unsigned);
  auto range = weft::analysis::integerRange(value, type);
  if (!range || range->minimum < 0 || range->maximum > stackBudget) return {};
  return range->maximum.getZExtValue();
}

struct Candidate {
  memref::AllocOp alloc;
  int64_t elements;
  uint64_t bytes;
  unsigned loopDepth;
  Block *scope;
  unsigned first, last;
};

struct StackSlot {
  Value storage;
  int64_t elements;
  Block *scope;
  SmallVector<std::pair<unsigned, unsigned>> intervals;
};

class MaterializeLocalStorage
    : public PassWrapper<MaterializeLocalStorage, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeLocalStorage)
  StringRef getArgument() const override { return "weft-x86-materialize-local-storage"; }
  StringRef getDescription() const override {
    return "Reserve bounded private snapshot slots once per call within an aggregate stack budget";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    legalizePayloadStorage(getOperation(), b);
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal()) continue;
      uint64_t reserved = 0;
      // Include already materialized stack snapshots. Unknown stack extents
      // prevent additional promotion instead of inventing a resource estimate.
      function.walk([&](LLVM::AllocaOp alloc) {
        auto count = upperBound(alloc.getArraySize());
        Type type = alloc.getElemType();
        uint64_t bytes = 0;
        if (auto vector = dyn_cast<VectorType>(type))
          bytes = vector.getNumElements() * llvm::divideCeil(vector.getElementTypeBitWidth(), 8u);
        else if (type.isIntOrFloat()) bytes = llvm::divideCeil(type.getIntOrFloatBitWidth(), 8u);
        if (!count || !bytes || *count > stackBudget / bytes) reserved = stackBudget;
        else reserved = std::min(stackBudget, reserved + llvm::alignTo(*count * bytes, slotAlignment));
      });
      function.walk([&](memref::AllocaOp alloc) {
        auto type = alloc.getType();
        if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat()) { reserved = stackBudget; return; }
        uint64_t bytes = type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
        reserved = std::min(stackBudget, reserved + llvm::alignTo(bytes, slotAlignment));
      });
      SmallVector<Candidate> candidates;
      DenseMap<Operation *, unsigned> positions;
      function.walk([&](Operation *op) {
        for (Region &region : op->getRegions())
          for (Block &block : region)
            for (auto [position, nested] : llvm::enumerate(block))
              positions[&nested] = position;
      });
      function.walk([&](memref::AllocOp alloc) {
        auto type = alloc.getType();
        if (type.getRank() != 1 || !type.getLayout().isIdentity() ||
            !type.getElementType().isIntOrFloat()) return;
        std::optional<uint64_t> count = type.hasStaticShape()
            ? std::optional<uint64_t>(type.getNumElements()) : upperBound(alloc.getDynamicSizes().front());
        uint64_t elementBytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
        if (!count || *count > stackBudget / elementBytes) return;
        auto uses = weft::x86::privateStorageUses(alloc);
        if (failed(uses)) return;
        unsigned first = positions.lookup(alloc), last = first;
        for (Operation *user : uses->users) {
          // A nested use occupies its complete parent region. Do not infer
          // disjoint lifetimes between allocations in different source blocks.
          while (user && user->getBlock() != alloc->getBlock()) user = user->getParentOp();
          if (!user || positions.lookup(user) < first) return;
          last = std::max(last, positions.lookup(user));
        }
        unsigned depth = 0;
        for (Operation *parent = alloc->getParentOp(); parent != function; parent = parent->getParentOp())
          depth += isa<scf::ForOp, scf::WhileOp>(parent);
        uint64_t bytes = llvm::alignTo(std::max(uint64_t(1), *count) * elementBytes, slotAlignment);
        candidates.push_back({alloc, int64_t(*count), bytes, depth, alloc->getBlock(), first, last});
      });
      // Prioritize repeated hot allocations, then smaller slots. Stable ties
      // preserve source order and give deterministic budget decisions.
      llvm::stable_sort(candidates, [](const Candidate &a, const Candidate &c) {
        if (a.loopDepth != c.loopDepth) return a.loopDepth > c.loopDepth;
        return a.bytes < c.bytes;
      });
      unsigned promoted = 0, reused = 0;
      SmallVector<StackSlot> slots;
      for (const auto &candidate : candidates) {
        auto alloc = candidate.alloc;
        auto type = alloc.getType();
        StackSlot *selected = nullptr;
        for (auto &slot : slots) {
          auto storageType = cast<MemRefType>(slot.storage.getType());
          if (slot.scope != candidate.scope || slot.elements < candidate.elements ||
              storageType.getElementType() != type.getElementType() ||
              storageType.getMemorySpace() != type.getMemorySpace()) continue;
          if (llvm::any_of(slot.intervals, [&](auto interval) {
                return candidate.first <= interval.second && interval.first <= candidate.last;
              })) continue;
          if (!selected || slot.elements < selected->elements) selected = &slot;
        }
        if (!selected) {
          if (candidate.bytes > stackBudget - reserved) continue;
          auto storageType = MemRefType::get({candidate.elements},
              type.getElementType(), MemRefLayoutAttrInterface{}, type.getMemorySpace());
          b.setInsertionPointToStart(&function.getBody().front());
          auto storage = memref::AllocaOp::create(b, alloc.getLoc(), storageType);
          storage.setAlignment(slotAlignment);
          slots.push_back({storage, candidate.elements, candidate.scope, {}});
          selected = &slots.back();
          reserved += candidate.bytes;
        } else {
          ++reused;
        }
        selected->intervals.emplace_back(candidate.first, candidate.last);
        b.setInsertionPoint(alloc);
        Value result = selected->storage;
        if (result.getType() != type) {
          OpFoldResult size = type.hasStaticShape() ? OpFoldResult(b.getIndexAttr(candidate.elements))
                                                   : OpFoldResult(alloc.getDynamicSizes().front());
          result = memref::ReinterpretCastOp::create(b, alloc.getLoc(), type, result,
              b.getIndexAttr(0), ArrayRef<OpFoldResult>{size},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        }
        b.replaceOp(alloc, result);
        ++promoted;
      }
      function->setAttr("weft.x86.stack_storage_bytes", b.getI64IntegerAttr(reserved));
      function->setAttr("weft.x86.stack_storage_budget", b.getI64IntegerAttr(stackBudget));
      function->setAttr("weft.x86.promoted_arrays", b.getI64IntegerAttr(promoted));
      function->setAttr("weft.x86.reused_stack_arrays", b.getI64IntegerAttr(reused));
    }
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86LocalStoragePass() {
  return std::make_unique<MaterializeLocalStorage>();
}
