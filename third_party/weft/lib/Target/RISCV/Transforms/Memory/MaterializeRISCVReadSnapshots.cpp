#include "Weft/Target/RISCV/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/MemoryEffects.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"

#include "Transforms/Support/RISCVPhysicalSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>
#include <memory>

using namespace weft;

namespace {

int64_t ownerDomain(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto level =
            parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      return level.getDomainId();
  return 0;
}

// A fixed private record also establishes an alias-independent supply for its
// vector consumers. The C toolchain cannot carry a conditional disjointness
// proof through a selected pointer: borrowing can serialize those vector loads
// with output stores and prevent issue scheduling. Retain that existing local
// supply until its native ordering cost can be represented. Scalar projections
// already freeze their values in explicit SSA and do not need this carrier.
bool hasVectorStorageConsumer(mlir::Value value) {
  llvm::SmallVector<mlir::Value> pending{value};
  llvm::SmallPtrSet<void *, 32> visited;
  while (!pending.empty()) {
    mlir::Value current = pending.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second) continue;
    if (visited.size() > 256) return true;
    for (mlir::OpOperand &use : current.getUses()) {
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(use.getOwner());
      if (effects) {
        if (effects.getEffectOnValue<mlir::MemoryEffects::Read>(current) &&
            llvm::any_of(use.getOwner()->getResultTypes(), [](mlir::Type type) {
              auto value = mlir::dyn_cast<riscv::ValueType>(type);
              return mlir::isa<riscv::FragmentType>(type) ||
                     (value && value.getLayout().getCarrier() == "rvv");
            })) return true;
      }
      riscv::appendStorageAliases(use, pending);
    }
  }
  return false;
}

void isolateSnapshotComputation(riscv::KernelOp kernel, mlir::IRRewriter &b) {
  llvm::SmallVector<mlir::Operation *> scopes;
  kernel.walk([&](riscv::LoadOp load) {
    auto storage = load.getSnapshotStorage();
    auto local = storage ? mlir::dyn_cast<riscv::LocalType>(storage.getType()) : riscv::LocalType();
    if (!local || local.getSizeBytes() >= 0) return;
    llvm::SmallVector<mlir::Value> pending{load.getResult()};
    llvm::SmallPtrSet<void *, 32> visited;
    llvm::SmallPtrSet<mlir::Operation *, 8> consumers;
    while (!pending.empty()) {
      mlir::Value value = pending.pop_back_val();
      if (!visited.insert(value.getAsOpaquePointer()).second) continue;
      for (mlir::OpOperand &use : value.getUses()) {
        auto root = load->getBlock()->findAncestorOpInBlock(*use.getOwner());
        if (root) consumers.insert(root);
        riscv::appendStorageAliases(use, pending);
      }
    }
    for (mlir::Operation &operation : *load->getBlock()) {
      if (!consumers.contains(&operation) ||
          !mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp>(operation) ||
          !load->isBeforeInBlock(&operation) || llvm::is_contained(scopes, &operation)) continue;
      bool dynamicStorage = false, returns = false, vectorWork = false;
      operation.walk([&](mlir::Operation *nested) {
        if (auto allocation = mlir::dyn_cast<riscv::LocalAllocOp>(nested))
          dynamicStorage |= allocation.getResult().getType().getSizeBytes() < 0;
        returns |= mlir::isa<riscv::HelperReturnOp>(nested);
        auto leaf = nested->getAttrOfType<riscv::LeafAttr>("leaf");
        vectorWork |= leaf && (leaf.getEngine() == "rvv" || leaf.getEngine() == "ime");
      });
      if (!dynamicStorage && !returns && vectorWork) {
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
          // Leave the domain dispatch next to the snapshot. Each selected
          // traversal then has its own frame and only its own captures; the
          // complete-domain body need not carry the masked alternate's state.
          for (auto &region : branch->getRegions())
            for (auto &block : region)
              for (auto &nested : block)
                if (mlir::isa<mlir::scf::ForOp>(nested) &&
                    !llvm::is_contained(scopes, &nested)) scopes.push_back(&nested);
        } else scopes.push_back(&operation);
      }
    }
  });
  for (mlir::Operation *scope : scopes) {
    // Outlining must fit the already supported helper ABI. Keep result-bearing
    // traversals in place until their return carriers have a selected transfer.
    if (scope->getNumResults()) continue;
    llvm::SetVector<mlir::Value> captures;
    for (auto &region : scope->getRegions()) mlir::getUsedValuesDefinedAbove(region, captures);
    if (llvm::any_of(captures, [](mlir::Value value) {
          return !mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
              riscv::ValueType, riscv::MemDescType, riscv::FragmentType,
              riscv::PointType, riscv::DomainType>(value.getType());
        })) continue;
    // Keep the allocation, optional copy and borrowed pointer at their original
    // read point. Only the already ordered consumer region becomes a closure,
    // so its native frame does not inherit the parent's dynamic stack object.
    // Moving instead of cloning preserves partial births and buffer identities.
    b.setInsertionPoint(scope);
    auto helper = riscv::HelperOp::create(b, scope->getLoc(), scope->getResultTypes(),
        mlir::ValueRange{}, b.getStringAttr("snapshot_compute"), b.getBoolAttr(true));
    riscv_internal::copyOrigin(scope, helper);
    helper.getBody().emplaceBlock();
    scope->replaceAllUsesWith(helper.getResults());
    scope->moveBefore(&helper.getBody().front(), helper.getBody().front().end());
    b.setInsertionPointAfter(scope);
    riscv::HelperReturnOp::create(b, scope->getLoc(), scope->getResults());
  }
}

class MaterializeRISCVReadSnapshotsPass
    : public mlir::PassWrapper<MaterializeRISCVReadSnapshotsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVReadSnapshotsPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-read-snapshots";
  }
  llvm::StringRef getDescription() const override {
    return "Preserve encoded read values across interfering memory effects";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    int64_t identity = 0;
    getOperation().walk([&](riscv::KernelOp kernel) {
      for (int64_t aliasSet : kernel.getArgAliasSets())
        identity = std::max(identity, aliasSet + 1);
    });
    getOperation().walk([&](riscv::LocalAllocOp allocation) {
      auto type = allocation.getResult().getType();
      identity = std::max({identity, type.getAliasSet() + 1,
                           type.getBirthId() + 1});
    });
    bool failed = false;
    getOperation().walk([&](riscv::LoadOp load) {
      if (load.getSnapshotStorage() ||
          !riscv_internal::needsReadSnapshot(load))
        return;
      auto kernel = load->getParentOfType<riscv::KernelOp>();
      auto memory = load.getRegion().getType();
      auto value = mlir::dyn_cast<riscv::ValueType>(load.getResult().getType());
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      const int64_t extent = value && value.getShape().size() == 1 ? value.getShape()[0] : 0;
      const int64_t recordBytes = memory.getStorageBits() / 8;
      if (!kernel || kernel.getResourcesMaterialized() || !value ||
          value.getLayout().getCarrier() != "local" ||
          value.getLayout().getValidity() != "full" ||
          encoding.getKind() != "base" ||
          !riscv::isContiguousRecordRegion(load.getRegion())) {
        load.emitError("interfering encoded read requires a contiguous full-record snapshot");
        failed = true;
        return;
      }
      int64_t bytes = -1;
      auto bound = memory.getShape().size() == 1 ? riscv::denseDescriptorExtent(load.getRegion(), 0) : mlir::OpFoldResult();
      auto fixed = mlir::dyn_cast_if_present<mlir::IntegerAttr>(bound.dyn_cast<mlir::Attribute>());
      if (fixed && fixed.getInt() == extent && extent > 0 && extent % memory.getElements() == 0) {
        if (extent / memory.getElements() > std::numeric_limits<int64_t>::max() / recordBytes) {
          load.emitError("encoded read snapshot size overflows"); failed = true; return;
        }
        bytes = extent / memory.getElements() * recordBytes;
      }
      // A borrow proof controls the transfer, independently of storage size.
      // Fixed snapshots remain fixed allocations even when borrowing is legal.
      const bool fixedSnapshot = bytes > 0 && bytes <= kernel.getTarget().getMaxPrivateStackBytes();
      mlir::Value borrow;
      auto writes = riscv::snapshotInterferingStorage(load.getRegion(), load);
      mlir::Value source = riscv::fullArgumentView(load.getRegion());
      llvm::SmallVector<mlir::Value> destinations;
      if ((!fixedSnapshot || !hasVectorStorageConsumer(load.getResult())) &&
          mlir::succeeded(writes) && source &&
          riscv::storageAliasesRemainInScope(load.getResult(), load->getBlock())) {
        for (auto write : *writes) {
          auto full = riscv::fullArgumentView(write);
          if (!full || full == source) { destinations.clear(); break; }
          if (!llvm::is_contained(destinations, full)) destinations.push_back(full);
        }
      }
      for (mlir::Value destination : destinations) {
        mlir::Operation *last = source.getDefiningOp();
        if (last->isBeforeInBlock(destination.getDefiningOp())) last = destination.getDefiningOp();
        rewriter.setInsertionPointAfter(last);
        mlir::Value disjoint = riscv::StorageDisjointOp::create(rewriter, load.getLoc(),
            rewriter.getI1Type(), source, destination);
        // The proofs are immutable ABI metadata and stay outside the author's
        // traversal. The actual snapshot remains at its original read point.
        rewriter.setInsertionPoint(load);
        borrow = borrow ? mlir::Value(mlir::arith::AndIOp::create(rewriter, load.getLoc(), borrow, disjoint)) : disjoint;
      }
      if (borrow && !fixedSnapshot) bytes = -1;
      if (bytes > kernel.getTarget().getMaxPrivateStackBytes()) {
        load.emitError("encoded read snapshot exceeds target local storage");
        failed = true;
        return;
      }
      const int64_t owner = ownerDomain(load);
      rewriter.setInsertionPoint(load);
      mlir::Value byteCount;
      if (bytes > 0) {
        byteCount = mlir::arith::ConstantIndexOp::create(rewriter, load.getLoc(), bytes);
      } else {
        llvm::SmallVector<mlir::Value> counts;
        for (unsigned axis = 0; axis < memory.getShape().size(); ++axis)
          counts.push_back(rewriter.createOrFold<riscv::ExtentOp>(load.getLoc(), rewriter.getIndexType(), load.getRegion(), axis));
        auto elements = mlir::arith::ConstantIndexOp::create(rewriter, load.getLoc(), memory.getElements());
        counts.back() = mlir::arith::DivUIOp::create(rewriter, load.getLoc(), counts.back(), elements);
        byteCount = riscv::LocalSizeOp::create(rewriter, load.getLoc(), rewriter.getIndexType(), counts, recordBytes,
            riscv_internal::leaf(rewriter, "scalar", "local-size", "local.size.checked", "local.size.checked", 0, 0));
        if (borrow) {
          auto zero = mlir::arith::ConstantIndexOp::create(rewriter, load.getLoc(), 0);
          byteCount = mlir::arith::SelectOp::create(rewriter, load.getLoc(), borrow, zero, byteCount);
        }
        // A zero-byte logical object needs no transfer. Keep the C allocation
        // defined while retaining its true transfer size in explicit SSA.
        byteCount = mlir::arith::MaxUIOp::create(rewriter, load.getLoc(), byteCount,
            mlir::arith::ConstantIndexOp::create(rewriter, load.getLoc(), 1));
      }
      auto storageType = riscv::LocalType::get(
          &getContext(), rewriter.getIntegerType(8),
          riscv_internal::integers(rewriter, {}),
          riscv_internal::integers(rewriter, {}), bytes, memory.getAlignment(),
          identity, "pack", owner, identity, owner, "read-snapshot");
      ++identity;
      auto storage = riscv::LocalAllocOp::create(rewriter,
          load.getLoc(), storageType, byteCount);
      riscv_internal::copyOrigin(load, storage);
      load.getSnapshotStorageMutable().assign(storage.getResult());
      load.getSnapshotBorrowMutable().assign(borrow ? mlir::ValueRange{borrow} : mlir::ValueRange{});
      load.setLeafAttr(
          riscv_internal::readSnapshotLeaf(rewriter, kernel.getTarget(), bytes > 0 ? bytes : kernel.getTarget().getVlenBits() / 8));
    });
    if (!failed)
      for (auto kernel : getOperation().getOps<riscv::KernelOp>())
        isolateSnapshotComputation(kernel, rewriter);
    if (failed)
      signalPassFailure();
  }
};

} // namespace

bool weft::riscv_internal::needsReadSnapshot(riscv::LoadOp load) {
  auto encoding =
      mlir::cast<kernel::EncodingType>(load.getRegion().getType().getEncoding());
  if (encoding.getKind() == "dense")
    return false;
  llvm::SmallVector<mlir::Value> pending{load.getResult()};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (riscv::storageWrittenBetween(load.getRegion(), load, user)) {
        return true;
      }
      if (mlir::isMemoryEffectFree(user) &&
          !mlir::isa<riscv::EncodedLocalBindOp>(user))
        for (mlir::Value result : user->getResults())
          if (!riscv::storageOrigins(result).isNumeric()) pending.push_back(result);
    }
  }
  return false;
}

std::unique_ptr<mlir::Pass> weft::createMaterializeRISCVReadSnapshotsPass() {
  return std::make_unique<MaterializeRISCVReadSnapshotsPass>();
}
