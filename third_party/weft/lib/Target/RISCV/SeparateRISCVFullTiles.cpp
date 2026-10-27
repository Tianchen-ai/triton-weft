#include "Weft/Target/FullTiles.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "RISCVPhysicalSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace {
int64_t guardedFieldPayloads(Operation *scope) {
  int64_t guarded = 0;
  scope->walk([&](weft::riscv::FieldOp field) {
    if (field.getResult().use_empty()) return;
    auto owner = dyn_cast<weft::riscv::ValueType>(field.getOwner().getType());
    if (!owner) return;
    for (auto [extent, axis] : llvm::zip(owner.getShape().asArrayRef(), owner.getAxisIds().asArrayRef())) {
      if (extent <= 0) continue;
      auto point = weft::riscv_internal::originPoint(field.getOwner(), axis);
      APInt partition, active;
      if (!point || !matchPattern(point.getPartition(), m_ConstantInt(&partition)) ||
          partition.isNegative() || partition.getLimitedValue() != uint64_t(extent)) continue;
      // Encoded transfers spell their record guards after loop specialization.
      // A dynamic active coordinate is real guarded payload work even before
      // the selected field read becomes an explicit branch or RVV mask.
      if (!matchPattern(point.getActive(), m_ConstantInt(&active))) ++guarded;
    }
  });
  return guarded;
}

class SeparateFullTiles : public PassWrapper<SeparateFullTiles, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeparateFullTiles)
  StringRef getArgument() const override { return "weft-riscv-separate-full-tiles"; }
  StringRef getDescription() const override {
    return "Prove complete physical iterations while retaining ordered Level tails and carries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](weft::riscv::KernelOp kernel) {
      if (kernel.getResourcesMaterialized()) {
        kernel.emitError("complete-tile separation requires open physical resources");
        invalid = true; return;
      }
      llvm::SmallPtrSet<Operation *, 16> snapshotConsumers;
      kernel.walk([&](weft::riscv::LoadOp load) {
        auto type = dyn_cast<weft::riscv::ValueType>(load.getResult().getType());
        if (!type ||
            !llvm::any_of(type.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; }) ||
            !weft::riscv_internal::needsReadSnapshot(load)) return;
        llvm::SmallVector<Value> pending{load.getResult()};
        llvm::DenseSet<Value> visited;
        while (!pending.empty()) {
          Value value = pending.pop_back_val();
          if (!visited.insert(value).second) continue;
          for (OpOperand &use : value.getUses()) {
            for (Operation *scope = use.getOwner(); scope && scope != kernel;
                 scope = scope->getParentOp())
              if (isa<scf::ForOp>(scope)) snapshotConsumers.insert(scope);
            weft::riscv::appendStorageAliases(use, pending);
          }
        }
      });
      llvm::SmallPtrSet<Operation *, 16> pipelineConsumers;
      llvm::SmallPtrSet<Operation *, 16> flatConsumers;
      kernel.walk([&](scf::ForOp loop) {
        auto schedule = loop->getAttrOfType<weft::riscv::ScheduleAttr>("weft.riscv.schedule");
        if (loop.getNumRegionIterArgs() == 0 || guardedFieldPayloads(loop) == 0) return;
        bool pipelined = schedule && schedule.getPipelineDepth() > 1;
        bool nested = false;
        loop.getBody()->walk([&](Operation *operation) {
          nested |= isa<scf::ForOp, scf::WhileOp>(operation);
        });
        bool flat = !nested && llvm::all_of(loop.getRegionIterArgs(), [](Value value) {
          return isa_and_nonnull<FloatType>(weft::riscv_internal::logicalElement(value.getType()));
        });
        if (!pipelined && !flat) return;
        // Close the independent input bounds before scheduling a pipeline or
        // a flat floating recurrence. Nested numerical recurrences retain the
        // smaller scope; their intermediate state has a separate lifetime.
        // Only existing ancestors can own the complete domain.
        for (Operation *scope = loop; scope && scope != kernel;
             scope = scope->getParentOp())
          if (isa<scf::ForOp>(scope)) {
            if (pipelined) pipelineConsumers.insert(scope);
            if (flat) flatConsumers.insert(scope);
          }
      });
      auto growth = weft::separateFullTiles(kernel, true, guardedFieldPayloads,
          [&](Operation *scope) {
            return snapshotConsumers.contains(scope) || pipelineConsumers.contains(scope) || flatConsumers.contains(scope);
          });
      if (failed(growth)) { invalid = true; return; }
      Builder b(&getContext());
      for (Operation *full : growth->fullDomains)
        full->walk([&](scf::ForOp loop) {
          auto trips = weft::analysis::staticTripCount(loop);
          if (!trips || *trips <= 1) return;
          bool vectorWork = false;
          loop.walk([&](Operation *operation) {
            for (Type type : operation->getResultTypes()) {
              auto value = dyn_cast<weft::riscv::ValueType>(type);
              vectorWork |= (value && value.getLayout().getCarrier() == "rvv") ||
                            isa<weft::riscv::FragmentType, weft::riscv::PartialSetType>(type);
            }
          });
          if (vectorWork)
            loop->setAttr("weft.riscv.bound_native_unroll", b.getUnitAttr());
        });
      kernel->setAttr("weft.riscv.full_tile_operation_budget", b.getI64IntegerAttr(growth->budget));
      kernel->setAttr("weft.riscv.full_tile_added_operations", b.getI64IntegerAttr(growth->operations));
      kernel->setAttr("weft.riscv.full_domain_versions", b.getI64IntegerAttr(growth->domainVersions));
      kernel->setAttr("weft.riscv.full_tile_snapshot_scopes", b.getI64IntegerAttr(snapshotConsumers.size()));
      kernel->setAttr("weft.riscv.full_tile_pipeline_scopes", b.getI64IntegerAttr(pipelineConsumers.size()));
      kernel->setAttr("weft.riscv.full_tile_flat_recurrence_scopes", b.getI64IntegerAttr(flatConsumers.size()));
    });
    if (invalid) signalPassFailure();
  }
};
}
std::unique_ptr<Pass> weft::createSeparateRISCVFullTilesPass() {
  return std::make_unique<SeparateFullTiles>();
}
