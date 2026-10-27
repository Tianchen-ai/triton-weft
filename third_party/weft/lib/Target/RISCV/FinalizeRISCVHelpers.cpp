#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace weft;

namespace {
class FinalizeRISCVHelpersPass
    : public PassWrapper<FinalizeRISCVHelpersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeRISCVHelpersPass)
  StringRef getArgument() const override { return "weft-riscv-finalize-helpers"; }
  StringRef getDescription() const override { return "Close native entry placement and helper captures after physical resource selection"; }
  void runOnOperation() override {
    // Freeze the native entry placement before emission. A bounded 32-byte
    // policy avoids inheriting arbitrary two-byte placement from the linker;
    // it is independent of vector length and of the numerical program.
    getOperation().walk([&](riscv::KernelOp kernel) {
      kernel->setAttr("native_entry_alignment",
                     IntegerAttr::get(IntegerType::get(&getContext(), 64), 32));
    });
    bool invalidAlignment = false;
    getOperation().walk([&](riscv::KernelOp kernel) {
      auto alignment = kernel->getAttrOfType<IntegerAttr>("native_loop_alignment");
      if (!alignment || !llvm::is_contained(ArrayRef<int64_t>{0, 2, 4, 8, 16, 32, 64, 128}, alignment.getInt())) {
        kernel.emitError("native loop placement requires a valid explicit alignment policy");
        invalidAlignment = true;
      }
    });
    if (invalidAlignment) { signalPassFailure(); return; }
    getOperation().walk([&](scf::ForOp loop) {
      auto kernel = loop->getParentOfType<riscv::KernelOp>();
      if (!kernel) return;
      const int64_t alignment = kernel->getAttrOfType<IntegerAttr>("native_loop_alignment").getInt();
      if (loop->hasAttr("weft.riscv.bound_native_unroll")) {
        auto trips = analysis::staticTripCount(loop);
        int64_t count = 1;
        if (!loop->hasAttr("weft.riscv.system_unroll") && trips)
          for (int64_t candidate : {2, 4}) {
            auto groups = riscv_internal::nativeUnrollRegisterGroups(loop, candidate);
            if (uint64_t(candidate) <= *trips && groups && *groups <= kernel.getTarget().getVectorRegisters())
              count = candidate;
          }
        // Closing an isolated computation frame needs a bounded issue window.
        // Original kernel loops retain the native optimizer's established
        // decision; full-domain versioning alone does not request a new bound.
        if (loop->getParentOfType<riscv::HelperOp>())
          loop->setAttr("weft.riscv.native_unroll_count",
                        IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
        loop->removeAttr("weft.riscv.bound_native_unroll");
      }
      bool vectorBody = false;
      bool retainedChild = false;
      loop.walk([&](Operation *operation) {
        auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
        vectorBody |= leaf && (leaf.getEngine() == "rvv" || leaf.getEngine() == "ime");
        if (operation != loop && isa<scf::ForOp>(operation)) {
          auto trips = analysis::staticTripCount(operation);
          auto unroll = operation->getAttrOfType<StringAttr>("weft.riscv.system_unroll");
          auto bounded = operation->getAttrOfType<IntegerAttr>("weft.riscv.native_unroll_count");
          retainedChild |= !trips || *trips > (bounded ? uint64_t(bounded.getInt()) : 4) ||
                           (unroll && unroll.getValue() == "disable");
        }
      });
      // A finite child traversal can disappear in native code generation.
      // Align its enclosing vector-work loop, but do not pad every outer
      // traversal around a surviving child. Those headers add no vector issue
      // locality and can displace the child and its neighbouring hot blocks.
      if (alignment && vectorBody && !retainedChild)
        loop->setAttr("weft.riscv.code_alignment",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), alignment));
      else
        loop->removeAttr("weft.riscv.code_alignment");
    });
    getOperation().walk<WalkOrder::PostOrder>([&](riscv::HelperOp helper) {
      llvm::SetVector<Value> captures;
      getUsedValuesDefinedAbove(helper.getBody(), captures);
      // Descriptor bindings retain their explicit origin and index values.
      // Capture those dependencies as well; numerical producers stay outside.
      for (unsigned i = 0; i < captures.size(); ++i) {
        Value value = captures[i];
        if (auto definition = value.getDefiningOp(); definition &&
            !isa<riscv::MemoryViewOp, riscv::PointerCastOp, riscv::LocalViewOp>(definition) &&
            (isa<riscv::MemDescType, riscv::DomainType, riscv::PointType>(value.getType()) ||
             isa<riscv::FieldOp>(definition)))
          captures.insert(definition->operand_begin(), definition->operand_end());
      }
      helper.getCapturesMutable().assign(captures.getArrayRef());
      helper->setAttr("native_restrict_captures",
          DenseI64ArrayAttr::get(&getContext(), riscv::helperRestrictCaptures(helper)));
      helper->setAttr("native_abi", StringAttr::get(&getContext(), "selected-parts"));
    });
  }
};
}

std::unique_ptr<Pass> weft::createFinalizeRISCVHelpersPass() {
  return std::make_unique<FinalizeRISCVHelpersPass>();
}
