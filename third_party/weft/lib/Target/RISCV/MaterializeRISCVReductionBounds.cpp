#include "Weft/Target/RISCVPasses.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "NumericalValidity.h"
#include "ReductionBounds.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;

namespace {

bool preservesMembers(Operation *op) {
  return isa<riscv::RegisterMaterializeOp, riscv::ConvertLayoutOp,
             riscv::CastOp, riscv::WidenOp, riscv::NarrowOp, riscv::BitcastOp,
             riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
             riscv::FMAOp, riscv::RVVIntegerMACOp>(op);
}

// Validity follows the saved physical carrier, not the current axis spelling.
// The query is bounded and local to one rewrite: no stale analysis survives an
// SSA change. Its index values subsequently participate in ordinary liveness.
class SavedLengths {
  IRRewriter &b;
  Location loc;
  DenseMap<Value, SmallVector<Value>> known;
  unsigned remaining = 256;
  Value index(int64_t n) { return arith::ConstantIndexOp::create(b, loc, n); }
  Value clip(Value active, int64_t offset, int64_t width) {
    Value base = index(offset);
    Value used = b.createOrFold<arith::MinUIOp>(loc, active, base);
    Value rest = b.createOrFold<arith::SubIOp>(loc, active, used);
    return b.createOrFold<arith::MinUIOp>(loc, rest, index(width));
  }
  FailureOr<SmallVector<Value>> rectangle(Value value) {
    auto type = cast<riscv::ValueType>(value.getType());
    auto coordinates = riscv::RegisterCoordinates::get(type.getLayout());
    auto factors = type.getLayout().getLaneFactors().asArrayRef();
    if (!coordinates || type.getLayout().getReplicaGroup() != 1 ||
        llvm::count_if(factors, [](int64_t n) { return n > 1; }) > 1)
      return failure(); // A rectangle with holes needs masks, never a prefix.
    auto extents = riscv_internal::activeNumericExtents(b, value);
    if (failed(extents) || extents->size() != type.getShape().size()) return failure();
    SmallVector<Value> lengths;
    for (int64_t part = 0; part < coordinates->parts(); ++part) {
      auto origin = coordinates->logical(part);
      if (!origin) return failure();
      Value length = index(coordinates->lanes());
      for (unsigned dimension = 0; dimension < origin->size(); ++dimension) {
        Value bound = (*extents)[dimension];
        if (factors[dimension] > 1) {
          Value active = clip(bound, (*origin)[dimension], coordinates->lanes());
          length = b.createOrFold<arith::MinUIOp>(loc, length, active);
        } else {
          Value valid = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, index((*origin)[dimension]), bound);
          length = b.createOrFold<arith::SelectOp>(loc, valid, length, index(0));
        }
      }
      lengths.push_back(length);
    }
    return lengths;
  }
  FailureOr<SmallVector<Value>> project(Value input, riscv::ValueType result) {
    auto lengths = get(input);
    if (failed(lengths)) return failure();
    auto type = cast<riscv::ValueType>(input.getType());
    auto from = riscv::RegisterCoordinates::get(type.getLayout());
    auto to = riscv::RegisterCoordinates::get(result.getLayout());
    if (!from || !to || to->lanes() > 4096 || to->parts() > 4096 / to->lanes()) return failure();
    SmallVector<Value> projected;
    for (int64_t part = 0; part < to->parts(); ++part) {
      SmallVector<riscv::RegisterLocation> members;
      for (int64_t lane = 0; lane < to->lanes(); ++lane) {
        auto point = to->logical(part, lane);
        if (!point) return failure();
        SmallVector<int64_t> selected;
        for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
          auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
          if (type.getShape()[position] == 1) selected.push_back(0);
          else if (found != result.getAxisIds().asArrayRef().end()) selected.push_back((*point)[found - result.getAxisIds().asArrayRef().begin()]);
          else return failure();
        }
        auto source = from->locate(selected);
        if (!source) return failure();
        members.push_back(*source);
      }
      const auto first = members.front();
      bool prefix = true, broadcast = true;
      for (auto [lane, member] : llvm::enumerate(members)) {
        prefix &= member.part == first.part && member.lane == first.lane + int64_t(lane);
        broadcast &= member.part == first.part && member.lane == first.lane;
      }
      if (prefix) projected.push_back(clip((*lengths)[first.part], first.lane, to->lanes()));
      else if (broadcast) {
        Value active = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, (*lengths)[first.part], index(first.lane));
        projected.push_back(b.createOrFold<arith::SelectOp>(loc, active, index(to->lanes()), index(0)));
      } else return failure();
    }
    return projected;
  }
public:
  SavedLengths(IRRewriter &b, Location loc) : b(b), loc(loc) {}
  FailureOr<SmallVector<Value>> get(Value value) {
    if (auto found = known.find(value); found != known.end()) return found->second;
    if (!remaining) return failure();
    --remaining;
    auto type = dyn_cast<riscv::ValueType>(value.getType());
    auto coordinates = type ? riscv::RegisterCoordinates::get(type.getLayout()) : std::nullopt;
    if (!coordinates) return failure();
    SmallVector<Value> result;
    if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>()) {
      auto source = get(reshape.getInput());
      if (failed(source)) return failure();
      if (auto order = reshape.getPartOrderAttr()) {
        for (int64_t part : order.asArrayRef()) result.push_back((*source)[part]);
      } else {
        if (!riscv::hasIdenticalLinearCarrier(reshape.getInput().getType(), type) &&
            !riscv::hasIdenticalReshapeCarrier(reshape.getInput().getType(), type, reshape.getOrder())) return failure();
        result = *source;
      }
    } else if (auto *op = value.getDefiningOp(); op && preservesMembers(op)) {
      result.assign(coordinates->parts(), index(coordinates->lanes()));
      for (Value input : op->getOperands()) {
        if (!isa<riscv::ValueType>(input.getType())) continue;
        auto projected = project(input, type);
        if (failed(projected)) return failure();
        for (auto [part, active] : llvm::enumerate(*projected))
          result[part] = b.createOrFold<arith::MinUIOp>(loc, result[part], active);
      }
    } else if (type.getLayout().getValidity() == "full") {
      result.assign(coordinates->parts(), index(coordinates->lanes()));
    } else {
      auto active = rectangle(value);
      if (failed(active)) return failure();
      result = *active;
    }
    known[value] = result;
    return result;
  }
};

LogicalResult captureBounds(IRRewriter &b, riscv::ReduceOp reduction) {
  if (!reduction.getActiveLengths().empty() ||
      !riscv_internal::needsSavedReductionBounds(reduction)) return success();
  b.setInsertionPoint(reduction);
  auto lengths = SavedLengths(b, reduction.getLoc()).get(reduction.getInput());
  if (failed(lengths)) return reduction.emitError("reshaped reduction requires a saved prefix domain; cartesian holes require an explicit validity mask");
  reduction.getActiveLengthsMutable().assign(*lengths);
  auto leaf = reduction.getLeaf();
  reduction.setLeafAttr(riscv::LeafAttr::get(b.getContext(), leaf.getEngine(), leaf.getFamily(),
      leaf.getInstruction(), leaf.getSpelling(), leaf.getOperandGroups(), leaf.getResultGroups(),
      1, leaf.getFragmentGroups(), leaf.getMask(), leaf.getTail(), leaf.getParameters(), leaf.getLocalBytes()));
  return success();
}

class MaterializeRISCVReductionBoundsPass
    : public PassWrapper<MaterializeRISCVReductionBoundsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVReductionBoundsPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-reduction-bounds"; }
  StringRef getDescription() const override { return "Capture saved active domains through physical register reshapes"; }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    WalkResult result = getOperation().walk([&](riscv::ReduceOp reduce) {
      return failed(captureBounds(b, reduce)) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};
}

bool weft::riscv_internal::needsSavedReductionBounds(riscv::ReduceOp reduction) {
  if (!reduction.getLeaf().getInstruction().starts_with("rvv.lane-reduce.")) return false;
  SmallVector<Value> pending{reduction.getInput()};
  DenseSet<Value> seen;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second) continue;
    if (seen.size() > 256) return true; // Cannot silently use implicit axis bounds.
    Operation *op = value.getDefiningOp();
    if (!op) continue;
    if (auto reshape = dyn_cast<riscv::ReshapeOp>(op)) {
      if (reshape.getInput().getType().getLayout().getValidity() == "tail" ||
          reshape.getResult().getType().getLayout().getValidity() == "tail") return true;
      pending.push_back(reshape.getInput());
    }
    if (preservesMembers(op))
      for (Value input : op->getOperands())
        if (isa<riscv::ValueType>(input.getType())) pending.push_back(input);
  }
  return false;
}

std::unique_ptr<Pass> weft::createMaterializeRISCVReductionBoundsPass() {
  return std::make_unique<MaterializeRISCVReductionBoundsPass>();
}
