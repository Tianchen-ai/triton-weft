#include "Weft/Target/X86Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <numeric>

using namespace mlir;
namespace {

struct Demand {
  Operation *user;
  SmallVector<int64_t> indices;
  int64_t begin, end;
};

// A replicated scalar has a different physical demand from a permutation of
// a whole vector. Keep one scalar snapshot per demanded element, then broadcast
// directly into native packets. This exposes memory broadcasts to LLVM without
// extending the original read past any intervening effect.
bool splitPacketBroadcasts(LLVM::MaskedLoadOp load, ArrayRef<Demand> demands,
                           IRRewriter &b) {
  auto function = load->getParentOfType<func::FuncOp>();
  auto bits = function ? function->getAttrOfType<IntegerAttr>("weft.x86.vector_bits") : IntegerAttr();
  auto type = cast<VectorType>(load.getRes().getType());
  unsigned elementBits = type.getElementTypeBitWidth();
  if (!bits || elementBits > unsigned(bits.getInt())) return false;
  int64_t lanes = bits.getInt() / elementBits;
  SmallVector<int64_t> selected;
  for (const Demand &demand : demands) {
    if (!isa<vector::ShuffleOp>(demand.user) || demand.indices.size() % lanes) return false;
    for (size_t start = 0; start < demand.indices.size(); start += lanes) {
      int64_t index = demand.indices[start];
      if (!llvm::all_of(ArrayRef(demand.indices).slice(start, lanes),
                        [&](int64_t other) { return other == index; })) return false;
      if (!llvm::is_contained(selected, index)) selected.push_back(index);
    }
  }
  if (selected.size() < 2 || selected.size() > 16) return false;
  b.setInsertionPoint(load);
  Location loc = load.getLoc();
  SmallVector<Value> snapshots;
  auto scalarVector = VectorType::get({1}, type.getElementType());
  for (int64_t index : selected) {
    SmallVector<int64_t> lane{index};
    Value offset = arith::ConstantIntOp::create(b, loc, index, 64);
    Value address = LLVM::GEPOp::create(b, loc, load.getData().getType(),
        type.getElementType(), load.getData(), ValueRange{offset});
    Value mask = vector::ShuffleOp::create(b, loc, load.getMask(), load.getMask(), lane);
    Value fill = load.getPassThru() ? Value(vector::ShuffleOp::create(b, loc,
        load.getPassThru(), load.getPassThru(), lane)) : Value();
    Value packet = LLVM::MaskedLoadOp::create(b, loc, scalarVector, address,
        mask, fill, b.getI32IntegerAttr(1), load.getNontemporalAttr());
    snapshots.push_back(vector::ExtractOp::create(b, loc, packet, ArrayRef<int64_t>{0}));
  }
  for (const Demand &demand : demands) {
    b.setInsertionPoint(demand.user);
    Location useLoc = demand.user->getLoc();
    auto resultType = cast<VectorType>(demand.user->getResult(0).getType());
    Value result = LLVM::UndefOp::create(b, useLoc, resultType);
    for (size_t start = 0; start < demand.indices.size(); start += lanes) {
      auto found = llvm::find(selected, demand.indices[start]);
      Value packet = vector::BroadcastOp::create(b, useLoc,
          VectorType::get({lanes}, type.getElementType()), snapshots[found - selected.begin()]);
      result = vector::InsertStridedSliceOp::create(b, useLoc, packet, result,
          ArrayRef<int64_t>{int64_t(start)}, ArrayRef<int64_t>{1});
    }
    b.replaceOp(demand.user, result);
  }
  b.eraseOp(load);
  return true;
}

// Choose memory packets at the original snapshot, not at its later consumers.
// Disjoint field ranges do not need to share one arbitrarily aligned wide
// register value. All direct consumers must be known, so splitting never adds
// a second copy of the original full-record read.
class SliceSnapshotLoads
    : public PassWrapper<SliceSnapshotLoads, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SliceSnapshotLoads)
  StringRef getArgument() const override { return "weft-x86-slice-snapshot-loads"; }
  StringRef getDescription() const override {
    return "Split masked snapshots by proven consumer ranges at the original read point";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    SmallVector<LLVM::MaskedLoadOp> loads;
    getOperation().walk([&](LLVM::MaskedLoadOp load) { loads.push_back(load); });
    IRRewriter b(&getContext());
    for (auto load : loads) {
      auto type = dyn_cast<VectorType>(load.getRes().getType());
      if (!type || type.isScalable() || type.getRank() != 1 || load.getRes().use_empty()) continue;
      if (!isa<IntegerType, FloatType>(type.getElementType()) ||
          type.getElementTypeBitWidth() % 8 != 0) continue;
      SmallVector<Demand> demands;
      llvm::SmallPtrSet<Operation *, 16> seen;
      bool supported = true;
      for (Operation *user : load.getRes().getUsers()) {
        if (!seen.insert(user).second) continue;
        SmallVector<int64_t> indices;
        if (auto shuffle = dyn_cast<vector::ShuffleOp>(user)) {
          int64_t leftSize = cast<VectorType>(shuffle.getV1().getType()).getNumElements();
          for (int64_t index : shuffle.getMask()) {
            Value source = index < leftSize ? shuffle.getV1() : shuffle.getV2();
            int64_t selected = index < leftSize ? index : index - leftSize;
            if (source != load.getRes() || selected < 0 || selected >= type.getNumElements()) {
              supported = false;
              break;
            }
            indices.push_back(selected);
          }
        } else if (auto extract = dyn_cast<vector::ExtractOp>(user)) {
          if (extract.hasDynamicPosition() || extract.getStaticPosition().size() != 1 ||
              isa<VectorType>(extract.getType())) { supported = false; break; }
          int64_t index = extract.getStaticPosition().front();
          if (index < 0 || index >= type.getNumElements()) { supported = false; break; }
          indices.push_back(index);
        } else { supported = false; break; }
        if (!supported || indices.empty()) { supported = false; break; }
        auto bounds = std::minmax_element(indices.begin(), indices.end());
        int64_t begin = *bounds.first, end = *bounds.second + 1;
        demands.push_back({user, std::move(indices), begin, end});
      }
      if (!supported) continue;
      if (splitPacketBroadcasts(load, demands, b)) continue;
      llvm::sort(demands, [](const Demand &a, const Demand &c) { return a.begin < c.begin; });
      struct Range { int64_t begin, end; size_t first, last; bool scalars; };
      SmallVector<Range> ranges;
      for (auto [i, demand] : llvm::enumerate(demands)) {
        bool scalar = isa<vector::ExtractOp>(demand.user);
        bool adjacentPacket = !ranges.empty() && scalar && ranges.back().scalars &&
            demand.begin == ranges.back().end &&
            (demand.end - ranges.back().begin) * type.getElementTypeBitWidth() <= 128;
        if (!ranges.empty() && (demand.begin < ranges.back().end || adjacentPacket)) {
          ranges.back().end = std::max(ranges.back().end, demand.end);
          ranges.back().last = i + 1;
          ranges.back().scalars &= scalar;
        } else ranges.push_back({demand.begin, demand.end, i, i + 1, scalar});
      }
      if (ranges.size() > 16 || (ranges.size() == 1 && ranges.front().begin == 0 &&
                                ranges.front().end == type.getNumElements())) continue;
      for (auto range : ranges) {
        b.setInsertionPoint(load);
        Location loc = load.getLoc();
        SmallVector<int64_t> indices(range.end - range.begin);
        std::iota(indices.begin(), indices.end(), range.begin);
        auto project = [&](Value value) -> Value {
          return vector::ShuffleOp::create(b, loc, value, value, indices);
        };
        Value offset = arith::ConstantIntOp::create(b, loc, range.begin, 64);
        Value address = LLVM::GEPOp::create(b, loc, load.getData().getType(),
            type.getElementType(), load.getData(), ValueRange{offset});
        Value mask = project(load.getMask());
        Value fill = load.getPassThru() ? project(load.getPassThru()) : Value();
        Value packet = LLVM::MaskedLoadOp::create(b, loc,
            VectorType::get({range.end - range.begin}, type.getElementType()),
            address, mask, fill, b.getI32IntegerAttr(1), load.getNontemporalAttr());
        for (size_t i = range.first; i < range.last; ++i) {
          auto &demand = demands[i];
          for (int64_t &index : demand.indices) index -= range.begin;
          b.setInsertionPoint(demand.user);
          Value result;
          if (isa<vector::ExtractOp>(demand.user))
            result = vector::ExtractOp::create(b, demand.user->getLoc(), packet, demand.indices);
          else result = vector::ShuffleOp::create(b, demand.user->getLoc(), packet, packet, demand.indices);
          b.replaceOp(demand.user, result);
        }
      }
      b.eraseOp(load);
    }
  }
};
} // namespace

std::unique_ptr<Pass> weft::createX86SnapshotLoadPass() {
  return std::make_unique<SliceSnapshotLoads>();
}
