#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace weft;

namespace {
class Bounds {
  IRRewriter &b;
  Operation *anchor;
  Location loc;
  Value constant(int64_t n) { return arith::ConstantIndexOp::create(b, loc, n); }
  Value both(Value a, Value c) { return b.createOrFold<arith::AndIOp>(loc, a, c); }
  Value asIndex(Value value) {
    if (value.getType().isIndex()) return value;
    return riscv::CastOp::create(b, loc, b.getIndexType(), value, riscv_internal::unselectedLeaf(b));
  }
  Value offset(Value coordinate, Value delta, Value &valid) {
    delta = asIndex(delta);
    Value sum = b.createOrFold<arith::AddIOp>(loc, coordinate, delta);
    valid = both(valid, arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, sum, coordinate));
    valid = both(valid, arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, delta, constant(0)));
    return sum;
  }
  FailureOr<Value> origin(Value view, unsigned axis, unsigned depth) {
    if (depth > 64) return failure();
    if (auto root = view.getDefiningOp<riscv::MemoryViewOp>()) return root.getOrigins()[axis];
    if (view.getDefiningOp<riscv::LocalViewOp>()) return constant(0);
    if (auto subview = view.getDefiningOp<riscv::SubviewOp>()) return origin(subview.getBase(), axis, depth + 1);
    if (auto reaxis = view.getDefiningOp<riscv::ReaxisOp>()) return origin(reaxis.getBase(), axis, depth + 1);
    if (auto stage = view.getDefiningOp<riscv::StagedViewOp>()) return origin(stage.getSource(), axis, depth + 1);
    if (auto slice = view.getDefiningOp<riscv::SliceOp>()) {
      unsigned retained = 0, operand = 0;
      for (auto [sourceAxis, attr] : llvm::enumerate(slice.getSelectors())) {
        StringRef kind = cast<StringAttr>(attr).getValue();
        if (kind == "index" || kind == "group_index") { ++operand; continue; }
        if (retained++ == axis) {
          if (kind == "all") return origin(slice.getBase(), sourceAxis, depth + 1);
          if (kind == "domain") return Value(riscv::PointBaseOp::create(b, loc, b.getIndexType(), slice.getIndices()[operand]));
          return failure();
        }
        if (kind != "all") ++operand;
      }
      return failure();
    }
    auto type = cast<riscv::MemDescType>(view.getType());
    if (isa<BlockArgument>(view) && type.getOrigins()[axis] >= 0) return constant(type.getOrigins()[axis]);
    return failure();
  }
public:
  Bounds(IRRewriter &b, Operation *anchor) : b(b), anchor(anchor), loc(anchor->getLoc()) {}
  FailureOr<Value> get(Value view, ArrayRef<Value> point, unsigned depth = 0,
                       ArrayRef<bool> absolute = {}) {
    auto type = dyn_cast<riscv::MemDescType>(view.getType());
    if (!type || point.size() != type.getShape().size() || depth > 64)
      return anchor->emitError("memory bounds require a closed descriptor coordinate chain"), failure();
    Value valid = arith::ConstantIntOp::create(b, loc, 1, 1);
    for (auto [axis, coordinate] : llvm::enumerate(point)) {
      Value logical = coordinate;
      if (!absolute.empty() && absolute[axis]) {
        auto base = origin(view, axis, 0);
        if (failed(base)) return anchor->emitError("memory bounds lost a domain origin"), failure();
        valid = both(valid, arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, coordinate, *base));
        logical = b.createOrFold<arith::SubIOp>(loc, coordinate, *base);
      }
      Value extent = riscv::ExtentOp::create(b, loc, b.getIndexType(), view, axis);
      valid = both(valid, arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, logical, extent));
    }
    if (isa_and_nonnull<riscv::MemoryViewOp, riscv::LocalViewOp>(view.getDefiningOp())) return valid;
    Value parent;
    SmallVector<Value> coordinates(point);
    SmallVector<bool> absoluteAxes(point.size(), false);
    if (!absolute.empty()) absoluteAxes.assign(absolute.begin(), absolute.end());
    if (auto subview = view.getDefiningOp<riscv::SubviewOp>()) {
      parent = subview.getBase();
      unsigned dynamic = 0;
      for (auto [axis, value] : llvm::enumerate(subview.getOffsets())) {
        Value delta = value < 0 ? subview.getDynamicBounds()[dynamic++] : constant(value);
        coordinates[axis] = offset(coordinates[axis], delta, valid);
      }
    } else if (auto slice = view.getDefiningOp<riscv::SliceOp>()) {
      parent = slice.getBase();
      coordinates.clear();
      absoluteAxes.clear();
      unsigned retained = 0, operand = 0;
      for (Attribute attr : slice.getSelectors()) {
        StringRef kind = cast<StringAttr>(attr).getValue();
        if (kind == "all") {
          absoluteAxes.push_back(!absolute.empty() && absolute[retained]);
          coordinates.push_back(point[retained++]); continue;
        }
        Value selected = slice.getIndices()[operand++];
        if (kind == "index") { coordinates.push_back(asIndex(selected)); absoluteAxes.push_back(false); }
        else if (kind == "domain") {
          Value coordinate = point[retained];
          // A nested domain point replaces the same axis's previous absolute
          // selection. Plain indices and subview offsets remain relative.
          if (absolute.empty() || !absolute[retained]) {
            Value base = riscv::PointBaseOp::create(b, loc, b.getIndexType(), selected);
            coordinate = offset(coordinate, base, valid);
          }
          ++retained;
          coordinates.push_back(coordinate);
          absoluteAxes.push_back(true);
        } else return slice.emitError("memory bounds require a scalar or domain selector"), failure();
      }
    } else if (auto reaxis = view.getDefiningOp<riscv::ReaxisOp>()) parent = reaxis.getBase();
    else if (auto staged = view.getDefiningOp<riscv::StagedViewOp>()) parent = staged.getSource();
    else if (isa<BlockArgument>(view)) return valid;
    else return anchor->emitError("memory bounds have no backing storage descriptor"), failure();
    auto backing = get(parent, coordinates, depth + 1, absoluteAxes);
    if (failed(backing)) return failure();
    return both(valid, *backing);
  }
};
}

FailureOr<Value> weft::riscv_internal::memoryContains(IRRewriter &b,
    Value view, ArrayRef<Value> coordinates, Operation *anchor) {
  return Bounds(b, anchor).get(view, coordinates);
}
