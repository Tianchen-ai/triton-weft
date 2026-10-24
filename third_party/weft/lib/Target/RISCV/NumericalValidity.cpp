#include "NumericalValidity.h"
#include "RISCVPhysicalSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
riscv::ValueType domainType(OpBuilder &b, Type element, ArrayRef<int64_t> shape,
                            ArrayRef<int64_t> axes) {
  return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape),
      b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
}
Type predicateType(OpBuilder &b, Type type) {
  if (auto shaped = dyn_cast<riscv::ValueType>(type))
    return domainType(b, b.getI1Type(), shaped.getShape().asArrayRef(), shaped.getAxisIds().asArrayRef());
  return b.getI1Type();
}

class ActiveExtents {
  IRRewriter &b;
  unsigned remaining = 256;
  DenseMap<Value, SmallVector<Value>> known;
public:
  explicit ActiveExtents(IRRewriter &b) : b(b) {}
  FailureOr<SmallVector<Value>> get(Value value) {
    if (auto found = known.find(value); found != known.end()) return found->second;
    auto type = dyn_cast<riscv::ValueType>(value.getType());
    Operation *op = value.getDefiningOp();
    if (!remaining || !type || !op) return failure();
    --remaining;
    SmallVector<Value> result;
    if (auto load = dyn_cast<riscv::LoadOp>(op)) {
      for (unsigned axis = 0; axis < type.getShape().size(); ++axis)
        result.push_back(b.createOrFold<riscv::ExtentOp>(op->getLoc(), b.getIndexType(), load.getRegion(), axis));
    } else if (auto field = dyn_cast<riscv::FieldOp>(op)) {
      auto ownerAxes = logicalAxes(field.getOwner().getType());
      SmallVector<Value> ownerBounds;
      if (!ownerAxes.empty()) {
        auto bounds = get(field.getOwner());
        if (failed(bounds)) return failure();
        ownerBounds = *bounds;
      }
      for (auto [extent, axis] : llvm::zip(type.getShape().asArrayRef(), type.getAxisIds().asArrayRef())) {
        auto owner = llvm::find(ownerAxes, axis);
        if (owner != ownerAxes.end()) result.push_back(ownerBounds[owner - ownerAxes.begin()]);
        else if (extent > 0) result.push_back(arith::ConstantIndexOp::create(b, op->getLoc(), extent));
        else return failure();
      }
    } else if (isa<riscv::FieldReadOp, riscv::ConvertLayoutOp,
                   riscv::RegisterMaterializeOp, riscv::MaterializeOp>(op)) {
      auto extents = get(op->getOperand(0));
      if (failed(extents)) return failure();
      result = *extents;
    } else if (isa<riscv::IotaOp>(op)) {
      for (int64_t extent : type.getShape().asArrayRef()) {
        if (extent <= 0) return failure();
        result.push_back(arith::ConstantIndexOp::create(b, op->getLoc(), extent));
      }
    } else if (auto lookup = dyn_cast<riscv::LookupOp>(op)) {
      // A rectangular index domain does not prove validity of arbitrary table
      // entries. Only an independently complete computed table qualifies here.
      auto table = dyn_cast<riscv::ValueType>(lookup.getTable().getType());
      auto tableExtents = get(lookup.getTable());
      if (!table || failed(tableExtents)) return failure();
      for (auto [bound, expected] : llvm::zip(*tableExtents, table.getShape().asArrayRef())) {
        APInt actual;
        if (!matchPattern(bound, m_ConstantInt(&actual)) || actual.getSExtValue() != expected) return failure();
      }
      auto extents = get(lookup.getIndices());
      if (failed(extents)) return failure();
      result = *extents;
    } else if (isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp, riscv::BitcastOp,
                   riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp>(op)) {
      SmallVector<std::pair<riscv::ValueType, SmallVector<Value>>> sources;
      for (Value operand : op->getOperands()) {
        auto shaped = dyn_cast<riscv::ValueType>(operand.getType());
        if (!shaped) continue;
        auto extents = get(operand);
        if (failed(extents)) return failure();
        sources.push_back({shaped, *extents});
      }
      for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
        Value bound;
        for (auto &[input, extents] : sources) {
          auto ids = input.getAxisIds().asArrayRef();
          auto found = llvm::find(ids, axis);
          if (found == ids.end()) continue;
          size_t index = found - ids.begin();
          if (input.getShape()[index] == 1 && type.getShape()[position] != 1) continue;
          bound = bound ? b.createOrFold<arith::MinUIOp>(op->getLoc(), bound, extents[index]) : extents[index];
        }
        if (!bound) return failure();
        result.push_back(bound);
      }
    } else return failure();
    known[value] = result;
    return result;
  }
};

class ValidityPropagation {
  IRRewriter &b;
  DenseMap<Value, Value> &masks;
  int64_t &nextAxis;
  DenseMap<std::pair<int64_t, int64_t>, int64_t> predicateAxes;
  DenseMap<Operation *, SmallVector<unsigned>> controlSlots;
  SmallVector<int64_t> completeAxes(riscv::ValueType type) {
    SmallVector<int64_t> axes;
    for (auto [axis, extent] : llvm::zip(type.getAxisIds().asArrayRef(), type.getShape().asArrayRef())) {
      auto [found, inserted] = predicateAxes.try_emplace(std::make_pair(axis, extent), nextAxis);
      if (inserted) ++nextAxis;
      axes.push_back(found->second);
    }
    return axes;
  }
  Value rename(Value value, riscv::ValueType output, ArrayRef<int64_t> axes, Location loc) {
    auto input = dyn_cast<riscv::ValueType>(value.getType());
    if (!input) return value;
    auto ownAxes = completeAxes(input);
    for (auto [position, axis] : llvm::enumerate(input.getAxisIds().asArrayRef())) {
      auto found = llvm::find(output.getAxisIds().asArrayRef(), axis);
      if (found != output.getAxisIds().asArrayRef().end())
        ownAxes[position] = axes[found - output.getAxisIds().asArrayRef().begin()];
    }
    return riscv::ReshapeOp::create(b, loc,
        domainType(b, input.getElementType(), input.getShape().asArrayRef(), ownAxes), value,
        input.getAxisIds(), DenseI64ArrayAttr());
  }
  Value restore(Value value, riscv::ValueType output, Location loc) {
    return riscv::ReshapeOp::create(b, loc, output, value,
        cast<riscv::ValueType>(value.getType()).getAxisIds(), DenseI64ArrayAttr());
  }
  Value choose(Value condition, Value yes, Value no, Type type, Location loc) {
    auto output = dyn_cast<riscv::ValueType>(type);
    if (!output) return riscv::SelectOp::create(b, loc, type, condition, yes, no, unselectedLeaf(b));
    auto axes = completeAxes(output);
    auto privateType = domainType(b, output.getElementType(), output.getShape().asArrayRef(), axes);
    Value selected = riscv::SelectOp::create(b, loc, privateType,
        rename(condition, output, axes, loc), rename(yes, output, axes, loc), rename(no, output, axes, loc),
        unselectedLeaf(b));
    return restore(selected, output, loc);
  }
  llvm::DenseSet<Value> affected;
  void collectAffected() {
    SmallVector<Value> pending;
    auto add = [&](Value value) {
      if (affected.insert(value).second) pending.push_back(value);
    };
    for (auto &entry : masks) add(entry.first);
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *owner = use.getOwner();
        unsigned position = use.getOperandNumber();
        if (auto loop = dyn_cast<scf::ForOp>(owner)) {
          if (position >= 3) {
            unsigned slot = position - 3;
            add(loop.getRegionIterArg(slot));
            add(loop.getResult(slot));
          }
        } else if (auto yield = dyn_cast<scf::YieldOp>(owner)) {
          Operation *parent = yield->getParentOp();
          if (position < parent->getNumResults()) add(parent->getResult(position));
          if (auto loop = dyn_cast<scf::ForOp>(parent)) add(loop.getRegionIterArg(position));
        } else if (auto ret = dyn_cast<riscv::HelperReturnOp>(owner)) {
          add(ret->getParentOfType<riscv::HelperOp>().getResult(position));
        } else if (!isa<scf::IfOp, riscv::HelperOp>(owner)) {
          // Conditions are structural control, and helper captures have their
          // actual uses inside the closure. Neither makes every return invalid.
          for (Value result : owner->getResults()) add(result);
        }
      }
    }
  }
  SmallVector<unsigned> carriedMasks(Operation *op) {
    SmallVector<unsigned> selected;
    for (auto [position, result] : llvm::enumerate(op->getResults()))
      if (affected.contains(result)) selected.push_back(position);
    return selected;
  }
  Value full(Type type, Location loc) {
    auto shaped = dyn_cast<riscv::ValueType>(type);
    Value result = riscv::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(true));
    if (!shaped) return result;
    auto complete = completeAxes(shaped);
    SmallVector<int64_t> shape, axes;
    for (auto [extent, axis] : llvm::zip(shaped.getShape().asArrayRef(), complete)) {
      auto u32 = IntegerType::get(b.getContext(), 32, IntegerType::Unsigned);
      auto indexType = domainType(b, u32, {extent}, {axis});
      Value index = riscv::IotaOp::create(b, loc, indexType, 0, extent, unselectedLeaf(b));
      Value zero = riscv::ConstantOp::create(b, loc, u32, b.getIntegerAttr(u32, 0));
      Value active = riscv::CompareOp::create(b, loc, predicateType(b, indexType), index, zero, "ge", unselectedLeaf(b));
      shape.push_back(extent); axes.push_back(axis);
      result = riscv::BinaryOp::create(b, loc, domainType(b, b.getI1Type(), shape, axes), result, active, "and", unselectedLeaf(b));
    }
    return restore(result, cast<riscv::ValueType>(predicateType(b, type)), loc);
  }
  Value conjunction(Value left, Value right, Type type, Location loc) {
    auto output = dyn_cast<riscv::ValueType>(predicateType(b, type));
    if (!output) return riscv::BinaryOp::create(b, loc, b.getI1Type(), left, right, "and", unselectedLeaf(b));
    // Validity is defined for the complete capacity, including false lanes.
    // Private coordinate aliases prevent the enclosing numerical tail from
    // truncating a predicate update and making those false lanes unspecified.
    auto axes = completeAxes(output);
    Value result = riscv::BinaryOp::create(b, loc,
        domainType(b, b.getI1Type(), output.getShape().asArrayRef(), axes),
        rename(left, output, axes, loc), rename(right, output, axes, loc), "and", unselectedLeaf(b));
    return restore(result, output, loc);
  }
  Value knownOrFull(Value value, Location loc) {
    if (Value mask = masks.lookup(value)) return mask;
    return full(value.getType(), loc);
  }
  Value combined(Operation *op, Type type) {
    SmallVector<Value> inputs;
    for (Value input : op->getOperands())
      if (Value mask = masks.lookup(input); mask && !llvm::is_contained(inputs, mask)) inputs.push_back(mask);
    if (inputs.size() == 1) {
      auto source = dyn_cast<riscv::ValueType>(inputs[0].getType());
      auto output = dyn_cast<riscv::ValueType>(type);
      if ((!source && !output) || (source && output && source.getShape() == output.getShape() &&
                                  source.getAxisIds() == output.getAxisIds())) return inputs[0];
    }
    Value result = full(type, op->getLoc());
    for (Value mask : inputs) result = conjunction(result, mask, type, op->getLoc());
    return result;
  }
public:
  ValidityPropagation(IRRewriter &b, DenseMap<Value, Value> &masks, int64_t &nextAxis)
      : b(b), masks(masks), nextAxis(nextAxis) {}
  LogicalResult run(ArrayRef<Operation *> operations) {
    if (masks.empty()) return success();
    collectAffected();
    for (Operation *op : operations) {
      // Predicates are physical SSA companions. Structured control carries them
      // through the same executed arms/iterations as the numerical result; no
      // predicate is recomputed by reading an earlier source snapshot again.
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto slots = carriedMasks(op);
        if (slots.empty()) continue;
        b.setInsertionPoint(op);
        SmallVector<Value> initial(loop.getInitArgs());
        for (unsigned slot : slots) initial.push_back(knownOrFull(loop.getInitArgs()[slot], op->getLoc()));
        auto next = scf::ForOp::create(b, op->getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), initial);
        next->setAttrs(loop->getAttrs());
        next.getRegion().takeBody(loop.getRegion());
        unsigned originalCount = loop.getNumResults();
        for (auto [position, slot] : llvm::enumerate(slots)) {
          auto mask = next.getBody()->addArgument(predicateType(b, loop.getResult(slot).getType()), op->getLoc());
          masks[next.getRegionIterArg(slot)] = mask;
          masks[next.getResult(slot)] = next.getResult(originalCount + position);
        }
        controlSlots[next] = slots;
        b.replaceOp(loop, next.getResults().take_front(originalCount));
        continue;
      }
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        auto slots = carriedMasks(op);
        if (slots.empty()) continue;
        b.setInsertionPoint(op);
        SmallVector<Type> types(branch.getResultTypes());
        for (unsigned slot : slots) types.push_back(predicateType(b, branch.getResult(slot).getType()));
        auto next = scf::IfOp::create(b, op->getLoc(), types, branch.getCondition(), true);
        next->setAttrs(branch->getAttrs());
        next.getThenRegion().takeBody(branch.getThenRegion());
        next.getElseRegion().takeBody(branch.getElseRegion());
        unsigned originalCount = branch.getNumResults();
        for (auto [position, slot] : llvm::enumerate(slots))
          masks[next.getResult(slot)] = next.getResult(originalCount + position);
        controlSlots[next] = slots;
        b.replaceOp(branch, next.getResults().take_front(originalCount));
        continue;
      }
      if (auto helper = dyn_cast<riscv::HelperOp>(op)) {
        auto slots = carriedMasks(op);
        if (slots.empty()) continue;
        b.setInsertionPoint(op);
        SmallVector<Type> types(helper.getResultTypes());
        SmallVector<Value> captures(helper.getCaptures());
        for (Value value : helper.getCaptures())
          if (Value mask = masks.lookup(value)) captures.push_back(mask);
        for (unsigned slot : slots) types.push_back(predicateType(b, helper.getResult(slot).getType()));
        auto next = riscv::HelperOp::create(b, op->getLoc(), types, captures, helper.getNameAttr(), helper.getNoinlineAttr());
        next->setAttrs(helper->getAttrs());
        next.getBody().takeBody(helper.getBody());
        unsigned originalCount = helper.getNumResults();
        for (auto [position, slot] : llvm::enumerate(slots))
          masks[next.getResult(slot)] = next.getResult(originalCount + position);
        controlSlots[next] = slots;
        b.replaceOp(helper, next.getResults().take_front(originalCount));
        continue;
      }
      if (isa<scf::YieldOp, riscv::HelperReturnOp>(op)) {
        Operation *owner = isa<riscv::HelperReturnOp>(op) ?
            op->getParentOfType<riscv::HelperOp>().getOperation() : op->getParentOp();
        auto found = controlSlots.find(owner);
        if (found != controlSlots.end()) {
          b.setInsertionPoint(op);
          SmallVector<Value> values(op->getOperands());
          for (unsigned slot : found->second) values.push_back(knownOrFull(op->getOperand(slot), op->getLoc()));
          op->setOperands(values);
          continue;
        }
      }
      if (!llvm::any_of(op->getOperands(), [&](Value value) { return masks.contains(value); })) continue;
      if (op->getNumResults() == 1 && masks.contains(op->getResult(0))) continue;
      b.setInsertionPoint(op);
      auto loc = op->getLoc();
      if (auto store = dyn_cast<riscv::PointerStoreOp>(op)) {
        Value mask = combined(op, store.getValue().getType());
        store.getMaskMutable().assign(conjunction(store.getMask(), mask, store.getValue().getType(), loc));
      } else if (auto atomic = dyn_cast<riscv::PointerAtomicOp>(op)) {
        Value mask = combined(op, atomic.getResult().getType());
        mask = conjunction(atomic.getMask(), mask, atomic.getResult().getType(), loc);
        atomic.getMaskMutable().assign(mask);
        masks[atomic.getResult()] = mask;
      } else if (auto store = dyn_cast<riscv::StoreOp>(op)) {
        if (isa<riscv::ValueType>(store.getValue().getType()))
          return store.emitError("grouped validity requires an explicit masked shaped store program");
        Value mask = combined(op, store.getValue().getType());
        auto branch = scf::IfOp::create(b, loc, mask, false);
        store->moveBefore(branch.thenBlock()->getTerminator());
      } else if (auto extract = dyn_cast<riscv::ExtractOp>(op)) {
        Value source = masks.lookup(extract.getInput());
        if (!source) return extract.emitError("numeric validity through an indexed operand requires a complete projection");
        masks[extract.getResult()] = riscv::ExtractOp::create(b, loc,
            predicateType(b, extract.getResult().getType()), source, extract.getIndices(),
            extract.getSelectorsAttr(), unassignedAccess(b), unselectedLeaf(b));
      } else if (auto reshape = dyn_cast<riscv::ReshapeOp>(op)) {
        masks[reshape.getResult()] = riscv::ReshapeOp::create(b, loc,
            cast<riscv::ValueType>(predicateType(b, reshape.getResult().getType())), masks.lookup(reshape.getInput()),
            reshape.getOrderAttr(), DenseI64ArrayAttr());
      } else if (auto select = dyn_cast<riscv::SelectOp>(op)) {
        Value condition = select.getCondition();
        if (Value active = masks.lookup(condition))
          condition = choose(active, condition,
              riscv::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(false)), condition.getType(), loc);
        Value chosen = choose(condition, knownOrFull(select.getTrueValue(), loc),
            knownOrFull(select.getFalseValue(), loc), predicateType(b, select.getResult().getType()), loc);
        if (Value condition = masks.lookup(select.getCondition()))
          chosen = conjunction(chosen, condition, select.getResult().getType(), loc);
        masks[select.getResult()] = chosen;
      } else if (isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp, riscv::BitcastOp,
                     riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp>(op)) {
        masks[op->getResult(0)] = combined(op, op->getResult(0).getType());
      } else {
        return op->emitError("numeric validity has no explicit physical propagation for this consumer");
      }
    }
    return success();
  }
};
}

FailureOr<SmallVector<Value>> weft::riscv_internal::activeNumericExtents(IRRewriter &b, Value value) {
  return ActiveExtents(b).get(value);
}

LogicalResult weft::riscv_internal::propagateNumericValidity(IRRewriter &b,
    ArrayRef<Operation *> operations, DenseMap<Value, Value> &masks, int64_t &nextAxis) {
  return ValidityPropagation(b, masks, nextAxis).run(operations);
}
