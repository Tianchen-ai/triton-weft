#include "IndexRelations.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
namespace wk = weft::kernel;
namespace {
IntegerType integerType(Value value) {
  Type type = value.getType();
  if (auto shaped = dyn_cast<wk::ValueType>(type)) type = shaped.getElementType();
  if (isa<IndexType>(type)) return IntegerType::get(type.getContext(), 64);
  return dyn_cast<IntegerType>(type);
}

uint64_t maximum(unsigned width) {
  return width ? std::numeric_limits<uint64_t>::max() >> (64 - width) : 0;
}

struct Bounds {
  uint64_t low, high;
};

class IndexRelations {
  DenseMap<Value, Bounds> bounds;
  DenseMap<Value, std::optional<int64_t>> steps;
  int64_t axis;

  Bounds range(Value value, unsigned depth = 0) {
    auto type = integerType(value);
    if (!type || type.getWidth() > 64) return {0, UINT64_MAX};
    Bounds full{0, maximum(type.getWidth())};
    if (depth > 64) return full;
    if (auto found = bounds.find(value); found != bounds.end()) return found->second;
    auto remember = [&](Bounds result) { bounds.try_emplace(value, result); return result; };
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      APInt lower, upper, step;
      if (loop && argument == loop.getInductionVar() &&
          matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
          matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) &&
          matchPattern(loop.getStep(), m_ConstantInt(&step)) &&
          !lower.isNegative() && !upper.isNegative() && upper.ugt(lower) &&
          !step.isNegative() && !step.isZero())
        return remember({lower.getZExtValue(), upper.getZExtValue() - 1});
    }
    Attribute attribute;
    if (auto constant = value.getDefiningOp<wk::ConstantOp>()) attribute = constant.getValue();
    else matchPattern(value, m_Constant(&attribute));
    if (auto constant = dyn_cast_or_null<IntegerAttr>(attribute)) {
      uint64_t bits = constant.getValue().zextOrTrunc(type.getWidth()).getZExtValue();
      return remember({bits, bits});
    }
    if (auto constant = dyn_cast_or_null<DenseIntElementsAttr>(attribute)) {
      Bounds result{full.high, 0};
      for (APInt bits : constant.getValues<APInt>()) {
        uint64_t number = bits.getZExtValue();
        result.low = std::min(result.low, number);
        result.high = std::max(result.high, number);
      }
      return remember(result);
    }
    if (auto iota = value.getDefiningOp<wk::IotaOp>())
      if (iota.getStart() >= 0 && iota.getEnd() > iota.getStart() &&
          uint64_t(iota.getEnd() - 1) <= full.high)
        return remember({uint64_t(iota.getStart()), uint64_t(iota.getEnd() - 1)});
    Operation *op = value.getDefiningOp();
    if (op && isa<wk::CastOp, wk::WidenOp>(op)) {
      auto inputType = integerType(op->getOperand(0));
      if (!inputType) return remember(full);
      Bounds input = range(op->getOperand(0), depth + 1);
      // Sign extension and truncation only retain this unsigned interval when
      // all bits denote nonnegative, exactly representable source values.
      uint64_t inputLimit = inputType.isSigned() ? maximum(inputType.getWidth() - 1) : maximum(inputType.getWidth());
      if (input.high <= inputLimit && input.high <= full.high) return remember(input);
    }
    if (auto fresh = dyn_cast_or_null<wk::NewOp>(op))
      if (fresh.getInitial()) return remember(range(fresh.getInitial(), depth + 1));
    if (auto binary = dyn_cast_or_null<wk::BinaryOp>(op)) {
      Bounds left = range(binary.getLhs(), depth + 1), right = range(binary.getRhs(), depth + 1);
      StringRef kind = binary.getKind();
      if (kind == "and") return remember({0, std::min(left.high, right.high)});
      if (kind == "or" || kind == "xor") {
        // All possibly set bits of either interval bound the result. OR-ing
        // the interval maxima themselves is not a sound upper bound.
        auto bits = [](uint64_t n) { return n ? UINT64_MAX >> llvm::countl_zero(n) : uint64_t(0); };
        return remember({0, bits(left.high) | bits(right.high)});
      }
      if (kind == "add" && right.high <= full.high - left.high)
        return remember({left.low + right.low, left.high + right.high});
      if (kind == "sub" && left.low >= right.high)
        return remember({left.low - right.high, left.high - right.low});
      if (kind == "mul" && (!right.high || left.high <= full.high / right.high))
        return remember({left.low * right.low, left.high * right.high});
      if (kind == "shl" && right.high < type.getWidth() && left.high <= (full.high >> right.high))
        return remember({left.low << right.low, left.high << right.high});
      if (kind == "shr" && type.isUnsigned() && right.high < type.getWidth())
        return remember({left.low >> right.high, left.high >> right.low});
      if (kind == "div" && type.isUnsigned() && right.low)
        return remember({left.low / right.high, left.high / right.low});
      if (kind == "mod" && type.isUnsigned() && right.low)
        return remember({0, std::min(left.high, right.high - 1)});
    }
    return remember(full);
  }

  std::optional<int64_t> step(Value value, unsigned depth = 0) {
    if (depth > 64) return {};
    auto type = integerType(value);
    if (!type || type.getWidth() > 64) return {};
    auto shaped = dyn_cast<wk::ValueType>(value.getType());
    if (!shaped) return 0;
    auto ids = shaped.getAxisIds().asArrayRef();
    auto found = llvm::find(ids, axis);
    if (found == ids.end() || shaped.getShape()[found - ids.begin()] == 1) return 0;
    if (auto saved = steps.find(value); saved != steps.end()) return saved->second;
    auto remember = [&](std::optional<int64_t> result) { steps.try_emplace(value, result); return result; };
    uint64_t limit = maximum(type.isSigned() ? type.getWidth() - 1 : type.getWidth());
    if (auto iota = value.getDefiningOp<wk::IotaOp>())
      return remember(iota.getStart() >= 0 && iota.getEnd() > iota.getStart() &&
          uint64_t(iota.getEnd() - 1) <= limit ? std::optional<int64_t>(1) : std::nullopt);
    Bounds known = range(value);
    if (known.low == known.high) return remember(0);
    Operation *op = value.getDefiningOp();
    if (op && isa<wk::CastOp, wk::WidenOp>(op)) {
      auto inputType = integerType(op->getOperand(0));
      if (!inputType) return remember({});
      Bounds input = range(op->getOperand(0));
      uint64_t inputLimit = maximum(inputType.isSigned() ? inputType.getWidth() - 1 : inputType.getWidth());
      if (input.high <= std::min(inputLimit, limit))
        return remember(step(op->getOperand(0), depth + 1));
    }
    if (auto fresh = dyn_cast_or_null<wk::NewOp>(op))
      if (fresh.getInitial()) return remember(step(fresh.getInitial(), depth + 1));
    if (auto binary = dyn_cast_or_null<wk::BinaryOp>(op)) {
      auto left = step(binary.getLhs(), depth + 1), right = step(binary.getRhs(), depth + 1);
      Bounds a = range(binary.getLhs()), b = range(binary.getRhs());
      // Only unit/invariant additions are needed for memory packets. Requiring
      // nonnegative ranges also handles signed reinterpretation at the final
      // pointer-width extension, rather than trusting a modular stride alone.
      if (left && right && binary.getKind() == "add" && *left >= 0 && *right >= 0 &&
          *left <= 1 && *right <= 1 && a.high <= limit && b.high <= limit - a.high)
        return remember(*left + *right);
    }
    return remember({});
  }

public:
  explicit IndexRelations(int64_t axis) : axis(axis) {}
  bool unit(Value value) { return step(value) == std::optional<int64_t>(1); }
  bool below(Value value, uint64_t extent) { return extent && range(value).high < extent; }
};
} // namespace

bool weft::x86::hasNonWrappingUnitStride(Value value, int64_t axis) {
  return IndexRelations(axis).unit(value);
}
bool weft::x86::hasUnsignedBound(Value value, uint64_t extent) {
  return IndexRelations(0).below(value, extent);
}
