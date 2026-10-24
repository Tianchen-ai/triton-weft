#include "Weft/Analysis/IntegerFacts.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>

using namespace mlir;
using namespace weft::analysis;
namespace wk = weft::kernel;
namespace {
using Kind = IntegerExpressionKind;

std::optional<IntegerRange> representable(__int128 minimum, __int128 maximum) {
  if (minimum > maximum || minimum < INT64_MIN || maximum > INT64_MAX)
    return std::nullopt;
  return IntegerRange{static_cast<int64_t>(minimum), static_cast<int64_t>(maximum)};
}

IntegerType interpreted(Type type, IntegerType context) {
  auto integer = integerElementType(type);
  if (integer && integer.isSignless() && context)
    return IntegerType::get(integer.getContext(), integer.getWidth(), context.getSignedness());
  return integer;
}

std::optional<IntegerRange> constantRange(Attribute attribute, IntegerType type) {
  if (!integerTypeRange(type)) return std::nullopt;
  auto number = [&](APInt value) {
    return type.isSigned() ? value.getSExtValue() : static_cast<int64_t>(value.getZExtValue());
  };
  if (auto scalar = dyn_cast_or_null<IntegerAttr>(attribute)) {
    int64_t value = number(scalar.getValue());
    return IntegerRange{value, value};
  }
  if (auto elements = dyn_cast_or_null<DenseIntElementsAttr>(attribute)) {
    if (elements.empty()) return std::nullopt;
    int64_t low = INT64_MAX, high = INT64_MIN;
    for (APInt value : elements.getValues<APInt>()) {
      int64_t n = number(value);
      low = std::min(low, n); high = std::max(high, n);
    }
    return IntegerRange{low, high};
  }
  return std::nullopt;
}

IntegerExpression describe(Value value, IntegerType type) {
  IntegerExpression expression{type};
  auto unary = [&](Value input, Kind kind = Kind::Preserve) {
    expression.kind = kind; expression.inputs = {input};
  };
  auto pair = [&](Value lhs, Value rhs, Kind kind) {
    expression.kind = kind; expression.inputs = {lhs, rhs};
  };
  auto known = [&](Attribute attr) {
    expression.kind = Kind::Known; expression.known = constantRange(attr, type);
  };
  if (auto op = value.getDefiningOp<arith::ConstantOp>()) known(op.getValue());
  else if (auto op = value.getDefiningOp<wk::ConstantOp>()) known(op.getValue());
  else if (auto op = value.getDefiningOp<wk::IotaOp>()) {
    expression.kind = Kind::Known;
    if (op.getEnd() > op.getStart() && op.getEnd() - 1 <= uint64_t(INT64_MAX))
      expression.known = IntegerRange{static_cast<int64_t>(op.getStart()), static_cast<int64_t>(op.getEnd() - 1)};
  } else if (auto op = value.getDefiningOp<arith::ExtUIOp>()) {
    unary(op.getIn(), Kind::Cast);
    auto input = integerElementType(op.getIn().getType());
    expression.interpretations = {IntegerType::get(type.getContext(), input.getWidth(), IntegerType::Unsigned)};
  } else if (auto op = value.getDefiningOp<arith::ExtSIOp>()) {
    unary(op.getIn(), Kind::Cast);
    auto input = integerElementType(op.getIn().getType());
    expression.interpretations = {IntegerType::get(type.getContext(), input.getWidth(), IntegerType::Signed)};
  } else if (auto op = value.getDefiningOp<arith::TruncIOp>()) unary(op.getIn(), Kind::Cast);
  else if (auto op = value.getDefiningOp<wk::WidenOp>()) unary(op.getInput(), Kind::Cast);
  else if (auto op = value.getDefiningOp<wk::CastOp>()) unary(op.getInput(), Kind::Cast);
  else if (auto op = value.getDefiningOp<wk::NarrowOp>()) unary(op.getInput(), op.getSaturate() ? Kind::Saturate : Kind::Cast);
  else if (auto op = value.getDefiningOp<wk::ExtractOp>()) unary(op.getInput());
  else if (auto op = value.getDefiningOp<vector::BroadcastOp>()) unary(op.getSource());
  else if (auto op = value.getDefiningOp<vector::ExtractStridedSliceOp>()) unary(op.getSource());
  else if (auto op = value.getDefiningOp<vector::ShapeCastOp>()) unary(op.getSource());
  else if (auto op = value.getDefiningOp<vector::ShuffleOp>()) {
    int64_t firstSize = cast<VectorType>(op.getV1().getType()).getShape().front();
    bool first = false, second = false, unknown = false;
    for (int64_t index : op.getMask()) {
      unknown |= index < 0;
      first |= index >= 0 && index < firstSize;
      second |= index >= firstSize;
    }
    if (!unknown) {
      if (first && second) pair(op.getV1(), op.getV2(), Kind::Union);
      else if (first || second) unary(first ? op.getV1() : op.getV2());
    }
  }
  else if (auto op = value.getDefiningOp<vector::InsertStridedSliceOp>()) pair(op.getValueToStore(), op.getDest(), Kind::Union);
  else if (auto op = value.getDefiningOp<arith::SelectOp>()) pair(op.getTrueValue(), op.getFalseValue(), Kind::Union);
  else if (auto op = value.getDefiningOp<wk::SelectOp>()) pair(op.getTrueValue(), op.getFalseValue(), Kind::Union);
  else if (auto op = value.getDefiningOp<wk::BinaryOp>()) pair(op.getLhs(), op.getRhs(), integerBinaryKind(op.getKind()));
  else if (auto op = value.getDefiningOp<wk::MacGroupsOp>()) {
    if (op.getOverflow() == "wrap") {
      pair(op.getLhs(), op.getRhs(), Kind::ProductSum); expression.group = op.getGroup();
    }
  } else if (auto *op = value.getDefiningOp(); op && op->getNumOperands() == 2) {
    auto kind = llvm::StringSwitch<Kind>(op->getName().getStringRef())
        .Case("arith.addi", Kind::Add).Case("arith.subi", Kind::Sub)
        .Case("arith.muli", Kind::Mul).Case("arith.andi", Kind::And)
        .Case("arith.ori", Kind::Or).Case("arith.shli", Kind::ShiftLeft)
        .Case("arith.shrui", Kind::ShiftRight).Case("arith.divui", Kind::Divide)
        .Default(Kind::Unknown);
    if (kind != Kind::Unknown) pair(op->getOperand(0), op->getOperand(1), kind);
  }
  return expression;
}

class Query {
public:
  explicit Query(const IntegerDescription &target) : target(target) {}
  std::optional<IntegerRange> get(Value value, IntegerType requested, unsigned depth = 0) {
    auto custom = target ? target(value) : std::optional<IntegerExpression>();
    IntegerType type = requested ? requested : custom ? custom->type : integerElementType(value.getType());
    auto full = integerTypeRange(type);
    if (!full || depth > 16 || visits++ >= 512) return full;
    auto key = std::make_pair(value, type);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    IntegerExpression expression = custom ? *custom : describe(value, type);
    auto fit = [&](std::optional<IntegerRange> range) -> std::optional<IntegerRange> {
      return range && range->fits(type) ? range : full;
    };
    auto evaluate = [&]() -> std::optional<IntegerRange> {
      if (expression.kind == Kind::Known) return fit(expression.known);
      if (expression.kind == Kind::Unknown || expression.inputs.empty()) return full;
      SmallVector<IntegerRange, 3> inputs;
      for (auto [i, input] : llvm::enumerate(expression.inputs)) {
        IntegerType inputType = i < expression.interpretations.size() ? expression.interpretations[i] : IntegerType();
        if (!inputType) inputType = interpreted(input.getType(), type);
        auto range = get(input, inputType, depth + 1);
        if (!range) return full;
        inputs.push_back(*range);
      }
      IntegerRange a = inputs.front();
      if (expression.kind == Kind::Preserve || expression.kind == Kind::Cast) return fit(a);
      if (expression.kind == Kind::ZeroOr)
        return fit(IntegerRange{std::min(int64_t{0}, a.minimum), std::max(int64_t{0}, a.maximum)});
      if (expression.kind == Kind::Saturate)
        return IntegerRange{std::clamp(a.minimum, full->minimum, full->maximum),
                            std::clamp(a.maximum, full->minimum, full->maximum)};
      if (expression.kind == Kind::Sum)
        return fit(productSumRange(a, {1, 1}, expression.group));
      if (expression.kind == Kind::ProductSum && inputs.size() == 3) {
        auto products = productSumRange(a, inputs[1], expression.group);
        if (!products) return full;
        return fit(representable(__int128(products->minimum) + inputs[2].minimum,
                                 __int128(products->maximum) + inputs[2].maximum));
      }
      if (inputs.size() != 2) return full;
      IntegerRange c = inputs[1];
      switch (expression.kind) {
      case Kind::Union: return fit(IntegerRange{std::min(a.minimum, c.minimum), std::max(a.maximum, c.maximum)});
      case Kind::Add: return fit(representable(__int128(a.minimum) + c.minimum, __int128(a.maximum) + c.maximum));
      case Kind::Sub: return fit(representable(__int128(a.minimum) - c.maximum, __int128(a.maximum) - c.minimum));
      case Kind::Mul: return fit(productSumRange(a, c, 1));
      case Kind::ProductSum: return fit(productSumRange(a, c, expression.group));
      case Kind::And:
        if (a.minimum >= 0 && c.minimum >= 0) return fit(IntegerRange{0, std::min(a.maximum, c.maximum)});
        if (a.minimum == a.maximum && a.minimum >= 0) return fit(IntegerRange{0, a.maximum});
        if (c.minimum == c.maximum && c.minimum >= 0) return fit(IntegerRange{0, c.maximum});
        break;
      case Kind::Or:
        if (a.minimum >= 0 && c.minimum >= 0) {
          unsigned width = std::max(llvm::bit_width(uint64_t(a.maximum)), llvm::bit_width(uint64_t(c.maximum)));
          return fit(IntegerRange{0, static_cast<int64_t>(llvm::maskTrailingOnes<uint64_t>(width))});
        }
        break;
      case Kind::ShiftLeft:
        if (a.minimum >= 0 && c.minimum == c.maximum && c.minimum >= 0 && c.maximum < type.getWidth())
          return fit(representable(__int128(a.minimum) << c.minimum, __int128(a.maximum) << c.minimum));
        break;
      case Kind::ShiftRight:
        if (a.minimum >= 0 && c.minimum == c.maximum && c.minimum >= 0 && c.maximum < type.getWidth())
          return fit(IntegerRange{a.minimum >> c.minimum, a.maximum >> c.minimum});
        break;
      case Kind::Divide:
        if (a.minimum >= 0 && c.minimum > 0) return fit(IntegerRange{a.minimum / c.maximum, a.maximum / c.minimum});
        break;
      default: break;
      }
      return full;
    };
    auto result = evaluate();
    cache[key] = result;
    return result;
  }
private:
  const IntegerDescription &target;
  unsigned visits = 0;
  llvm::DenseMap<std::pair<Value, IntegerType>, std::optional<IntegerRange>> cache;
};
} // namespace

IntegerType weft::analysis::integerElementType(Type type) {
  if (auto value = dyn_cast<wk::ValueType>(type)) type = value.getElementType();
  else if (auto shaped = dyn_cast<ShapedType>(type)) type = shaped.getElementType();
  return dyn_cast<IntegerType>(type);
}

std::optional<IntegerRange> weft::analysis::integerTypeRange(IntegerType type) {
  if (!type || type.isSignless() || !type.getWidth() || type.getWidth() >= 63) return std::nullopt;
  int64_t limit = int64_t{1} << (type.getWidth() - (type.isSigned() ? 1 : 0));
  return IntegerRange{type.isSigned() ? -limit : 0, limit - 1};
}

bool IntegerRange::fits(IntegerType type) const {
  auto domain = integerTypeRange(type);
  return domain && minimum >= domain->minimum && maximum <= domain->maximum && minimum <= maximum;
}

std::optional<IntegerRange> weft::analysis::productSumRange(IntegerRange lhs, IntegerRange rhs, int64_t group) {
  if (group <= 0 || lhs.minimum > lhs.maximum || rhs.minimum > rhs.maximum) return std::nullopt;
  const __int128 products[] = {__int128(lhs.minimum) * rhs.minimum, __int128(lhs.minimum) * rhs.maximum,
                             __int128(lhs.maximum) * rhs.minimum, __int128(lhs.maximum) * rhs.maximum};
  __int128 low, high;
  if (__builtin_mul_overflow(*std::min_element(std::begin(products), std::end(products)), __int128(group), &low) ||
      __builtin_mul_overflow(*std::max_element(std::begin(products), std::end(products)), __int128(group), &high))
    return std::nullopt;
  return representable(low, high);
}

bool weft::analysis::saturatingProductSumIsExact(IntegerRange lhs, IntegerRange rhs, int64_t group, IntegerType accumulator) {
  auto result = productSumRange(lhs, rhs, group);
  return result && result->fits(accumulator);
}

IntegerExpressionKind weft::analysis::integerBinaryKind(llvm::StringRef name) {
  return llvm::StringSwitch<Kind>(name).Case("add", Kind::Add).Case("sub", Kind::Sub)
      .Case("mul", Kind::Mul).Case("and", Kind::And).Case("or", Kind::Or)
      .Case("shl", Kind::ShiftLeft).Case("shr", Kind::ShiftRight).Case("div", Kind::Divide)
      .Default(Kind::Unknown);
}

std::optional<IntegerRange> weft::analysis::integerRange(Value value, IntegerType interpretation,
                                                        const IntegerDescription &targetDescription) {
  return Query(targetDescription).get(value, interpretation);
}
