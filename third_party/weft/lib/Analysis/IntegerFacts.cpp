#include "Weft/Analysis/IntegerFacts.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
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

using Bound = llvm::APSInt;

Bound signedBound(Bound value) {
  return value.isUnsigned() ? Bound(value.zext(value.getBitWidth() + 1), false) : value;
}

Bound low(const Bound &a, const Bound &b) {
  return Bound::compareValues(a, b) <= 0 ? a : b;
}
Bound high(const Bound &a, const Bound &b) {
  return Bound::compareValues(a, b) >= 0 ? a : b;
}
Bound add(const Bound &a, const Bound &b) {
  unsigned bits = std::max(a.getBitWidth(), b.getBitWidth()) + 1;
  return a.extend(bits) + b.extend(bits);
}
Bound subtract(const Bound &a, const Bound &b) {
  unsigned bits = std::max(a.getBitWidth(), b.getBitWidth()) + 1;
  return a.extend(bits) - b.extend(bits);
}
Bound multiply(const Bound &a, const Bound &b) {
  unsigned bits = a.getBitWidth() + b.getBitWidth();
  return a.extend(bits) * b.extend(bits);
}

IntegerType interpreted(Type type, IntegerType context) {
  // Index has no language-level signedness/width. A target requesting a
  // concrete index interpretation supplies that context explicitly.
  if (type.isIndex() && context) return context;
  auto integer = integerElementType(type);
  if (integer && integer.isSignless() && context)
    return IntegerType::get(integer.getContext(), integer.getWidth(), context.getSignedness());
  return integer;
}

std::optional<IntegerRange> constantRange(Attribute attribute, IntegerType type) {
  if (!integerTypeRange(type)) return std::nullopt;
  auto number = [&](APInt value) {
    value = type.isSigned() ? value.sextOrTrunc(type.getWidth()) : value.zextOrTrunc(type.getWidth());
    return signedBound(Bound(value, !type.isSigned()));
  };
  if (auto scalar = dyn_cast_or_null<IntegerAttr>(attribute)) {
    Bound value = number(scalar.getValue());
    return IntegerRange{value, value};
  }
  if (auto elements = dyn_cast_or_null<DenseIntElementsAttr>(attribute)) {
    if (elements.empty()) return std::nullopt;
    std::optional<IntegerRange> result;
    for (APInt value : elements.getValues<APInt>()) {
      Bound n = number(value);
      result = result ? IntegerRange{low(result->minimum, n), high(result->maximum, n)} :
                        IntegerRange{n, n};
    }
    return result;
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
    if (op.getEnd() > op.getStart())
      expression.known = IntegerRange{Bound::getUnsigned(op.getStart()), Bound::getUnsigned(op.getEnd() - 1)};
  } else if (auto op = value.getDefiningOp<wk::WidenOp>()) unary(op.getInput(), Kind::Cast);
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
  else if (auto op = value.getDefiningOp<wk::SelectOp>()) pair(op.getTrueValue(), op.getFalseValue(), Kind::Union);
  else if (auto op = value.getDefiningOp<wk::BinaryOp>()) pair(op.getLhs(), op.getRhs(), integerBinaryKind(op.getKind()));
  else if (auto op = value.getDefiningOp<wk::MacGroupsOp>()) {
    if (op.getOverflow() == "wrap") {
      pair(op.getLhs(), op.getRhs(), Kind::ProductSum); expression.group = op.getGroup();
    }
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
    // Standard MLIR operations already define signed/unsigned, wrap and shift
    // behavior. Reuse that interface instead of another arithmetic interpreter.
    if (!custom && expression.kind != Kind::Known)
      if (auto inferred = infer(value, type, depth)) {
        cache[key] = inferred;
        return inferred;
      }
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
        return fit(IntegerRange{low(Bound::get(0), a.minimum), high(Bound::get(0), a.maximum)});
      if (expression.kind == Kind::Saturate)
        return IntegerRange{low(high(a.minimum, full->minimum), full->maximum),
                            low(high(a.maximum, full->minimum), full->maximum)};
      if (expression.kind == Kind::Sum)
        return fit(productSumRange(a, {1, 1}, expression.group));
      if (expression.kind == Kind::ProductSum && inputs.size() == 3) {
        auto products = productSumRange(a, inputs[1], expression.group);
        if (!products) return full;
        return fit(IntegerRange{add(products->minimum, inputs[2].minimum),
                                add(products->maximum, inputs[2].maximum)});
      }
      if (inputs.size() != 2) return full;
      IntegerRange c = inputs[1];
      switch (expression.kind) {
      case Kind::Union: return fit(IntegerRange{low(a.minimum, c.minimum), high(a.maximum, c.maximum)});
      case Kind::Add: return fit(IntegerRange{add(a.minimum, c.minimum), add(a.maximum, c.maximum)});
      case Kind::Sub: return fit(IntegerRange{subtract(a.minimum, c.maximum), subtract(a.maximum, c.minimum)});
      case Kind::Mul: return fit(productSumRange(a, c, 1));
      case Kind::ProductSum: return fit(productSumRange(a, c, expression.group));
      case Kind::And:
        if (a.minimum >= 0 && c.minimum >= 0) return fit(IntegerRange{Bound::get(0), low(a.maximum, c.maximum)});
        if (a.minimum == a.maximum && a.minimum >= 0) return fit(IntegerRange{Bound::get(0), a.maximum});
        if (c.minimum == c.maximum && c.minimum >= 0) return fit(IntegerRange{Bound::get(0), c.maximum});
        break;
      case Kind::Or:
        if (a.minimum >= 0 && c.minimum >= 0) {
          unsigned width = std::max(a.maximum.getActiveBits(), c.maximum.getActiveBits());
          return fit(IntegerRange{Bound::get(0), Bound(APInt::getLowBitsSet(width + 1, width), false)});
        }
        break;
      case Kind::ShiftLeft:
        if (a.minimum >= 0 && c.minimum == c.maximum && c.minimum >= 0 && c.maximum < type.getWidth())
          return fit(IntegerRange{a.minimum.extend(a.minimum.getBitWidth() + c.minimum.getZExtValue()) << c.minimum.getZExtValue(),
                                  a.maximum.extend(a.maximum.getBitWidth() + c.minimum.getZExtValue()) << c.minimum.getZExtValue()});
        break;
      case Kind::ShiftRight:
        if (a.minimum >= 0 && c.minimum == c.maximum && c.minimum >= 0 && c.maximum < type.getWidth())
          return fit(IntegerRange{a.minimum >> c.minimum.getZExtValue(), a.maximum >> c.minimum.getZExtValue()});
        break;
      case Kind::Divide:
        if (a.minimum >= 0 && c.minimum > 0) {
          unsigned bits = std::max(a.maximum.getBitWidth(), c.maximum.getBitWidth());
          return fit(IntegerRange{a.minimum.extend(bits) / c.maximum.extend(bits),
                                  a.maximum.extend(bits) / c.minimum.extend(bits)});
        }
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
  std::optional<IntegerRange> infer(Value value, IntegerType type, unsigned depth) {
    auto result = dyn_cast<OpResult>(value);
    auto interface = result ? dyn_cast<InferIntRangeInterface>(result.getOwner()) : InferIntRangeInterface();
    if (!interface) return std::nullopt;
    SmallVector<ConstantIntRanges> operands;
    for (Value operand : result.getOwner()->getOperands()) {
      unsigned bits = ConstantIntRanges::getStorageBitwidth(operand.getType());
      auto operandType = interpreted(operand.getType(), type);
      auto range = operandType ? get(operand, operandType, depth + 1) : std::nullopt;
      if (!range) operands.push_back(ConstantIntRanges::maxRange(bits));
      else operands.push_back(ConstantIntRanges::range(range->minimum.extOrTrunc(bits),
           range->maximum.extOrTrunc(bits), operandType.isSigned()));
    }
    std::optional<IntegerRange> answer;
    interface.inferResultRanges(operands, [&](Value inferred, const ConstantIntRanges &range) {
      if (inferred != value) return;
      answer = IntegerRange{Bound(type.isSigned() ? range.smin() : range.umin(), !type.isSigned()),
                            Bound(type.isSigned() ? range.smax() : range.umax(), !type.isSigned())};
    });
    return answer;
  }
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
  if (!type || type.isSignless() || !type.getWidth()) return std::nullopt;
  unsigned bits = type.getWidth();
  return type.isSigned() ? IntegerRange{Bound(APInt::getSignedMinValue(bits), false),
                                        Bound(APInt::getSignedMaxValue(bits), false)} :
                          IntegerRange{Bound::get(0), Bound(APInt::getMaxValue(bits), true)};
}

IntegerRange::IntegerRange(int64_t minimum, int64_t maximum)
    : IntegerRange(Bound::get(minimum), Bound::get(maximum)) {}

IntegerRange::IntegerRange(Bound low, Bound high)
    : minimum(signedBound(std::move(low))), maximum(signedBound(std::move(high))) {
  // One spare sign bit also makes negating a bound safe for magnitude queries.
  unsigned bits = std::max(minimum.getSignificantBits(), maximum.getSignificantBits()) + 1;
  minimum = minimum.extOrTrunc(bits);
  maximum = maximum.extOrTrunc(bits);
}

std::optional<IntegerRange> weft::analysis::integerConstantRange(Attribute attr, IntegerType type) {
  return constantRange(attr, type);
}

bool IntegerRange::fits(IntegerType type) const {
  auto domain = integerTypeRange(type);
  return domain && Bound::compareValues(minimum, domain->minimum) >= 0 &&
      Bound::compareValues(maximum, domain->maximum) <= 0 && minimum <= maximum;
}

std::optional<IntegerRange> weft::analysis::productSumRange(IntegerRange lhs, IntegerRange rhs, int64_t group) {
  if (group <= 0 || lhs.minimum > lhs.maximum || rhs.minimum > rhs.maximum) return std::nullopt;
  const Bound products[] = {multiply(lhs.minimum, rhs.minimum), multiply(lhs.minimum, rhs.maximum),
                            multiply(lhs.maximum, rhs.minimum), multiply(lhs.maximum, rhs.maximum)};
  return IntegerRange{multiply(*std::min_element(std::begin(products), std::end(products)), Bound::get(group)),
                      multiply(*std::max_element(std::begin(products), std::end(products)), Bound::get(group))};
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
