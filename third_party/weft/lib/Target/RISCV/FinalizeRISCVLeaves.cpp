#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "PredicateRegisters.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/Numerics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>

using namespace weft;

namespace {

std::optional<uint64_t> indexResidue(mlir::Value value, uint64_t modulus,
                                     unsigned &remaining) {
  if (!remaining || !value.getType().isIndex() || !modulus ||
      (modulus & (modulus - 1)))
    return std::nullopt;
  --remaining;
  const uint64_t mask = modulus - 1;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return static_cast<uint64_t>(constant.value()) & mask;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
    if (loop && argument == loop.getInductionVar()) {
      auto step = indexResidue(loop.getStep(), modulus, remaining);
      if (step == 0)
        return indexResidue(loop.getLowerBound(), modulus, remaining);
    }
    return std::nullopt;
  }
  auto *op = value.getDefiningOp();
  if (!op || !mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                        mlir::arith::MulIOp, mlir::arith::RemUIOp>(op))
    return std::nullopt;
  if (auto remainder = mlir::dyn_cast<mlir::arith::RemUIOp>(op)) {
    auto divisor = remainder.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (divisor && divisor.value() > 0 &&
        static_cast<uint64_t>(divisor.value()) % modulus == 0)
      return indexResidue(remainder.getLhs(), modulus, remaining);
    return std::nullopt;
  }
  auto lhs = indexResidue(op->getOperand(0), modulus, remaining);
  auto rhs = indexResidue(op->getOperand(1), modulus, remaining);
  if (mlir::isa<mlir::arith::MulIOp>(op) && (lhs == 0 || rhs == 0))
    return 0;
  if (!lhs || !rhs)
    return std::nullopt;
  // These are low bits of RISC-V's 64-bit index arithmetic. Unsigned wrap
  // preserves a power-of-two residue, including add/sub/mul overflow.
  return (mlir::isa<mlir::arith::AddIOp>(op) ? *lhs + *rhs
          : mlir::isa<mlir::arith::SubIOp>(op) ? *lhs - *rhs
                                            : *lhs * *rhs) & mask;
}

void foldConstantReplicaLayers(mlir::IRRewriter &rewriter, mlir::ModuleOp module) {
  module.walk([&](riscv::RVVReplicaStorageLoadOp load) {
    auto kernel = load->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() ||
        load.getAccess().getMapping() != "grouped_layered") return;
    auto base = riscv_internal::constantInt(load.getLogicalBase());
    auto plan = load.getPlan();
    const int64_t group = load.getAccess().getGroupSize();
    const int64_t layer = load.getAccess().getLayerSize();
    if (!base || *base < 0 || group <= 0 || layer <= 0 || group % layer ||
        *base % layer || plan.getProjectionExtent() > layer ||
        llvm::any_of(load.getLayerForPart(), [](int64_t value) { return value != 0; }))
      return;
    const int64_t logicalLayer = (*base % group) / layer;
    if (logicalLayer == 0 && plan.getOffsetAlignment() % group == 0) return;
    const int64_t bits = plan.getElementBits();
    const int64_t physicalLayer = load.getAccess().getOrder() == "lo_first"
                                      ? logicalLayer : group / layer - 1 - logicalLayer;
    const int64_t mask = (physicalLayer + 1) * bits == 8 ? 0 : (int64_t{1} << bits) - 1;
    // Index residue folding exposes constant layer bases after issue unroll.
    // Move only the layer coordinate into the existing decode map; the byte
    // address and read snapshot stay unchanged at the complete group origin.
    rewriter.setInsertionPoint(load);
    const int64_t groupBase = *base - *base % group;
    mlir::Value alignedBase;
    if (load.getLogicalBase().getType().isIndex())
      alignedBase = mlir::arith::ConstantIndexOp::create(rewriter, load.getLoc(), groupBase);
    else {
      auto type = mlir::cast<mlir::IntegerType>(load.getLogicalBase().getType());
      alignedBase = riscv::ConstantOp::create(rewriter,
          load.getLoc(), type, rewriter.getIntegerAttr(type, groupBase));
    }
    auto alignedPlan = riscv::StorageWindowPlanAttr::get(
        rewriter.getContext(), plan.getKind(), plan.getReductionAxis(),
        plan.getProjectionBase(), plan.getProjectionStride(), plan.getProjectionRepeat(),
        plan.getProjectionExtent(), group, plan.getRecordElements(),
        plan.getByteOffset(), bits, plan.getGroupSize(), plan.getLayerSize(),
        plan.getPhysicalLayerBase(), plan.getPhysicalLayerStep(), plan.getShiftBase(),
        plan.getShiftStep(), plan.getMaskValue());
    auto parts = [&](int64_t value) {
      return rewriter.getDenseI64ArrayAttr(
          llvm::SmallVector<int64_t>(load.getLayerForPart().size(), value));
    };
    rewriter.modifyOpInPlace(load, [&] {
      load.getLogicalBaseMutable().assign(alignedBase);
      load.setPlanAttr(alignedPlan);
      load.setLayerForPartAttr(parts(logicalLayer));
      load.setPhysicalLayerForPartAttr(parts(physicalLayer));
      load.setShiftOffsetForPartAttr(parts(physicalLayer * bits));
      load.setShiftBaseFactorForPartAttr(parts(0));
      load.setMaskValueForPartAttr(parts(mask));
    });
  });
}

void foldPhysicalIndexRemainders(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<mlir::Operation *> candidates;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                  mlir::arith::MulIOp, mlir::arith::DivUIOp,
                  mlir::arith::RemUIOp>(op) && op->getResult(0).getType().isIndex())
      candidates.push_back(op);
  });
  unsigned folded = 0;
  for (auto *op : candidates) {
    auto kernel = op->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      continue;
    rewriter.setInsertionPoint(op);
    if (auto remainder = mlir::dyn_cast<mlir::arith::RemUIOp>(op)) {
      auto divisor = remainder.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
      unsigned remaining = 64;
      if (divisor && divisor.value() > 0)
        if (auto residue = indexResidue(remainder.getLhs(), divisor.value(), remaining)) {
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(op, *residue);
          ++folded;
          continue;
        }
    }
    llvm::SmallVector<mlir::Attribute> operands;
    for (mlir::Value operand : op->getOperands()) {
      auto constant = operand.getDefiningOp<mlir::arith::ConstantOp>();
      if (!constant)
        break;
      operands.push_back(constant.getValue());
    }
    if (operands.size() != op->getNumOperands())
      continue;
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::succeeded(op->fold(operands, results)) && results.size() == 1)
      if (auto attribute = mlir::dyn_cast_if_present<mlir::Attribute>(results.front()))
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op->getResult(0).getType(), mlir::cast<mlir::TypedAttr>(attribute));
  }
  if (folded)
    llvm::errs() << "weft-index-residue: constant-remainders=" << folded
                 << " visits-per-query<=64\n";
  foldConstantReplicaLayers(rewriter, module);
}

bool isIntegerZero(mlir::Value value, unsigned &remaining) {
  if (!remaining ||
      !mlir::isa<mlir::IntegerType>(
          riscv_internal::logicalElement(value.getType())))
    return false;
  --remaining;
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    attribute = constant.getValue();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  if (auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute))
    return integer.getValue().isZero();
  auto *producer = value.getDefiningOp();
  if (!producer)
    return false;
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(producer))
    return conversion.getConversion().getEffect() == "pure" &&
           isIntegerZero(conversion.getInput(), remaining);
  if (mlir::isa<riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                riscv::RVVSplatOp, riscv::RVVAxisBroadcastOp>(producer))
    return isIntegerZero(producer->getOperand(0), remaining);
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(producer))
    if (binary.getKind() == "mul")
      return isIntegerZero(binary.getLhs(), remaining) ||
             isIntegerZero(binary.getRhs(), remaining);
  return false;
}

void foldIntegerIdentities(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  module.walk([&](riscv::BinaryOp operation) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (!kernel || kernel.getResourcesMaterialized() ||
        !result || result.getLayout().getCarrier() != "rvv" ||
        (operation.getKind() != "add" && operation.getKind() != "sub"))
      return;
    auto tryIdentity = [&](mlir::Value input, mlir::Value zero) {
      // Equality includes all axes, validity, and carrier factors. Removing an
      // integer identity must not silently remove an author's broadcast axis.
      unsigned remaining = 32;
      if (input.getType() != operation.getResult().getType() ||
          !isIntegerZero(zero, remaining))
        return false;
      rewriter.replaceOp(operation, input);
      return true;
    };
    if (tryIdentity(operation.getLhs(), operation.getRhs()))
      return;
    if (operation.getKind() == "add")
      tryIdentity(operation.getRhs(), operation.getLhs());
  });
  module.walk([&](riscv::RVVIntegerMACOp operation) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    auto instruction = operation.getLeaf().getInstruction();
    unsigned remaining = 32;
    if (!kernel || kernel.getResourcesMaterialized() ||
        (instruction != "rvv.vwmacc.vv" && instruction != "rvv.vwmaccu.vv") ||
        riscv_internal::logicalBitWidth(operation.getVector().getType()) > 16 ||
        !isIntegerZero(operation.getAccumulator(), remaining)) return;
    auto type = operation.getResult().getType();
    auto multiply = instruction == "rvv.vwmacc.vv" ? "rvv.vwmul.vv" : "rvv.vwmulu.vv";
    rewriter.setInsertionPoint(operation);
    auto product = riscv::RVVWidenMultiplyOp::create(rewriter, operation.getLoc(), type,
        operation.getVector(), operation.getMultiplier(),
        riscv_internal::leaf(rewriter, "rvv", "widen-multiply", multiply, multiply,
            riscv_internal::issueRegisterGroups(operation.getVector()) +
                riscv_internal::issueRegisterGroups(operation.getMultiplier()),
            type.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
    riscv_internal::copyOrigin(operation, product);
    rewriter.replaceOp(operation, product.getResult());
  });
}

void foldIntegerPowerOfTwoArithmetic(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  unsigned quotients = 0, remainders = 0, products = 0;
  llvm::SmallVector<riscv::BinaryOp> candidates;
  module.walk([&](riscv::BinaryOp op) { candidates.push_back(op); });
  for (auto operation : candidates) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    auto element = result
        ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
        : mlir::IntegerType();
    const bool multiply = operation.getKind() == "mul";
    if (!kernel || kernel.getResourcesMaterialized() || !element ||
        (!element.isUnsigned() && !multiply) || result.getLayout().getCarrier() != "rvv" ||
        (!multiply && operation.getKind() != "div" && operation.getKind() != "mod"))
      continue;
    mlir::Value vector = operation.getLhs(), scalar = operation.getRhs();
    if (multiply && vector.getType() == element && scalar.getType() == result)
      std::swap(vector, scalar);
    if (vector.getType() != result || scalar.getType() != element)
      continue;
    mlir::Attribute attribute;
    if (auto constant = scalar.getDefiningOp<riscv::ConstantOp>())
      attribute = constant.getValue();
    if (auto constant = scalar.getDefiningOp<mlir::arith::ConstantOp>())
      attribute = constant.getValue();
    auto divisor = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
    if (!divisor || !divisor.getValue().isPowerOf2())
      continue;
    const bool quotient = operation.getKind() == "div";
    // Left shift and multiplication by a single bit have identical low bits
    // for both signed and unsigned wrap arithmetic, including the sign bit.
    llvm::APInt immediate = (quotient || multiply)
        ? llvm::APInt(element.getWidth(), divisor.getValue().logBase2())
        : divisor.getValue() - 1;
    rewriter.setInsertionPoint(operation);
    auto constant = riscv::ConstantOp::create(rewriter,
        operation.getLoc(), element, rewriter.getIntegerAttr(element, immediate));
    llvm::StringRef instruction = multiply ? "rvv.vsll.vx" : quotient ? "rvv.vsrl.vx" : "rvv.vand.vx";
    auto replacement = riscv::BinaryOp::create(rewriter,
        operation.getLoc(), result, vector, constant.getResult(),
        multiply ? "shl" : quotient ? "shr" : "and",
        riscv_internal::leaf(
            rewriter, "rvv", "pointwise", instruction, instruction,
            result.getLayout().getRegisterGroups(),
            result.getLayout().getRegisterGroups(), 0, 0, "none", "agnostic"));
    riscv_internal::copyOrigin(operation, replacement);
    rewriter.replaceOp(operation, replacement.getResult());
    multiply ? ++products : quotient ? ++quotients : ++remainders;
  }
  if (quotients || remainders || products)
    llvm::errs() << "weft-constant-arithmetic: power-of-two-quotients=" << quotients
                 << " power-of-two-remainders=" << remainders
                 << " power-of-two-products=" << products << '\n';
}

void selectUnsignedMultiplyHigh(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::NarrowOp> candidates;
  module.walk([&](riscv::NarrowOp op) { candidates.push_back(op); });
  for (auto narrow : candidates) {
    auto kernel = narrow->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() || narrow.getSaturate() ||
        (narrow.getRounding() != "rtz" && narrow.getRounding() != "none"))
      continue;
    auto shift = narrow.getInput().getDefiningOp<riscv::BinaryOp>();
    if (!shift || shift.getKind() != "shr" ||
        !shift.getResult().hasOneUse())
      continue;
    auto multiply =
        shift.getLhs().getDefiningOp<riscv::RVVWidenScalarMultiplyOp>();
    if (!multiply || !multiply.getResult().hasOneUse())
      continue;
    auto input = multiply.getLhs().getType();
    auto element = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
    mlir::Attribute amount;
    if (auto constant = shift.getRhs().getDefiningOp<riscv::ConstantOp>())
      amount = constant.getValue();
    if (auto constant = shift.getRhs().getDefiningOp<mlir::arith::ConstantOp>())
      amount = constant.getValue();
    auto bits = mlir::dyn_cast_or_null<mlir::IntegerAttr>(amount);
    if (!element || !element.isUnsigned() ||
        (element.getWidth() != 8 && element.getWidth() != 16) ||
        narrow.getResult().getType() != input ||
        shift.getResult().getType() != multiply.getResult().getType() ||
        multiply.getLeaf().getInstruction() != "rvv.vwmulu.vx" ||
        !bits || bits.getValue().getLimitedValue() != element.getWidth())
      continue;
    // The doubled-width unsigned product cannot overflow; taking its upper
    // half needs neither rounding nor saturation. The preceding byte wrap,
    // every logical axis, and every later accumulation remain unchanged.
    rewriter.setInsertionPoint(narrow);
    auto high = riscv::RVVMultiplyHighScalarOp::create(rewriter,
        narrow.getLoc(), input, multiply.getLhs(), multiply.getRhs(),
        riscv_internal::leaf(
            rewriter, "rvv", "multiply-high-scalar", "rvv.vmulhu.vx",
            "rvv.vmulhu.vx", input.getLayout().getRegisterGroups(),
            input.getLayout().getRegisterGroups()));
    riscv_internal::copyOrigin(narrow, high);
    rewriter.replaceOp(narrow, high.getResult());
    rewriter.eraseOp(shift);
    rewriter.eraseOp(multiply);
  }
}

bool addsOnlySingletonAxes(riscv::ValueType source, riscv::ValueType result) {
  if (!source || !result || source.getElementType() != result.getElementType() ||
      source.getAxisIds().size() >= result.getAxisIds().size())
    return false;
  auto from = source.getLayout(), to = result.getLayout();
  if (from.getCarrier() != "rvv" || to.getCarrier() != "rvv" ||
      from.getSew() != to.getSew() || from.getLmulEighths() != to.getLmulEighths() ||
      from.getVl() != to.getVl() || from.getValidity() != to.getValidity() ||
      from.getRegisterGroups() != to.getRegisterGroups())
    return false;
  size_t next = 0;
  for (size_t position = 0; position < result.getAxisIds().size(); ++position) {
    const bool shared = next < source.getAxisIds().size() &&
        source.getAxisIds()[next] == result.getAxisIds()[position];
    for (auto [left, right] : {
             std::pair(source.getShape(), result.getShape()),
             std::pair(from.getTimeFactors(), to.getTimeFactors()),
             std::pair(from.getLaneFactors(), to.getLaneFactors()),
             std::pair(from.getReplicaFactors(), to.getReplicaFactors()),
             std::pair(from.getFragmentFactors(), to.getFragmentFactors()),
             std::pair(from.getLocalFactors(), to.getLocalFactors())})
      if (right[position] != (shared ? left[next] : 1))
        return false;
    next += shared;
  }
  return next == source.getAxisIds().size();
}

bool isSingletonScalar(mlir::Type type, mlir::Type element) {
  if (type == element)
    return true;
  auto value = mlir::dyn_cast<riscv::ValueType>(type);
  if (!value || value.getElementType() != element ||
      value.getLayout().getCarrier() != "scalar" ||
      value.getLayout().getValidity() != "full")
    return false;
  for (auto factors : {value.getShape(), value.getLayout().getTimeFactors(),
                       value.getLayout().getLaneFactors(),
                       value.getLayout().getReplicaFactors(),
                       value.getLayout().getFragmentFactors(),
                       value.getLayout().getLocalFactors()})
    if (llvm::any_of(factors.asArrayRef(), [](int64_t factor) { return factor != 1; }))
      return false;
  return true;
}

void selectUnsignedNarrowShifts(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::NarrowOp> candidates;
  module.walk([&](riscv::NarrowOp op) { candidates.push_back(op); });
  unsigned selected = 0, clusteredSplats = 0;
  for (size_t index = 0; index < candidates.size(); ++index) {
    auto narrow = candidates[index];
    auto kernel = narrow->getParentOfType<riscv::KernelOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(narrow.getResult().getType());
    auto element = result
        ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
        : mlir::IntegerType();
    auto shift = narrow.getInput().getDefiningOp<riscv::BinaryOp>();
    if (!kernel || kernel.getResourcesMaterialized() || narrow.getSaturate() ||
        narrow.getRounding() != "rtz" || !element || !element.isUnsigned() ||
        !shift || shift.getKind() != "shr" ||
        shift->getBlock() != narrow->getBlock() || !shift.getResult().hasOneUse())
      continue;
    auto input = mlir::dyn_cast<riscv::ValueType>(shift.getResult().getType());
    if (!riscv::supportsRVVWidenBinary(kernel.getTarget(), result, result, input))
      continue;
    const bool splatInput = isSingletonScalar(shift.getLhs().getType(),
                                             input.getElementType());
    if (!splatInput && shift.getLhs().getType() != input)
      continue;
    auto amountType = mlir::dyn_cast<riscv::ValueType>(shift.getRhs().getType());
    const bool broadcastAmount = addsOnlySingletonAxes(amountType, input);
    auto lanes = riscv_internal::staticProduct(input.getLayout().getLaneFactors().asArrayRef());
    if (broadcastAmount && (!lanes || *lanes <= 0))
      continue;
    const bool vectorAmount = shift.getRhs().getType() == input || broadcastAmount;
    if (!vectorAmount && shift.getRhs().getType() != input.getElementType())
      continue;
    if (shift.getLeaf().getInstruction() !=
        (splatInput ? "rvv.vsrl.vv.splat-lhs" :
         vectorAmount ? "rvv.vsrl.vv" : "rvv.vsrl.vx"))
      continue;
    if (splatInput && (!vectorAmount || riscv::fieldReadProjection(shift.getLhs())))
      continue;
    // Retain the original shift's read point. The explicit RHS projection
    // preserves all low log2(input SEW) bits consumed by vnsrl.
    rewriter.setInsertionPoint(shift);
    mlir::Value source = shift.getLhs();
    if (splatInput) {
      mlir::OpBuilder::InsertionGuard insertion(rewriter);
      auto *producer = source.getDefiningOp();
      if (producer && producer->getBlock() == shift->getBlock() &&
          input.getLayout().getRegisterGroups() == result.getLayout().getRegisterGroups()) {
        unsigned remaining = 32;
        bool ordered = true, crossedPeer = false;
        for (auto *cursor = producer->getNextNode(); cursor != shift;
             cursor = cursor->getNextNode()) {
          if (!cursor || !remaining-- || cursor->getNumRegions() ||
              !mlir::isMemoryEffectFree(cursor)) {
            ordered = false;
            break;
          }
          if (auto peer = mlir::dyn_cast<riscv::RVVNarrowShiftRightOp>(cursor))
            crossedPeer |= peer.getResult().getType() == result;
        }
        if (ordered && crossedPeer) {
          rewriter.setInsertionPointAfter(producer);
          ++clusteredSplats;
        }
      }
      if (auto scalar = mlir::dyn_cast<riscv::ValueType>(source.getType())) {
        auto zero = mlir::arith::ConstantIndexOp::create(rewriter, shift.getLoc(), 0);
        llvm::SmallVector<mlir::Value> indices(scalar.getShape().size(), zero);
        llvm::SmallVector<mlir::Attribute> selectors(scalar.getShape().size(),
                                                    rewriter.getStringAttr("index"));
        source = riscv::ExtractOp::create(rewriter,
            shift.getLoc(), input.getElementType(), source, indices,
            rewriter.getArrayAttr(selectors),
            riscv::AccessAttr::get(rewriter.getContext(), "unit", "dense", 1,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1),
            riscv_internal::leaf(rewriter, "transfer", "extract",
                                 "rvv.extract.unit", "rvv.extract.unit", 0, 0));
      }
      source = riscv::RVVSplatOp::create(rewriter,
          shift.getLoc(), input, source,
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat", "rvv.splat",
                               0, input.getLayout().getRegisterGroups()));
    }
    mlir::Value amount = shift.getRhs();
    if (broadcastAmount) {
      amount = riscv::RVVAxisBroadcastOp::create(rewriter,
          shift.getLoc(), input, amount,
          riscv_internal::leaf(
              rewriter, "rvv", "axis-broadcast", "rvv.axis-broadcast",
              "rvv.axis-broadcast", amountType.getLayout().getRegisterGroups(),
              input.getLayout().getRegisterGroups(), 0, 0, "none", "exact", {*lanes, *lanes}));
    }
    if (vectorAmount) {
      auto widen = amount.getDefiningOp<riscv::WidenOp>();
      if (widen && widen.getInput().getType() == result) {
        amount = widen.getInput();
      } else {
        auto projection = mlir::cast<riscv::NarrowOp>(rewriter.clone(*narrow));
        projection.getInputMutable().assign(amount);
        amount = projection.getResult();
        // Each successful selection removes one existing shift; no shift
        // producers are cloned, so this worklist is bounded by the input DAG.
        if (projection.getInput().getDefiningOp<riscv::BinaryOp>())
          candidates.push_back(projection);
      }
    }
    const llvm::StringRef instruction =
        vectorAmount ? "rvv.vnsrl.wv" : "rvv.vnsrl.wx";
    auto replacement = riscv::RVVNarrowShiftRightOp::create(rewriter,
        shift.getLoc(), result, source, amount,
        riscv_internal::leaf(
            rewriter, "rvv", "narrow-shift-right", instruction, instruction,
            input.getLayout().getRegisterGroups() +
                (vectorAmount ? result.getLayout().getRegisterGroups() : 0),
            result.getLayout().getRegisterGroups(), 0, 0, "none",
            result.getLayout().getValidity() == "tail" ? "agnostic" : "exact"));
    riscv_internal::copyOrigin(narrow, replacement);
    rewriter.replaceOp(narrow, replacement.getResult());
    rewriter.eraseOp(shift);
    ++selected;
  }
  if (selected)
    llvm::errs() << "weft-narrow-shift: selected=" << selected
                 << " clustered-splats=" << clusteredSplats << '\n';
}

void selectReducedScalarScales(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::RVVPartialFinalizeOp> finalizers;
  module.walk([&](riscv::RVVPartialFinalizeOp op) { finalizers.push_back(op); });
  unsigned selected = 0;
  for (auto finalize : finalizers) {
    auto combine =
        finalize.getInput().getDefiningOp<riscv::RVVPartialScaleCombineOp>();
    auto reduced = combine
        ? combine.getInput().getDefiningOp<riscv::RVVPartialReduceOp>()
        : riscv::RVVPartialReduceOp();
    auto result = mlir::dyn_cast<mlir::IntegerType>(finalize.getResult().getType());
    auto kernel = finalize->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() || !combine || !reduced ||
        combine->getBlock() != finalize->getBlock() ||
        !combine.getResult().hasOneUse() ||
        combine.getInput().getType().getSlots() != 1 ||
        combine.getResult().getType().getSlots() != 1 ||
        combine.getScales().size() != 1 || combine.getScaleReplicas()[0] != 0 ||
        combine.getSlotOrder()[0] != 0 || !result || !result.isSigned() ||
        result.getWidth() != 32 ||
        finalize.getLeaf().getInstruction() != "rvv.partial-finalize.extract" ||
        combine.getLeaf().getInstruction() != "rvv.partial-scale-combine")
      continue;
    auto source = reduced.getInput().getType().getPartialType();
    auto sourceElement = mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
    auto lanes = riscv::rvvLaneCount(source);
    auto scale = combine.getScales()[0];
    auto scaleType = mlir::dyn_cast<riscv::ValueType>(scale.getType());
    bool singleton = scale.getType() == result;
    if (scaleType && scaleType.getElementType() == result &&
        scaleType.getLayout().getCarrier() == "scalar" &&
        scaleType.getLayout().getValidity() == "full") {
      singleton = true;
      for (auto factors : {scaleType.getShape(),
                           scaleType.getLayout().getTimeFactors(),
                           scaleType.getLayout().getLaneFactors(),
                           scaleType.getLayout().getReplicaFactors(),
                           scaleType.getLayout().getFragmentFactors(),
                           scaleType.getLayout().getLocalFactors()})
        singleton &= llvm::all_of(factors.asArrayRef(),
                                  [](int64_t factor) { return factor == 1; });
    }
    auto scaleRange = riscv_internal::integerRange(scale);
    if (!sourceElement || !sourceElement.isSigned() ||
        sourceElement.getWidth() != 16 ||
        !lanes || *lanes <= 0 || source.getShape().size() != 1 ||
        source.getShape()[0] != *lanes || source.getLayout().getValidity() != "full" ||
        source.getLayout().getTimeFactors()[0] != 1 ||
        source.getLayout().getReplicaFactors()[0] != 1 ||
        source.getLayout().getFragmentFactors()[0] != 1 ||
        source.getLayout().getLocalFactors()[0] != 1 || !singleton || !scaleRange)
      continue;
    auto products = analysis::productSumRange(
        {std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()},
        *scaleRange, *lanes);
    if (!products || !products->fits(result))
      continue;
    // One reduced lane and one coefficient have no vector-chain reuse. The
    // checked i32 product can use the scalar ALU after the required extraction.
    rewriter.setInsertionPoint(combine);
    if (scaleType) {
      auto zero = mlir::arith::ConstantIndexOp::create(rewriter, finalize.getLoc(), 0);
      llvm::SmallVector<mlir::Value> indices(scaleType.getShape().size(), zero);
      llvm::SmallVector<mlir::Attribute> selectors(
          scaleType.getShape().size(), rewriter.getStringAttr("index"));
      scale = riscv::ExtractOp::create(rewriter,
          finalize.getLoc(), result, scale, indices, rewriter.getArrayAttr(selectors),
          riscv::AccessAttr::get(rewriter.getContext(), "unit", "dense", 1, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, "none", 1),
          riscv_internal::leaf(rewriter, "transfer", "extract",
                               "rvv.extract.unit", "rvv.extract.unit", 0, 0));
    }
    finalize.getInputMutable().assign(combine.getInput());
    // Consume the coefficient at the original combine point, including any
    // deferred field projection. Only the pure partial extraction moves here.
    rewriter.moveOpBefore(finalize, combine);
    rewriter.setInsertionPoint(combine);
    auto product = riscv::BinaryOp::create(rewriter,
        finalize.getLoc(), result, finalize.getResult(), scale, "mul",
        riscv_internal::leaf(rewriter, "scalar", "pointwise", "scalar.mul",
                             "scalar.mul", 0, 0));
    riscv_internal::copyOrigin(combine, product);
    finalize.getResult().replaceAllUsesExcept(product.getResult(), product);
    rewriter.eraseOp(combine);
    ++selected;
  }
  if (selected)
    llvm::errs() << "weft-scalar-reduced-scale: selected=" << selected << '\n';
}

mlir::LogicalResult materializeFieldReads(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::FieldOp> fields;
  module.walk([&](riscv::FieldOp field) { fields.push_back(field); });
  for (auto field : fields) {
    if (riscv::isScalarHalfBitRead(field))
      field.setLeafAttr(riscv_internal::leaf(rewriter, "transfer", "encoded-field",
          "scalar.encoded.f16-bits", "scalar.encoded.f16-bits", 0, 0));
    auto readsStorage = [&](mlir::Operation *user) {
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
      if (!effects) return false;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      return llvm::any_of(instances, [&](const auto &effect) {
        return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) &&
               effect.getValue() == field.getResult();
      });
    };
    auto result = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
    auto encoding = mlir::dyn_cast<kernel::EncodingType>(
        riscv_internal::logicalElement(field.getOwner().getType()));
    auto access = field.getAccess();
    const bool scalarBytes = result &&
        riscv::fieldReadInstruction(field.getResult(), result, access) == "encoded.field-read.scalar-bytes";
    if (!result || !encoding || (!scalarBytes && (result.getLayout().getCarrier() != "rvv" ||
        access.getMapping() != "natural" ||
        !((access.getAlignment() < 2 && riscv_internal::logicalBitWidth(result.getElementType()) == 16) ||
          riscv_internal::fieldFacts(field).scalarPerRecord) ||
        riscv_internal::interleaveRows(field, encoding) <= 0)) ||
        llvm::all_of(field.getResult().getUsers(), readsStorage))
      continue;
    auto readType = result;
    if (!scalarBytes && riscv_internal::fieldFacts(field).scalarPerRecord &&
        !riscv::isContiguousInterleavedScalarRead(field.getResult(), result, access)) {
      auto geometry = riscv::RegisterCoordinates::get(result.getLayout());
      if (!geometry) return field.emitError("record scalar read has no physical coordinates");
      llvm::SmallVector<int64_t> one(result.getShape().size(), 1);
      auto layout = riscv::LayoutAttr::get(rewriter.getContext(), "scalar", result.getAxisIds(),
          rewriter.getDenseI64ArrayAttr(one), rewriter.getDenseI64ArrayAttr(one),
          rewriter.getDenseI64ArrayAttr(geometry->extents()), rewriter.getDenseI64ArrayAttr(one),
          rewriter.getDenseI64ArrayAttr(one), result.getLayout().getSew(), 0, 1, 0,
          result.getLayout().getValidity());
      readType = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(result, layout));
    }
    auto temporaries = riscv::fieldReadTemporaryGroups(field.getResult(), readType, access, {});
    if (!temporaries)
      return field.emitError("interleaved byte assembly has no closed read resources");
    rewriter.setInsertionPointAfter(field);
    auto instruction = riscv::fieldReadInstruction(field.getResult(), readType, access);
    auto read = riscv::FieldReadOp::create(rewriter, field.getLoc(), readType,
        field.getResult(), mlir::ValueRange{}, access,
        riscv_internal::leaf(rewriter, "transfer", "field-read", instruction,
                            instruction, 0, 0, *temporaries, 0, "none", "exact"));
    mlir::Value numeric = read.getResult();
    if (readType != result) {
      auto conversion = riscv::ConvertLayoutOp::create(rewriter, field.getLoc(), result, numeric,
          riscv_internal::layoutConversion(rewriter, readType.getLayout(), result.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(field, conversion);
      numeric = conversion.getResult();
    }
    // Explicit reads consume the storage projection, not another read's value.
    field.getResult().replaceUsesWithIf(numeric, [&](mlir::OpOperand &use) {
      return !readsStorage(use.getOwner());
    });
    riscv_internal::copyOrigin(field, read);
  }
  llvm::SmallVector<riscv::ConvertLayoutOp> candidates;
  module.walk([&](riscv::ConvertLayoutOp op) { candidates.push_back(op); });
  for (auto conversion : candidates) {
    auto kernel = conversion->getParentOfType<riscv::KernelOp>();
    auto kind = conversion.getConversion().getKind();
    auto input = conversion.getInput();
    if (!kernel || kernel.getResourcesMaterialized() ||
        !conversion.getSourceAccess())
      continue;
    // Only storage projections are reads. A computed numeric SSA value must
    // retain its explicit representation conversion, never reload its source.
    llvm::SmallVector<mlir::Value> indices;
    if (!riscv::fieldReadProjection(input, &indices))
      continue;
    bool resupply = kind == "local_load" || kind == "time_to_lane" ||
        (kind == "register_to_lane" && riscv_internal::contiguousInterleavedReadAccess(
            input, conversion.getResult().getType()).has_value());
    // A storage read cannot silently widen the lanes of its address operands.
    // Preserve the selected read carrier and explicitly repack its numeric
    // result when the consumer requires a different gather coordinate map.
    if (resupply)
      for (mlir::Value index : indices) {
        auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
        if (!indexType) continue;
        auto required = riscv_internal::projectLayout(rewriter, indexType,
            conversion.getResult().getType().getLayout(), kernel.getTarget());
        if (!required || required != indexType.getLayout()) resupply = false;
      }
    auto readType = resupply ? conversion.getResult().getType() : input.getType();
    if (readType.getLayout().getCarrier() != "rvv" &&
        readType.getLayout().getCarrier() != "scalar")
      continue;
    auto temporaries = riscv::fieldReadTemporaryGroups(
        input, readType, *conversion.getSourceAccess(), indices);
    if (!temporaries)
      return conversion.emitError("encoded read has no closed local resource contract");
    rewriter.setInsertionPoint(conversion);
    auto instruction = riscv::fieldReadInstruction(input, readType, *conversion.getSourceAccess());
    auto read = riscv::FieldReadOp::create(rewriter,
        conversion.getLoc(), readType, input, indices,
        *conversion.getSourceAccess(),
        riscv_internal::leaf(
            rewriter, "transfer", "field-read", instruction,
            instruction, 0, 0,
            *temporaries, 0, "none", "exact"));
    riscv_internal::copyOrigin(conversion, read);
    if (resupply)
      rewriter.replaceOp(conversion, read.getResult());
    else {
      conversion.getInputMutable().assign(read.getResult());
      conversion.removeSourceAccessAttr();
      riscv_internal::selectLaneRepack(rewriter, conversion);
    }
  }
  llvm::SmallVector<riscv::ExtractOp> extracts;
  module.walk([&](riscv::ExtractOp extract) { extracts.push_back(extract); });
  for (auto extract : extracts) {
    auto kernel = extract->getParentOfType<riscv::KernelOp>();
    llvm::SmallVector<mlir::Value> indices;
    auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    auto instruction = riscv::fieldReadInstruction(extract, result, extract.getAccess());
    auto numericConsumer = [&](mlir::Operation *user) {
      if (mlir::isa<riscv::ExtractOp, riscv::FieldReadOp>(user)) return false;
      if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        if (llvm::any_of(instances, [&](const auto &effect) {
              return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) &&
                     effect.getValue() == extract.getResult();
            })) return false;
      }
      return true;
    };
    const bool sharedRegionRead = result && result.getLayout().getCarrier() == "rvv" &&
        riscv::fieldReadProjection(extract) &&
        llvm::any_of(extract.getResult().getUsers(), [&](mlir::Operation *user) {
          return user->getParentRegion() != extract->getParentRegion() &&
                 numericConsumer(user);
        });
    if (kernel && !kernel.getResourcesMaterialized() &&
        (instruction != "encoded.field-read" || sharedRegionRead) &&
        llvm::any_of(extract.getResult().getUsers(), numericConsumer)) {
      riscv::fieldReadProjection(extract, &indices);
      auto temporary = riscv::fieldReadTemporaryGroups(extract, result, extract.getAccess(), indices);
      if (!temporary) return extract.emitError("selected field read has no closed resources");
      rewriter.setInsertionPointAfter(extract);
      auto read = riscv::FieldReadOp::create(rewriter, extract.getLoc(), result, extract,
          indices, extract.getAccess(), riscv_internal::leaf(rewriter, "transfer", "field-read",
              instruction, instruction, 0, 0, *temporary, 0, "none", "exact"));
      riscv_internal::copyOrigin(extract, read);
      extract.getResult().replaceUsesWithIf(read.getResult(), [&](mlir::OpOperand &use) {
        return use.getOwner() != read && numericConsumer(use.getOwner());
      });
      continue;
    }
    if (!kernel || kernel.getResourcesMaterialized() ||
        extract.getAccess().getForm() != "register" ||
        !riscv::fieldReadProjection(extract.getInput(), &indices))
      continue;
    auto input = extract.getInput();
    auto access = input.getDefiningOp()->getAttrOfType<riscv::AccessAttr>("access");
    auto temporaries = riscv::fieldReadTemporaryGroups(input, input.getType(), access, indices);
    if (!access || !temporaries)
      return extract.emitError("register extract source has no closed encoded read");
    rewriter.setInsertionPoint(extract);
    instruction = riscv::fieldReadInstruction(input, input.getType(), access);
    auto read = riscv::FieldReadOp::create(rewriter,
        extract.getLoc(), input.getType(), input, indices, access,
        riscv_internal::leaf(rewriter, "transfer", "field-read",
                             instruction, instruction,
                             0, 0, *temporaries, 0, "none", "exact"));
    riscv_internal::copyOrigin(input.getDefiningOp(), read);
    extract.getInputMutable().assign(read.getResult());
  }
  llvm::SmallVector<riscv::FieldReadOp> reads;
  module.walk([&](riscv::FieldReadOp read) { reads.push_back(read); });
  for (auto read : reads) {
    auto input = read.getInput().getType(), result = read.getResult().getType();
    auto field = riscv::fieldReadProjection(read.getInput());
    auto kernel = read->getParentOfType<riscv::KernelOp>();
    auto encoding = field ? mlir::dyn_cast<kernel::EncodingType>(
        riscv_internal::logicalElement(field.getOwner().getType())) : kernel::EncodingType();
    if (!kernel || kernel.getResourcesMaterialized() || !encoding ||
        input.getLayout().getCarrier() != "scalar" || result.getLayout().getCarrier() != "rvv" ||
        riscv_internal::interleaveRows(field, encoding) <= 1 ||
        !riscv_internal::fieldFacts(field).scalarPerRecord ||
        riscv::isContiguousInterleavedScalarRead(read.getInput(), result, read.getAccess()))
      continue;
    auto temporary = riscv::fieldReadTemporaryGroups(read.getInput(), input, read.getAccess(), read.getIndices());
    if (!temporary) return read.emitError("record scalar supply has no closed resources");
    auto instruction = riscv::fieldReadInstruction(read.getInput(), input, read.getAccess());
    read.getResult().setType(input);
    read.setLeafAttr(riscv_internal::leaf(rewriter, "transfer", "field-read", instruction,
                                        instruction, 0, 0, *temporary, 0, "none", "exact"));
    rewriter.setInsertionPointAfter(read);
    auto conversion = riscv::ConvertLayoutOp::create(rewriter, read.getLoc(), result, read.getResult(),
        riscv_internal::layoutConversion(rewriter, input.getLayout(), result.getLayout()),
        riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(read, conversion);
    read.getResult().replaceAllUsesExcept(conversion.getResult(), conversion);
  }
  return mlir::success();
}

void selectDirectCompareMerges(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::SelectOp> candidates;
  module.walk([&](riscv::SelectOp select) { candidates.push_back(select); });
  for (auto select : candidates) {
    auto compare = select.getCondition().getDefiningOp<riscv::CompareOp>();
    auto type = mlir::dyn_cast<riscv::ValueType>(select.getResult().getType());
    if (!compare || !compare.getResult().hasOneUse() || compare->getBlock() != select->getBlock() ||
        !type || type.getLayout().getCarrier() != "rvv" || type.getLayout().getReplicaGroup() != 1 ||
        select.getLeaf().getInstruction() != "rvv.select") continue;
    auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
    bool numeric = type.getElementType().isF32() || type.getElementType().isF64() ||
        (integer && !integer.isSignless() && (integer.getWidth() == 8 || integer.getWidth() == 16 ||
                                             integer.getWidth() == 32 || integer.getWidth() == 64));
    if (!numeric) continue;
    auto coordinates = riscv::RegisterCoordinates::get(type.getLayout());
    if (!coordinates || coordinates->streams() != 1 || coordinates->lanes() != type.getLayout().getVl()) continue;
    llvm::SmallVector<mlir::Value> inputs{compare.getLhs(), compare.getRhs(),
                                        select.getTrueValue(), select.getFalseValue()};
    // The comparison and merge share one lane domain. Scalar operands select
    // the ISA's vf/vx forms directly; the false branch always supplies the
    // vector merge destination. Other broadcasts remain explicit operations.
    if (llvm::any_of(inputs, [&](mlir::Value value) {
          return value.getType() != type && value.getType() != type.getElementType();
        })) continue;
    if (inputs[0].getType() != type && inputs[1].getType() != type) continue;
    std::string predicate = compare.getPredicate().str();
    if (inputs[0].getType() != type) {
      std::swap(inputs[0], inputs[1]);
      if (predicate == "lt") predicate = "gt";
      else if (predicate == "le") predicate = "ge";
      else if (predicate == "gt") predicate = "lt";
      else if (predicate == "ge") predicate = "le";
    }
    bool rhsVector = inputs[1].getType() == type, trueVector = inputs[2].getType() == type;
    int64_t groups = type.getLayout().getRegisterGroups();
    int64_t inputGroups = (2 + int(rhsVector) + int(trueVector)) * groups;
    auto target = select->getParentOfType<riscv::KernelOp>().getTarget();
    if (inputGroups + groups + 1 > target.getVectorRegisters()) continue;
    rewriter.setInsertionPoint(select);
    if (inputs[3].getType() != type) {
      auto splat = riscv::RVVSplatOp::create(rewriter, select.getLoc(), type, inputs[3],
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat", "rvv.splat",
                               0, groups));
      riscv_internal::copyOrigin(select, splat);
      inputs[3] = splat;
    }
    bool floating = mlir::isa<mlir::FloatType>(type.getElementType());
    std::string instruction = std::string("rvv.compare-select.") +
        (rhsVector ? "vv" : floating ? "vf" : "vx") + "." +
        (trueVector ? "vvm" : floating ? "vfm" : "vxm");
    auto fused = riscv::RVVCompareSelectOp::create(rewriter, select.getLoc(), type,
        inputs[0], inputs[1], inputs[2], inputs[3], predicate,
        riscv_internal::leaf(rewriter, "rvv", "compare-select", instruction, instruction, inputGroups, groups, 1));
    riscv_internal::copyOrigin(select, fused);
    rewriter.replaceOp(select, fused.getResult());
    rewriter.eraseOp(compare);
  }
}

class FinalizeRISCVLeavesPass
    : public mlir::PassWrapper<FinalizeRISCVLeavesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeRISCVLeavesPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-finalize-leaves";
  }
  llvm::StringRef getDescription() const override {
    return "Finalize exact terminal leaf instructions from solved physical types";
  }

  void runOnOperation() override {
    foldPhysicalIndexRemainders(getOperation());
    foldIntegerIdentities(getOperation());
    foldIntegerPowerOfTwoArithmetic(getOperation());
    selectUnsignedMultiplyHigh(getOperation());
    if (mlir::failed(materializeFieldReads(getOperation()))) {
      signalPassFailure();
      return;
    }
    riscv_internal::materializeWholeLayeredReads(getOperation());
    mlir::Builder builder(&getContext());
    // Broadcast the narrow bits before exact integer extension. This keeps the
    // larger representation out of the shuffle and preserves every coordinate.
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<riscv::RVVAxisBroadcastOp> broadcasts;
    getOperation().walk([&](riscv::RVVAxisBroadcastOp broadcast) { broadcasts.push_back(broadcast); });
    for (auto broadcast : broadcasts) {
      auto widen = broadcast.getInput().getDefiningOp<riscv::WidenOp>();
      if (!widen || !widen.getResult().hasOneUse() || widen->getBlock() != broadcast->getBlock()) continue;
      auto source = mlir::dyn_cast<riscv::ValueType>(widen.getInput().getType());
      auto result = broadcast.getResult().getType();
      auto element = source ? mlir::dyn_cast<mlir::IntegerType>(source.getElementType()) : mlir::IntegerType();
      auto wideElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
      auto layout = result.getLayout();
      auto replicas = riscv_internal::staticProduct(layout.getReplicaFactors().asArrayRef());
      if (!element || !wideElement || element.getSignedness() != wideElement.getSignedness() ||
          element.getWidth() != 8 || wideElement.getWidth() != 16 || layout.getLmulEighths() % 2 ||
          !replicas || *replicas <= 0) continue;
      auto narrowLayout = riscv::LayoutAttr::get(builder.getContext(), "rvv", result.getAxisIds(),
          layout.getTimeFactors(), layout.getLaneFactors(), layout.getReplicaFactors(),
          layout.getFragmentFactors(), layout.getLocalFactors(), 8, layout.getLmulEighths() / 2,
          layout.getVl(), *replicas * std::max<int64_t>(1, (layout.getLmulEighths() / 2 + 7) / 8),
          layout.getValidity());
      auto narrowType = riscv::ValueType::get(builder.getContext(), element, result.getShape(), result.getAxisIds(), narrowLayout);
      if (!riscv::packedAxisBroadcastFactor(source, narrowType) ||
          !riscv::supportsRVVLayout(broadcast->getParentOfType<riscv::KernelOp>().getTarget(), narrowLayout)) continue;
      rewriter.setInsertionPoint(broadcast);
      auto narrow = riscv::RVVAxisBroadcastOp::create(rewriter, broadcast.getLoc(), narrowType, widen.getInput(),
          riscv_internal::leaf(builder, "rvv", "axis-broadcast", "rvv.axis-broadcast.packed",
              "rvv.axis-broadcast.packed", source.getLayout().getRegisterGroups(), narrowLayout.getRegisterGroups(),
              2 * narrowLayout.getRegisterGroups(), 0, "none", "exact", broadcast.getLeaf().getParameters().asArrayRef()));
      auto extended = riscv::WidenOp::create(rewriter, broadcast.getLoc(), result, narrow.getResult(),
          riscv_internal::unselectedLeaf(builder));
      auto instruction = riscv_internal::terminalInstruction(extended);
      extended.setLeafAttr(riscv_internal::leaf(builder, "rvv", "widen", instruction, instruction,
          narrowLayout.getRegisterGroups(), layout.getRegisterGroups()));
      riscv_internal::copyOrigin(broadcast, narrow);
      riscv_internal::copyOrigin(widen, extended);
      rewriter.replaceOp(broadcast, extended.getResult());
      rewriter.eraseOp(widen);
    }
    getOperation().walk([&](riscv::RVVAxisBroadcastOp broadcast) {
      if (riscv::repeatedTileBroadcastFactor(broadcast.getInput().getType(), broadcast.getResult().getType())) {
        auto before = broadcast.getLeaf();
        broadcast.setLeafAttr(riscv_internal::leaf(builder, "rvv", "axis-broadcast",
            "rvv.axis-broadcast.repeat-tile", "rvv.axis-broadcast.repeat-tile",
            before.getOperandGroups(), before.getResultGroups(), 0,
            0, "none", "exact", before.getParameters().asArrayRef()));
        return;
      }
      if (!riscv::packedAxisBroadcastFactor(broadcast.getInput().getType(), broadcast.getResult().getType()))
        return;
      auto before = broadcast.getLeaf();
      broadcast.setLeafAttr(riscv_internal::leaf(builder, "rvv", "axis-broadcast",
          "rvv.axis-broadcast.packed", "rvv.axis-broadcast.packed",
          before.getOperandGroups(), before.getResultGroups(),
          2 * ((broadcast.getResult().getType().getLayout().getLmulEighths() + 7) / 8),
          0, "none", "exact", before.getParameters().asArrayRef()));
    });
    bool failed = false;
    getOperation().walk([&](mlir::Operation *operation) {
      // Layout canonicalization, scheduling, and partial materialization can
      // all create conversions after composite lowering.  Close conversions
      // here, after the last physical rewrite, so every conversion in the
      // final program receives its leaf from the same typed ConversionAttr.
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
        llvm::StringRef kind = conversion.getConversion().getKind();
        llvm::StringRef instruction =
            kind == "splat"          ? "rvv.splat"
            : kind == "extract"      ? "rvv.extract"
            : kind == "local_load"   ? "rvv.local-load"
            : kind == "local_store"  ? "rvv.local-store"
            : kind == "tuple"        ? "rvv.tuple-convert"
            : kind == "register_to_lane" ? "rvv.register-to-lane"
            : kind == "time_to_lane" ? "rvv.time-to-lane"
            : kind == "lane_to_register" ? "rvv.lane-to-register"
            : kind == "lane_repack" ? "rvv.lane-repack"
            : kind == "reshape"      ? "rvv.layout-reshape"
                                      : llvm::StringRef();
        if (instruction.empty()) {
          conversion.emitError()
              << "no terminal leaf implements typed layout conversion kind '"
              << kind << "'";
          failed = true;
          return;
        }
        riscv::LeafAttr oldLeaf = conversion.getLeaf();
        conversion.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "layout-conversion", instruction, instruction,
            oldLeaf.getOperandGroups(), oldLeaf.getResultGroups(),
            conversion.getConversion().getTemporaryGroups(), 0));
        conversion->removeAttr("implementation");
        return;
      }
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      if (!implementation)
        return;
      std::string instruction;
      int64_t temporaries = 0;
      llvm::StringRef tail = "agnostic";
      llvm::StringRef engine = implementation.getEngine();
      if (!operation->getResultTypes().empty() &&
          !mlir::isa<riscv::ReduceOp>(operation)) {
        llvm::StringRef carrier =
            riscv_internal::layoutOf(operation->getResult(0).getType())
                ? riscv_internal::layoutOf(operation->getResult(0).getType())
                      .getCarrier()
                : llvm::StringRef("scalar");
        if ((implementation.getEngine() == "rvv" && carrier != "rvv") ||
            (implementation.getEngine() == "scalar" && carrier != "scalar")) {
          operation->emitError()
              << "selected " << operation->getName() << " engine "
              << implementation.getEngine()
              << " disagrees with propagated result carrier " << carrier;
          failed = true;
          return;
        }
      }
      if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(iota);
      } else if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(unary);
        if (riscv::isVectorMathInstruction(instruction)) {
          auto count = riscv::unaryTemporaryGroups(unary,
              mlir::cast<riscv::ValueType>(unary.getResult().getType()), instruction);
          if (!count) {
            unary.emitError("vector math has no closed temporary resource contract");
            failed = true;
            return;
          }
          temporaries = *count;
        }
      } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(binary);
        if (llvm::StringRef(instruction).ends_with(".broadcast"))
          temporaries = riscv::binaryBroadcastTemporaryGroups(binary).value();
        if (instruction == "rvv.mul.row-scale")
          temporaries = riscv::rvvRowScaleTemporaryGroups(binary).value();
      } else if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(compare);
        if (llvm::StringRef(instruction).starts_with("rvv.cmp.bool."))
          temporaries = *riscv_internal::booleanComparisonTemporaryGroups(compare);
      } else if (auto select = mlir::dyn_cast<riscv::SelectOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(select);
        auto groups = riscv::selectTemporaryGroups(select);
        if (groups) temporaries = *groups;
      } else if (mlir::isa<riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp,
                           riscv::WidenOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(operation);
        if (instruction == "rvv.f8e4m3fn-to-f32" || instruction == "rvv.f8e5m2-to-f32" ||
            instruction == "rvv.bf16-to-f32" || instruction == "rvv.f32-to-bf16-rne")
          temporaries = *riscv_internal::storageFloatCastTemporaryGroups(operation);
      } else if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(reduce);
        if (llvm::StringRef(instruction).starts_with("rvv.lane-reduce.")) {
          // vl=0 leaves vd unchanged. The seed is therefore also the explicit
          // passthrough destination, including for completely inactive parts.
          tail = "exact";
          auto input = mlir::cast<riscv::ValueType>(reduce.getInput().getType());
          auto layout = input.getLayout();
          auto streams = riscv_internal::staticProduct(layout.getTimeFactors().asArrayRef());
          temporaries = 1;
          if (streams && *streams > 1 && layout.getValidity() == "full" && reduce.getActiveLengths().empty())
            temporaries += (layout.getLmulEighths() + 7) / 8;
        }
      } else if (mlir::isa<riscv::Fold2Op>(operation)) {
        instruction = riscv_internal::terminalInstruction(operation);
      } else if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation)) {
        auto layout = riscv_internal::layoutOf(update.getResult().getType());
        if (layout && layout.getCarrier() == "local") {
          instruction = "local.update";
          engine = "scalar";
        } else {
          instruction = implementation.getOperation().str();
        }
      } else if (mlir::isa<riscv::LocalLoadOp,
                           riscv::LocalStoreOp>(operation)) {
        instruction = implementation.getOperation().str();
        engine = "transfer";
      } else {
        operation->emitError(
            "structural implementation was not lowered to a terminal physical operation");
        failed = true;
        return;
      }
      if (instruction.empty()) {
        operation->emitError()
            << "no exact terminal instruction implements the solved physical representation for "
            << operation->getName() << " with implementation " << implementation
            << ", operands " << operation->getOperandTypes() << ", results "
            << operation->getResultTypes();
        failed = true;
        return;
      }
      operation->setAttr(
          "leaf", riscv_internal::leaf(
                      builder, engine, implementation.getFamily(), instruction,
                      instruction, 0, 0, temporaries, 0, "none", tail,
                      implementation.getParameters().asArrayRef()));
      operation->removeAttr("implementation");
    });
    if (failed) {
      signalPassFailure();
      return;
    }
    selectUnsignedNarrowShifts(getOperation());
    selectReducedScalarScales(getOperation());
    if (mlir::failed(riscv_internal::selectPredicateRegisters(getOperation()))) {
      signalPassFailure();
      return;
    }
    selectDirectCompareMerges(getOperation());
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createFinalizeRISCVLeavesPass() {
  return std::make_unique<FinalizeRISCVLeavesPass>();
}
