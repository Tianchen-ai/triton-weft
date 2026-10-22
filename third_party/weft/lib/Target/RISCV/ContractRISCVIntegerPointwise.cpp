#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
std::optional<int64_t> sharedConstant(Value value) {
  for (unsigned depth = 0; depth < 8; ++depth) {
    if (auto shared = value.getDefiningOp<riscv::RegisterMaterializeOp>();
        shared && shared.getRealization() == "physical-share") {
      value = shared.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      auto source = dyn_cast<IntegerType>(cast.getInput().getType());
      auto result = dyn_cast<IntegerType>(cast.getResult().getType());
      if (!source || !result || source.getWidth() != result.getWidth()) return std::nullopt;
      value = cast.getInput();
      continue;
    }
    if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
      if (isa<IntegerAttr>(constant.getValue())) return constantInt(value);
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
      if (isa<IntegerAttr>(constant.getValue())) return constantInt(value);
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<unsigned> unsignedBits(Value value, unsigned depth = 0) {
  if (depth >= 8) return std::nullopt;
  auto type = dyn_cast<IntegerType>(logicalElement(value.getType()));
  if (!type) return std::nullopt;
  if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
    auto bits = unsignedBits(widen.getInput(), depth + 1);
    if (bits && *bits < type.getWidth()) return bits;
  }
  if (type.isUnsigned()) return type.getWidth();
  return std::nullopt;
}

// A low-bit extraction cannot observe an addend divisible by the entire
// extracted bit range. Prove this in the original unsigned modular domain:
// ((x + c) >> s) & (2^k - 1) == (x >> s) & (2^k - 1).
// Keeping the mask is essential, including when the addition wraps.
unsigned simplifyLowBitOffsets(ModuleOp module, IRRewriter &b) {
  SmallVector<riscv::BinaryOp> masks;
  module.walk([&](riscv::BinaryOp op) {
    if (op.getKind() == "and" || op.getKind() == "mod") masks.push_back(op);
  });
  unsigned changed = 0;
  for (auto mask : masks) {
    auto kernel = mask->getParentOfType<riscv::KernelOp>();
    auto type = dyn_cast<riscv::ValueType>(mask.getResult().getType());
    auto element = type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType();
    if (!kernel || kernel.getResourcesMaterialized() || !element || !element.isUnsigned() ||
        type.getLayout().getCarrier() != "rvv" ||
        mask.getLhs().getType() != type || mask.getRhs().getType() != element) continue;
    auto bits = sharedConstant(mask.getRhs());
    if (!bits || *bits <= 0) continue;
    uint64_t period = static_cast<uint64_t>(*bits) + (mask.getKind() == "and");
    if (!llvm::isPowerOf2_64(period)) continue;
    Value input = mask.getLhs();
    auto shift = input.getDefiningOp<riscv::BinaryOp>();
    int64_t amount = 0;
    if (shift && (shift.getKind() == "shr" || shift.getKind() == "div")) {
      auto constant = sharedConstant(shift.getRhs());
      if (!constant || *constant < 0 ||
          shift.getLhs().getType() != type || shift.getRhs().getType() != element ||
          shift->getBlock() != mask->getBlock()) continue;
      if (shift.getKind() == "div") {
        if (!llvm::isPowerOf2_64(*constant)) continue;
        amount = llvm::Log2_64(*constant);
      } else amount = *constant;
      if (amount >= element.getWidth()) continue;
      input = shift.getLhs();
    } else {
      shift = {};
    }
    unsigned coveredBits = llvm::Log2_64(period) + amount;
    if (coveredBits >= 63 || coveredBits > element.getWidth()) continue;
    int64_t divisor = int64_t{1} << coveredBits;
    auto add = input.getDefiningOp<riscv::BinaryOp>();
    if (!add || add.getKind() != "add" || add.getResult().getType() != type ||
        add->getBlock() != mask->getBlock()) continue;
    Value retained;
    for (unsigned side = 0; side < 2; ++side) {
      Value offset = add->getOperand(side), other = add->getOperand(1 - side);
      if (other.getType() == type &&
          (offset.getType() == type || offset.getType() == element) &&
          knownMultipleOf(offset, divisor)) { retained = other; break; }
    }
    if (!retained) continue;
    b.setInsertionPoint(mask);
    if (shift) {
      auto replacement = riscv::BinaryOp::create(b, shift.getLoc(), type, retained,
          shift.getRhs(), shift.getKind(), shift.getLeaf());
      copyOrigin(shift, replacement);
      retained = replacement.getResult();
    }
    b.modifyOpInPlace(mask, [&] { mask.getLhsMutable().assign(retained); });
    if (shift && shift.getResult().use_empty()) b.eraseOp(shift);
    if (add.getResult().use_empty()) b.eraseOp(add);
    ++changed;
  }
  return changed;
}

class ContractRISCVIntegerPointwisePass
    : public PassWrapper<ContractRISCVIntegerPointwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractRISCVIntegerPointwisePass)
  StringRef getArgument() const override { return "weft-riscv-contract-integer-pointwise"; }
  StringRef getDescription() const override {
    return "Contract disjoint integer bit assembly and bias into modular RVV MAC";
  }
  void runOnOperation() override {
    IRRewriter b(&getContext());
    unsigned indexOffsets = simplifyLowBitOffsets(getOperation(), b);
    if (indexOffsets)
      llvm::errs() << "weft-integer-index-offsets: selected=" << indexOffsets << '\n';
    SmallVector<riscv::BinaryOp> roots;
    getOperation().walk([&](riscv::BinaryOp op) {
      if (op.getKind() == "sub" || op.getKind() == "add") roots.push_back(op);
    });
    unsigned selected = 0;
    for (auto root : roots) {
      auto kernel = root->getParentOfType<riscv::KernelOp>();
      auto type = dyn_cast<riscv::ValueType>(root.getResult().getType());
      auto element = type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType();
      if (!kernel || kernel.getResourcesMaterialized() || !element || element.isSignless() ||
          element.getWidth() < 8 || element.getWidth() > 64 ||
          root.getRhs().getType() != element || type.getLayout().getCarrier() != "rvv" ||
          type.getLayout().getSew() != element.getWidth() ||
          (type.getLayout().getValidity() != "full" && type.getLayout().getValidity() != "tail")) continue;
      Value assembled = root.getLhs();
      auto reinterpret = assembled.getDefiningOp<riscv::CastOp>();
      if (reinterpret) {
        auto source = dyn_cast<riscv::ValueType>(reinterpret.getInput().getType());
        auto sourceElement = source ? dyn_cast<IntegerType>(source.getElementType()) : IntegerType();
        if (!sourceElement || sourceElement.getWidth() != element.getWidth() ||
            source.getShape() != type.getShape() || source.getAxisIds() != type.getAxisIds() ||
            source.getLayout() != type.getLayout() || !reinterpret.getResult().hasOneUse() ||
            reinterpret->getBlock() != root->getBlock()) continue;
        assembled = reinterpret.getInput();
      }
      auto joined = assembled.getDefiningOp<riscv::BinaryOp>();
      if (!joined || joined.getKind() != "or" || !joined.getResult().hasOneUse() ||
          joined->getBlock() != root->getBlock() ||
          (!reinterpret && joined.getResult().getType() != type)) continue;
      Type assemblyType = assembled.getType();
      for (unsigned side = 0; side < 2; ++side) {
        auto shift = joined->getOperand(side).getDefiningOp<riscv::BinaryOp>();
        Value low = joined->getOperand(1 - side);
        if (!shift || shift.getKind() != "shl" || !shift.getResult().hasOneUse() ||
            shift->getBlock() != root->getBlock() || shift.getLhs().getType() != assemblyType ||
            shift.getResult().getType() != assemblyType || low.getType() != assemblyType) continue;
        auto amount = sharedConstant(shift.getRhs());
        auto bits = unsignedBits(low);
        if (!amount || *amount <= 0 || *amount >= element.getWidth() || !bits || *bits > *amount) continue;
        bool ordered = true;
        for (Operation *op = shift->getNextNode(); op != root; op = op->getNextNode()) {
          if (op->getNumRegions() || !riscv::hasOnlyReadEffects(op)) { ordered = false; break; }
        }
        if (!ordered) continue;
        b.setInsertionPoint(root);
        auto factor = riscv::ConstantOp::create(b, root.getLoc(), element,
            b.getIntegerAttr(element, llvm::APInt(element.getWidth(), 1).shl(*amount)));
        Value high = shift.getLhs();
        if (reinterpret) {
          low = riscv::CastOp::create(b, root.getLoc(), type, low, unselectedLeaf(b));
          high = riscv::CastOp::create(b, root.getLoc(), type, high, unselectedLeaf(b));
        }
        auto biased = riscv::BinaryOp::create(b, root.getLoc(), type, low, root.getRhs(),
            root.getKind(), root.getLeaf());
        auto mac = riscv::RVVIntegerMACOp::create(b, root.getLoc(), type, high,
            factor.getResult(), biased.getResult(),
            leaf(b, "rvv", "integer-mac", "rvv.vmacc.vx", "rvv.vmacc.vx",
                 2 * type.getLayout().getRegisterGroups(), type.getLayout().getRegisterGroups(),
                 0, 0, "none", "exact"));
        copyOrigin(root, biased);
        copyOrigin(root, mac);
        b.replaceOp(root, mac.getResult());
        if (reinterpret) b.eraseOp(reinterpret);
        b.eraseOp(joined);
        b.eraseOp(shift);
        ++selected;
        break;
      }
    }
    if (selected) llvm::errs() << "weft-integer-bit-assembly: selected=" << selected << '\n';
  }
};
} // namespace

std::unique_ptr<Pass> weft::createContractRISCVIntegerPointwisePass() {
  return std::make_unique<ContractRISCVIntegerPointwisePass>();
}
