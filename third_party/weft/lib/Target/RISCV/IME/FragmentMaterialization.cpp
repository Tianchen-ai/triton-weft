#include "FragmentMaterialization.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "../RISCVPhysicalSupport.h"
#include "../LocalMemory.h"
#include "Weft/Target/RISCVFragment.h"
#include "Weft/Target/RISCVPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include <limits>

using namespace weft;
using riscv_internal::accessOf;

namespace {
riscv::LayoutAttr fragmentLayout(mlir::Builder &builder, riscv::ValueType value,
                                 mlir::DenseI64ArrayAttr axisIds,
                                 llvm::ArrayRef<int64_t> physicalShape,
                                 llvm::ArrayRef<int64_t> atomShape) {
  auto axes = axisIds.asArrayRef();
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  llvm::SmallVector<int64_t> fragment(atomShape.begin(), atomShape.end());
  llvm::SmallVector<int64_t> replicas;
  for (auto [extent, atom] : llvm::zip(physicalShape, atomShape))
    replicas.push_back(extent / atom);
  return riscv::LayoutAttr::get(
      builder.getContext(), "ime", axisIds,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, one),
      std::max<int64_t>(8, riscv_internal::logicalBitWidth(value)), 0, 1, 0,
      value.getLayout().getValidity());
}

} // namespace

mlir::FailureOr<mlir::Value> weft::riscv_internal::materializeFragmentProduct(
    mlir::IRRewriter &rewriter, mlir::Operation *operation,
    riscv::ImplementationAttr implementation, riscv::ValueType resultType) {

  auto parameters = implementation.getParameters().asArrayRef();
  riscv::FragmentCapabilityAttr capability;
  for (mlir::Attribute candidate :
       operation->getParentOfType<riscv::KernelOp>()
           .getTarget()
           .getFragments()) {
    auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(candidate);
    if (fragment.getInstruction() == implementation.getOperation()) {
      capability = fragment;
      break;
    }
  }
  if (parameters.size() != 5 || !capability) {
    operation->emitError("IME selection has no fragment shape/resources");
    return mlir::failure();
  }
  const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
  if (!contract) {
    operation->emitError("selected fragment has no registered local contract");
    return mlir::failure();
  }
  auto lhsType =
      mlir::cast<riscv::ValueType>(operation->getOperand(0).getType());
  auto rhsType =
      mlir::cast<riscv::ValueType>(operation->getOperand(1).getType());
  llvm::SmallVector<int64_t, 2> lhsShape = {
      capability.getActiveMFactor() * parameters[3], capability.getKFactor()};
  llvm::SmallVector<int64_t, 2> rhsShape = {
      capability.getKFactor(), capability.getNFactor() * parameters[4]};
  llvm::SmallVector<int64_t, 2> resultShape = {
      lhsShape[0], rhsShape[1]};
  if (!supportsRISCVFragmentTile(*contract, resultShape[0], resultShape[1], lhsShape[1])) {
    operation->emitError("selected fragment repetitions exceed the bounded product domain");
    return mlir::failure();
  }
  auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
  if (!over || over.size() != 1 || resultType.getAxisIds().size() != 2) {
    operation->emitError("IME product has no explicit matrix axis relation");
    return mlir::failure();
  }
  auto lhsAxes = rewriter.getDenseI64ArrayAttr({resultType.getAxisIds()[0], over[0]});
  auto rhsAxes = rewriter.getDenseI64ArrayAttr({over[0], resultType.getAxisIds()[1]});
  auto matchesAxes = [](riscv::ValueType value, mlir::DenseI64ArrayAttr axes) {
    return value.getAxisIds().size() == axes.size() &&
           llvm::all_of(value.getAxisIds().asArrayRef(), [&](int64_t axis) {
             return llvm::is_contained(axes.asArrayRef(), axis);
           });
  };
  if (!matchesAxes(lhsType, lhsAxes) || !matchesAxes(rhsType, rhsAxes)) {
    operation->emitError("selected fragment operands must carry the exact matrix axes");
    return mlir::failure();
  }
  auto makeFragment = [&](riscv::ValueType value, llvm::StringRef role,
                          riscv::FragmentPackingAttr packing,
                          mlir::DenseI64ArrayAttr axes,
                          llvm::ArrayRef<int64_t> shape) {
    const int64_t resources =
        role == "lhs" ? capability.getLhsResourceGroups()
        : role == "rhs" ? capability.getRhsResourceGroups()
                        : capability.getAccumulatorResourceGroups();
    llvm::SmallVector<int64_t> atom = role == "lhs"
        ? llvm::SmallVector<int64_t>{capability.getActiveMFactor(), capability.getKFactor()}
        : role == "rhs" ? llvm::SmallVector<int64_t>{capability.getKFactor(), capability.getNFactor()}
                        : llvm::SmallVector<int64_t>{capability.getActiveMFactor(), capability.getNFactor()};
    return riscv::FragmentType::get(
        value.getContext(), implementation.getOperation(),
        role, packing, value.getElementType(),
        riscv_internal::integers(rewriter, shape), axes,
        fragmentLayout(rewriter, value, axes, shape, atom),
        resources * (shape[0] / atom[0]) * (shape[1] / atom[1]));
  };
  riscv::FragmentType lhsFragment =
      makeFragment(lhsType, "lhs", capability.getLhsPacking(), lhsAxes, lhsShape);
  riscv::FragmentType rhsFragment =
      makeFragment(rhsType, "rhs", capability.getRhsPacking(), rhsAxes, rhsShape);
  riscv::FragmentType resultFragment = makeFragment(
      resultType, "accumulator", capability.getAccumulatorPacking(), resultType.getAxisIds(),
      resultShape);
  const int64_t reductionExtent = physicalExtent(operation->getOperand(0), over[0]);
  if (reductionExtent != capability.getKFactor()) {
    const int64_t atomK = capability.getKFactor();
    if (reductionExtent <= 0 || reductionExtent % atomK ||
        physicalExtent(operation->getOperand(1), over[0]) != reductionExtent ||
        !resultType.getElementType().isSignedInteger(32)) {
      operation->emitError("repeated IME reduction requires equal complete K atoms and a signed i32 accumulator");
      return mlir::failure();
    }
    auto zero = riscv::IMEZeroOp::create(rewriter, operation->getLoc(), resultFragment,
        leaf(rewriter, "ime", "fragment-init", "ime.zero", "ime.zero", 0, 0));
    auto constant = [&](int64_t value) {
      return mlir::arith::ConstantIndexOp::create(rewriter, operation->getLoc(), value).getResult();
    };
    auto loop = mlir::scf::ForOp::create(rewriter, operation->getLoc(), constant(0),
        constant(reductionExtent / atomK), constant(1), mlir::ValueRange{zero.getResult()});
    loop->setAttr("weft.riscv.system_unroll", rewriter.getStringAttr("disable"));
    loop->setAttr("weft.riscv.direction", rewriter.getStringAttr("ascending"));
    copyOrigin(operation, loop);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto issueOffset = mlir::arith::MulIOp::create(rewriter, operation->getLoc(),
        loop.getInductionVar(), constant(atomK));
    llvm::DenseMap<mlir::Value, mlir::Value> sliced;
    auto project = [&](auto &&self, mlir::Value input) -> mlir::FailureOr<mlir::Value> {
      auto type = mlir::dyn_cast<riscv::ValueType>(input.getType());
      if (!type || !llvm::is_contained(type.getAxisIds().asArrayRef(), over[0])) return input;
      if (auto found = sliced.find(input); found != sliced.end()) return found->second;
      const size_t position = llvm::find(type.getAxisIds().asArrayRef(), over[0]) - type.getAxisIds().asArrayRef().begin();
      if (type.getShape()[position] == 1) return input;
      llvm::SmallVector<int64_t> shape(type.getShape().asArrayRef()); shape[position] = atomK;
      llvm::SmallVector<int64_t> one(shape.size(), 1), lanes(one), replicas(shape);
      lanes[position] = atomK; replicas[position] = 1;
      const int64_t sew = std::max<int64_t>(8, logicalBitWidth(type.getElementType()));
      const int64_t lmul = std::max<int64_t>(1, atomK * sew * 8 / contract->vlenBits);
      auto count = staticProduct(replicas);
      if (!count) return mlir::failure();
      auto layout = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", type.getAxisIds(),
          integers(rewriter, one), integers(rewriter, lanes), integers(rewriter, replicas),
          integers(rewriter, one), integers(rewriter, one), sew, lmul, atomK,
          *count * ((lmul + 7) / 8), type.getLayout().getValidity());
      auto target = riscv::ValueType::get(rewriter.getContext(), type.getElementType(),
          integers(rewriter, shape), type.getAxisIds(), layout);
      auto producer = input.getDefiningOp();
      mlir::Value result;
      if (auto conversion = input.getDefiningOp<riscv::ConvertLayoutOp>()) {
        auto projected = self(self, conversion.getInput());
        if (mlir::failed(projected)) return mlir::failure();
        result = *projected;
      } else if (producer && mlir::isa<riscv::CastOp, riscv::WidenOp, riscv::NarrowOp,
                                riscv::UnaryOp, riscv::BinaryOp>(producer)) {
        llvm::SmallVector<mlir::Value> operands;
        for (auto operand : producer->getOperands()) {
          auto projected = self(self, operand);
          if (mlir::failed(projected)) return mlir::failure();
          operands.push_back(*projected);
        }
        mlir::OperationState state(producer->getLoc(), producer->getName());
        state.addOperands(operands); state.addTypes(target); state.addAttributes(producer->getAttrs());
        state.attributes.set("leaf", unselectedLeaf(rewriter));
        result = rewriter.create(state)->getResult(0);
      } else {
        auto origin = originPoint(input, over[0]);
        if (!origin) {
          auto field = sourceField(input);
          auto load = field ? sourceLoad(field.getOwner()) : riscv::LoadOp();
          if (load) origin = originPoint(load.getRegion(), over[0]);
        }
        mlir::Value selector;
        if (origin) {
          auto base = mlir::arith::AddIOp::create(rewriter, operation->getLoc(),
              origin.getBase(), issueOffset);
          auto remaining = mlir::arith::SubIOp::create(rewriter, operation->getLoc(),
              origin.getActive(), issueOffset);
          auto active = mlir::arith::MaxSIOp::create(rewriter, operation->getLoc(),
              remaining, constant(0));
          mlir::Value bounded = mlir::arith::MinUIOp::create(rewriter, operation->getLoc(),
              active, constant(atomK));
          auto full = origin.getActive().getDefiningOp<mlir::arith::ConstantIndexOp>();
          if (full && full.value() >= reductionExtent) bounded = constant(atomK);
          auto point = riscv::PhysicalPointOp::create(rewriter, operation->getLoc(),
              origin.getType(), origin.getParent(), base, bounded, constant(atomK));
          copyOrigin(operation, point);
          selector = point;
        } else {
          mlir::Value memory;
          auto load = input.getDefiningOp<riscv::LoadOp>();
          if (load && !load.getSnapshotStorage() &&
              !riscv::storageWrittenBetween(load.getRegion(), load, operation)) {
            memory = load.getRegion();
          } else {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(loop);
            mlir::Value local = input;
            if (type.getLayout().getCarrier() != "local") {
              int64_t identity = lastLocalIdentity(operation->getParentOfType<mlir::ModuleOp>());
              auto allocated = allocateLocalValue(rewriter, type, operation, identity);
              if (mlir::failed(allocated) ||
                  mlir::failed(initializeLocalValue(rewriter, *allocated, input, operation)))
                return mlir::failure();
              local = *allocated;
            }
            auto view = localView(rewriter, local, operation);
            if (mlir::failed(view)) return mlir::failure();
            memory = *view;
          }
          llvm::SmallVector<mlir::Value> coordinates(shape.size(), constant(0));
          coordinates[position] = issueOffset;
          auto window = valueMemoryWindow(rewriter, memory, type, coordinates, target, operation);
          if (mlir::failed(window)) return mlir::failure();
          result = riscv::LoadOp::create(rewriter, operation->getLoc(), target, *window,
              mlir::Value(), unassignedAccess(rewriter), unselectedLeaf(rewriter));
          sliced.try_emplace(input, result);
          return result;
        }
        llvm::SmallVector<mlir::Attribute> selectors(shape.size(), rewriter.getStringAttr("all"));
        selectors[position] = rewriter.getStringAttr("domain");
        auto extract = riscv::ExtractOp::create(rewriter, operation->getLoc(), target, input,
            mlir::ValueRange{selector}, rewriter.getArrayAttr(selectors),
            accessOf(input) ? accessOf(input) : unassignedAccess(rewriter), unselectedLeaf(rewriter));
        copyOrigin(operation, extract); result = extract.getResult();
      }
      sliced.try_emplace(input, result); return result;
    };
    auto lhs = project(project, operation->getOperand(0)), rhs = project(project, operation->getOperand(1));
    if (mlir::failed(lhs) || mlir::failed(rhs)) {
      operation->emitError("IME K issue cannot preserve its operand coordinates"); return mlir::failure();
    }
    auto *atom = rewriter.clone(*operation);
    atom->setOperand(0, *lhs); atom->setOperand(1, *rhs);
    auto value = materializeFragmentProduct(rewriter, atom, implementation, resultType);
    if (mlir::failed(value)) return mlir::failure();
    auto unpack = (*value).getDefiningOp<riscv::IMEUnpackOp>();
    auto mma = unpack.getInput().getDefiningOp<riscv::IMEFragmentMMAOp>();
    auto redundantZero = mma.getAccumulator().getDefiningOp<riscv::IMEZeroOp>();
    mma.getAccumulatorMutable().assign(loop.getRegionIterArg(0));
    rewriter.setInsertionPointToEnd(loop.getBody());
    mlir::scf::YieldOp::create(rewriter, operation->getLoc(), mma.getResult());
    auto conversion = unpack.getConversion(); auto unpackLeaf = unpack.getLeaf();
    rewriter.eraseOp(unpack); rewriter.eraseOp(redundantZero); rewriter.eraseOp(atom);
    rewriter.setInsertionPointAfter(loop);
    auto final = riscv::IMEUnpackOp::create(rewriter, operation->getLoc(), resultType,
        loop.getResult(0), conversion, unpackLeaf);
    copyOrigin(operation, final);
    return final.getResult();
  }
  auto packLeaf = [&](llvm::StringRef role) {
    return riscv_internal::leaf(
        rewriter, "ime", "fragment-pack", ("ime.pack." + role).str(),
        ("ime.pack." + role).str(), 0, 0, contract->packTemporaryGroups,
        role == "lhs" ? lhsFragment.getResourceGroups() : rhsFragment.getResourceGroups(),
        "none", "exact", parameters.take_front(3),
        role == "lhs" ? contract->lhsPackLocalBytes : contract->rhsPackLocalBytes);
  };
  auto packOperand = [&](mlir::Value input, riscv::FragmentType fragment) -> mlir::Value {
    auto type = mlir::cast<riscv::ValueType>(input.getType());
    const int64_t innerAxis = fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[1]];
    const int64_t inner = fragment.getPacking().getColumnsPerTile();
    llvm::SmallVector<int64_t> one(type.getShape().size(), 1), lanes(one), replicas(one);
    for (auto [dimension, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      const auto position = llvm::find(fragment.getAxisIds().asArrayRef(), axis) -
                            fragment.getAxisIds().asArrayRef().begin();
      if (axis == innerAxis)
        lanes[dimension] = inner;
      else
        replicas[dimension] = fragment.getShape()[position];
    }
    const int64_t lmul = std::max<int64_t>(1, inner * 8 * 8 / contract->vlenBits);
    auto layout = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", type.getAxisIds(),
        riscv_internal::integers(rewriter, one), riscv_internal::integers(rewriter, lanes),
        riscv_internal::integers(rewriter, replicas), riscv_internal::integers(rewriter, one),
        riscv_internal::integers(rewriter, one), 8, lmul, inner,
        (lmul + 7) / 8 * fragment.getShape()[fragment.getPacking().getAxisOrder()[0]],
        type.getLayout().getValidity());
    if (type.getLayout() == layout)
      return input;
    return riscv::ConvertLayoutOp::create(rewriter, operation->getLoc(),
        riscv::ValueType::get(type.getContext(), type.getElementType(), type.getShape(), type.getAxisIds(), layout),
        input, riscv_internal::layoutConversion(rewriter, type.getLayout(), layout),
        riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter)).getResult();
  };
  auto operandAccess = [&](mlir::Value value) -> riscv::AccessAttr {
    if (auto access = accessOf(value); access && access.getForm() != "unassigned")
      return access;
    auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!type || type.getLayout().getCarrier() != "rvv")
      return {};
    return riscv::AccessAttr::get(rewriter.getContext(), "register", "dense", 1,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1);
  };
  riscv::AccessAttr lhsAccess = operandAccess(operation->getOperand(0));
  riscv::AccessAttr rhsAccess = operandAccess(operation->getOperand(1));
  if (!lhsAccess || !rhsAccess) {
    operation->emitError("IME operands have no typed memory access contract");
    return mlir::failure();
  }
  auto contiguousOperand = [&](mlir::Value input, riscv::FragmentType fragment) -> mlir::Value {
    if (!input.hasOneUse())
      return {};
    auto pointwise = input.getDefiningOp<riscv::BinaryOp>();
    mlir::Value projection = input;
    std::optional<int64_t> temporaryGroups;
    if (pointwise && (pointwise.getKind() == "add" || pointwise.getKind() == "sub")) {
      temporaryGroups = riscv::fragmentPointwiseTemporaryGroups(fragment, pointwise.getRhs().getType());
      if (!temporaryGroups)
        return {};
      projection = pointwise.getLhs();
      if (!projection.hasOneUse())
        return {};
    } else {
      pointwise = {};
    }
    if (auto widen = projection.getDefiningOp<riscv::WidenOp>())
      projection = widen.getInput();
    else if (auto cast = projection.getDefiningOp<riscv::CastOp>())
      projection = cast.getInput();
    if (!projection.hasOneUse())
      return {};
    auto field = riscv::fieldReadProjection(projection);
    auto load = field ? sourceLoad(field.getOwner()) : riscv::LoadOp();
    auto access = riscv::fragmentFieldReadAccess(projection, fragment);
    if (!access || !load || needsReadSnapshot(load))
      return {};
    auto integer = mlir::cast<mlir::IntegerType>(
        mlir::cast<riscv::ValueType>(projection.getType()).getElementType());
    const int64_t bits = integer.getWidth();
    const int64_t innerAxis = fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[1]];
    if (bits == 8 && !pointwise) {
      auto chunkAccess = riscv::fragmentChunkReadAccess(field.getResult(), fragment);
      auto point = originPoint(projection, innerAxis);
      if (chunkAccess && point && field->getNumOperands() == 1) {
        riscv::IMEFragmentChunkSplitLoadOp chunks;
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfter(load);
          auto fullField = mlir::cast<riscv::FieldOp>(rewriter.clone(*field));
          const auto instruction = riscv::fragmentChunkInstruction(fragment);
          const int64_t loadGroups = riscv::fragmentChunkLoadGroups(fragment,
              operation->getParentOfType<riscv::KernelOp>().getTarget());
          chunks = riscv::IMEFragmentChunkSplitLoadOp::create(rewriter, load.getLoc(),
              fragment, fragment, fullField.getResult(), *chunkAccess,
              riscv_internal::leaf(rewriter, "ime", "fragment-chunk-split-load",
                  instruction, instruction,
                  0, 0, riscv::fragmentChunkTemporaryGroups(fragment), 2 * fragment.getResourceGroups(),
                  "none", "agnostic", {loadGroups}));
          copyOrigin(load, chunks);
        }
        auto blockSize = mlir::arith::ConstantIndexOp::create(rewriter, operation->getLoc(), 32);
        auto zero = mlir::arith::ConstantIndexOp::create(rewriter, operation->getLoc(), 0);
        auto offset = mlir::arith::RemUIOp::create(rewriter, operation->getLoc(), point.getBase(), blockSize);
        auto first = mlir::arith::CmpIOp::create(rewriter, operation->getLoc(),
            mlir::arith::CmpIPredicate::eq, offset, zero);
        auto selected = mlir::scf::IfOp::create(rewriter, operation->getLoc(),
            mlir::TypeRange{fragment}, first, true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(selected.thenBlock());
          mlir::scf::YieldOp::create(rewriter, operation->getLoc(), chunks.getLow());
          rewriter.setInsertionPointToStart(selected.elseBlock());
          mlir::scf::YieldOp::create(rewriter, operation->getLoc(), chunks.getHigh());
        }
        return selected.getResult(0);
      }
    }
    auto point = bits < 8 ? originPoint(projection, innerAxis) : riscv::PhysicalPointOp();
    if (bits < 8 && !point)
      return {};
    const bool windows = access->getForm() == "strided";
    const bool grouped = !windows && riscv::supportsFragmentGroupedLoad(fragment,
        operation->getParentOfType<riscv::KernelOp>().getTarget());
    const char *loadSpelling = windows ? "ime.field-raw-load.row-windows" :
                              grouped ? "ime.field-raw-load.grouped" : "ime.field-raw-load";
    auto read = riscv::IMEFragmentStorageLoadOp::create(rewriter, operation->getLoc(), fragment,
        projection, *access, riscv_internal::leaf(rewriter, "ime", "fragment-storage-load", loadSpelling,
            loadSpelling, 0, 0, grouped ? 0 : 1, fragment.getResourceGroups(), "none", "agnostic"));
    copyOrigin(projection.getDefiningOp(), read);
    mlir::Value supplied = read.getResult();
    if (bits < 8) {
      auto constant = [&](int64_t value) -> mlir::Value {
        return mlir::arith::ConstantIndexOp::create(rewriter, operation->getLoc(), value);
      };
      auto record = constant(load.getRegion().getType().getElements());
      auto group = constant(access->getGroupSize());
      auto layer = constant(access->getLayerSize());
      auto width = constant(bits);
      auto withinRecord = mlir::arith::RemUIOp::create(rewriter, operation->getLoc(), point.getBase(), record);
      auto withinGroup = mlir::arith::RemUIOp::create(rewriter, operation->getLoc(), withinRecord, group);
      mlir::Value layerIndex = mlir::arith::DivUIOp::create(rewriter, operation->getLoc(), withinGroup, layer);
      if (access->getOrder() == "hi_first")
        layerIndex = mlir::arith::SubIOp::create(rewriter, operation->getLoc(), constant(8 / bits - 1), layerIndex);
      auto shift = mlir::arith::MulIOp::create(rewriter, operation->getLoc(), layerIndex, width);
      const char *decodeSpelling = grouped ? "ime.decode.grouped-bits" : "ime.decode.bits";
      auto decoded = riscv::IMEFragmentDecodeOp::create(rewriter, operation->getLoc(), fragment,
          supplied, shift.getResult(), bits, integer.isSigned(),
          riscv_internal::leaf(rewriter, "ime", "fragment-decode", decodeSpelling, decodeSpelling,
                              0, 0, grouped ? 0 : 1, fragment.getResourceGroups()));
      copyOrigin(projection.getDefiningOp(), decoded);
      supplied = decoded.getResult();
    }
    if (!pointwise)
      return supplied;
    auto instruction = ("ime.fragment." + pointwise.getKind()).str();
    auto computed = riscv::IMEFragmentPointwiseOp::create(rewriter, pointwise.getLoc(), fragment,
        supplied, pointwise.getRhs(), pointwise.getKind(), "copy",
        riscv_internal::leaf(rewriter, "ime", "fragment-pointwise", instruction, instruction,
                            0, 0, *temporaryGroups, fragment.getResourceGroups()));
    copyOrigin(pointwise, computed);
    return computed.getResult();
  };
  auto materializeOperand = [&](mlir::Value input, riscv::FragmentType fragment,
                                llvm::StringRef role, riscv::AccessAttr access) -> mlir::Value {
    if (auto contiguous = contiguousOperand(input, fragment))
      return contiguous;
    return riscv::IMEPackOp::create(rewriter, operation->getLoc(), fragment,
        packOperand(input, fragment), role, fragment.getPacking(), access, packLeaf(role)).getResult();
  };
  auto lhsPack = materializeOperand(operation->getOperand(0), lhsFragment, "lhs", lhsAccess);
  auto rhsPack = materializeOperand(operation->getOperand(1), rhsFragment, "rhs", rhsAccess);
  auto mmaLeaf = riscv_internal::leaf(
      rewriter, "ime", "fragment-mma", implementation.getOperation(),
      implementation.getOperation(), 0, 0, 0,
      riscv::fragmentMMAAdditionalGroups(capability, riscv::fragmentPartCount(resultFragment)), "none", "exact",
      parameters.take_front(3), contract->mmaLocalBytes);
  auto initial = riscv::IMEZeroOp::create(rewriter, operation->getLoc(), resultFragment,
      riscv_internal::leaf(rewriter, "ime", "fragment-init", "ime.zero", "ime.zero", 0, 0));
  auto mma = riscv::IMEFragmentMMAOp::create(rewriter,
      operation->getLoc(), resultFragment, lhsPack,
      rhsPack, initial.getResult(), "copy", capability.getMmaGroups(),
      capability.getMmaChunks(), capability.getClobbers(),
      capability.getVolatileAsm(), capability.getMemoryClobber(), mmaLeaf);
  const auto unpackInstruction = riscv::fragmentUnpackInstruction(resultFragment, resultType);
  auto unpackLeaf = riscv_internal::leaf(
      rewriter, "ime", "fragment-unpack", unpackInstruction,
      unpackInstruction, 0, 0,
      riscv::fragmentUnpackTemporaryGroups(resultFragment, resultType), 0);
  auto unpack = riscv::IMEUnpackOp::create(rewriter,
      operation->getLoc(), resultType, mma.getResult(),
      riscv_internal::conversion(rewriter, "fragment_to_rvv", "handoff",
                                 riscv::fragmentUnpackTemporaryGroups(resultFragment, resultType)),
      unpackLeaf);
  riscv_internal::copyOrigin(operation, mma);
  riscv_internal::copyOrigin(operation, unpack);
  if (auto canonical = operation->getAttr("canonical_op")) {
    mma->setAttr("canonical_op", canonical);
    unpack->setAttr("canonical_op", canonical);
  }
  return unpack.getResult();
}

namespace {
using namespace weft::riscv_internal;
void retainFragmentAccumulators(
    mlir::IRRewriter &rewriter, mlir::ModuleOp module) {
  llvm::SmallVector<riscv::IMEUnpackOp> operations;
  module.walk([&](riscv::IMEUnpackOp operation) { operations.push_back(operation); });
  for (auto unpack : operations) {
    auto mma = unpack.getInput().getDefiningOp<riscv::IMEFragmentMMAOp>();
    if (!mma || !unpack.getResult().hasOneUse() || !mma.getResult().hasOneUse() ||
        !mma.getAccumulator().getDefiningOp<riscv::IMEZeroOp>())
      continue;
    auto add = mlir::dyn_cast<riscv::BinaryOp>(*unpack.getResult().getUsers().begin());
    if (!add || add.getKind() != "add" || !add.getResult().hasOneUse() ||
        add->getBlock() != unpack->getBlock())
      continue;
    auto carried = mlir::dyn_cast<mlir::BlockArgument>(
        add.getLhs() == unpack.getResult() ? add.getRhs() : add.getLhs());
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(add->getParentOp());
    if (!carried || !loop || carried.getOwner() != loop.getBody() ||
        carried.getArgNumber() == 0)
      continue;
    auto update = mlir::dyn_cast<riscv::UpdateOp>(*add.getResult().getUsers().begin());
    mlir::Value next = add.getResult();
    if (update) {
      if (!update.coversWholeState() || update.getInput() != carried ||
          update.getValue() != next || !update.getResult().hasOneUse() ||
          update->getBlock() != add->getBlock() ||
          llvm::any_of(carried.getUses(), [&](mlir::OpOperand &use) {
            return use.getOwner() != add && use.getOwner() != update;
          })) continue;
      next = update.getResult();
    } else if (!carried.hasOneUse()) continue;
    const unsigned index = carried.getArgNumber() - 1;
    auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    auto initial = loop.getInitArgs()[index].getDefiningOp<riscv::NewOp>();
    if (!initial || !initial.getInitialized() || initial.getPlacement() != "register" ||
        !initial.getResult().hasOneUse() ||
        constantInt(initial.getInitial()) != 0 ||
        yield.getOperand(index) != next ||
        carried.getType() != unpack.getResult().getType() ||
        add.getResult().getType() != carried.getType())
      continue;
    auto lower = constantInt(loop.getLowerBound());
    auto upper = constantInt(loop.getUpperBound());
    auto step = constantInt(loop.getStep());
    const auto *contract = findRISCVFragmentCapability(mma.getResult().getType().getFamily());
    if (!lower || !upper || !step || *lower < 0 || *upper <= *lower || *step <= 0 ||
        !contract || contract->lhsElementBits != 8 || contract->rhsElementBits != 8 ||
        contract->accumulatorElementBits != 32)
      continue;
    const int64_t trips = (*upper - *lower - 1) / *step + 1;
    const int64_t productBound = int64_t(128) * 128 * contract->kFactor;
    if (trips > std::numeric_limits<int32_t>::max() / productBound)
      continue;
    auto oldType = unpack.getResult().getType();
    auto fragment = mma.getResult().getType();
    rewriter.setInsertionPoint(initial);
    auto zero = riscv::IMEZeroOp::create(rewriter, loop.getLoc(), fragment,
        riscv_internal::leaf(rewriter, "ime", "fragment-init", "ime.zero", "ime.zero", 0, 0));
    initial.getInitialMutable().assign(zero.getResult());
    initial.getResult().setType(fragment);
    carried.setType(fragment);
    mma->setOperand(2, carried);
    yield.setOperand(index, mma.getResult());
    loop.getResult(index).setType(fragment);
    rewriter.setInsertionPointAfter(loop);
    auto final = riscv::IMEUnpackOp::create(rewriter, loop.getLoc(), oldType, loop.getResult(index),
        unpack.getConversion(), unpack.getLeaf());
    loop.getResult(index).replaceAllUsesExcept(final.getResult(), final);
    copyOrigin(unpack, final);
    if (update) rewriter.eraseOp(update);
    rewriter.eraseOp(add);
    rewriter.eraseOp(unpack);
  }
}

class RetainFragmentAccumulatorsPass final
    : public mlir::PassWrapper<RetainFragmentAccumulatorsPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RetainFragmentAccumulatorsPass)
  llvm::StringRef getArgument() const override { return "weft-riscv-retain-fragment-accumulators"; }
  llvm::StringRef getDescription() const override { return "Retain compatible integer fragment state across bounded physical issue loops"; }
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    retainFragmentAccumulators(rewriter, getOperation());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> weft::createRetainRISCVFragmentAccumulatorsPass() {
  return std::make_unique<RetainFragmentAccumulatorsPass>();
}
