#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/MemoryEffects.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Verification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace weft;
using namespace weft::riscv;
using namespace weft::riscv::detail;

namespace {
bool nativeNibbleFieldOrder(FieldOp field, MemDescType memory) {
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding());
  auto module = field->getParentOfType<mlir::ModuleOp>();
  if (!encoding || !module) return false;
  llvm::StringRef family = encoding.getKind() == "base" ? encoding.getFamily() : llvm::StringRef();
  module.walk([&](DerivedEncodingOp derived) {
    if (family.empty() && derived.getResultFamily() == encoding.getFamily() &&
        derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
        derived.getParameterValues() == encoding.getParameters().asArrayRef())
      family = derived.getSourceFamily();
  });
  bool nativeOrder = false;
  module.walk([&](EncodingDeclOp declaration) {
    if (declaration.getSymName() == family)
      nativeOrder = declaration.getByteOrder() == "little" && declaration.getBitOrder() == "lsb_first";
  });
  return nativeOrder;
}

std::optional<int64_t> nonnegativeIndexUpperBound(mlir::Value value, unsigned depth = 0) {
  if (depth > 12)
    return std::nullopt;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return constant.value() >= 0 ? std::optional<int64_t>(constant.value()) : std::nullopt;
  if (auto remainder = value.getDefiningOp<mlir::arith::RemUIOp>()) {
    auto divisor = remainder.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (divisor && divisor.value() > 0)
      return divisor.value() - 1;
  }
  if (auto division = value.getDefiningOp<mlir::arith::DivUIOp>()) {
    auto divisor = division.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto bound = nonnegativeIndexUpperBound(division.getLhs(), depth + 1);
    if (divisor && divisor.value() > 0 && bound)
      return *bound / divisor.value();
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    auto lhs = nonnegativeIndexUpperBound(multiply.getLhs(), depth + 1);
    auto rhs = nonnegativeIndexUpperBound(multiply.getRhs(), depth + 1);
    if (lhs && rhs && (*rhs == 0 || *lhs <= std::numeric_limits<int64_t>::max() / *rhs))
      return *lhs * *rhs;
  }
  if (auto subtract = value.getDefiningOp<mlir::arith::SubIOp>()) {
    auto lhs = subtract.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto bound = nonnegativeIndexUpperBound(subtract.getRhs(), depth + 1);
    if (lhs && bound && lhs.value() >= *bound)
      return lhs.value();
  }
  return std::nullopt;
}

llvm::StringRef signedness(RISCVFragmentSignedness value) {
  return value == RISCVFragmentSignedness::Signed ? "signed" : "unsigned";
}

FragmentPackingAttr packingAttr(mlir::MLIRContext *context,
                                const RISCVFragmentPacking &packing) {
  return FragmentPackingAttr::get(
      context, packing.schema,
      mlir::DenseI64ArrayAttr::get(context, packing.axisOrder),
      packing.rowsPerTile, packing.columnsPerTile, packing.tileOrder,
      packing.elementOrder, packing.storageBits, packing.alignment);
}

FragmentCapabilityAttr fragmentCapability(mlir::Operation *operation,
                                          llvm::StringRef instruction) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel)
    return {};
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto capability = mlir::cast<FragmentCapabilityAttr>(attribute);
    if (capability.getInstruction() == instruction)
      return capability;
  }
  return {};
}

} // namespace

mlir::FailureOr<llvm::SmallVector<llvm::StringRef>>
weft::riscv::usedInstructionExtensions(KernelOp kernel) {
  llvm::SmallVector<llvm::StringRef> extensions;
  auto walk = kernel.walk([&](IMEFragmentMMAOp mma) -> mlir::WalkResult {
    if (mlir::failed(mma.verify()))
      return mlir::WalkResult::interrupt();
    const auto *contract = findRISCVFragmentCapability(mma.getLeaf().getInstruction());
    extensions.push_back(contract->extension);
    return mlir::WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return mlir::failure();
  auto splits = kernel.walk([&](IMEFragmentChunkSplitLoadOp split) -> mlir::WalkResult {
    if (mlir::failed(split.verify())) return mlir::WalkResult::interrupt();
    const auto *contract = findRISCVFragmentCapability(split.getLow().getType().getFamily());
    if (split.getLow().getType().getPacking().getRowsPerTile() > 1)
      extensions.push_back(contract->extension);
    return mlir::WalkResult::advance();
  });
  if (splits.wasInterrupted()) return mlir::failure();
  auto narrowed = kernel.walk([&](IMEFragmentStorageLoadOp load) -> mlir::WalkResult {
    if (load.getLeaf().getInstruction() != "ime.field-nibble-pack") return mlir::WalkResult::advance();
    if (mlir::failed(load.verify())) return mlir::WalkResult::interrupt();
    extensions.push_back(findRISCVFragmentCapability(load.getResult().getType().getFamily())->extension);
    return mlir::WalkResult::advance();
  });
  if (narrowed.wasInterrupted()) return mlir::failure();
  auto unpacks = kernel.walk([&](IMEUnpackOp unpack) -> mlir::WalkResult {
    if (mlir::failed(unpack.verify())) return mlir::WalkResult::interrupt();
    const auto *contract = findRISCVFragmentCapability(unpack.getInput().getType().getFamily());
    if (unpack.getLeaf().getInstruction() == contract->rowMergeInstruction)
      extensions.push_back(contract->extension);
    return mlir::WalkResult::advance();
  });
  if (unpacks.wasInterrupted()) return mlir::failure();
  llvm::sort(extensions);
  extensions.erase(std::unique(extensions.begin(), extensions.end()), extensions.end());
  return extensions;
}

std::optional<AccessAttr> weft::riscv::fragmentFieldReadAccess(
    mlir::Value input, FragmentType result) {
  llvm::SmallVector<mlir::Value> indices;
  auto field = fieldReadProjection(input, &indices);
  auto source = mlir::dyn_cast<ValueType>(input.getType());
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  auto slice = owner ? owner.getRegion().getDefiningOp<SliceOp>() : SliceOp();
  auto view = slice ? slice.getBase().getDefiningOp<MemoryViewOp>() : MemoryViewOp();
  auto capability = field ? fragmentCapability(field, result.getFamily()) : FragmentCapabilityAttr();
  auto fieldType = field ? mlir::dyn_cast<ValueType>(field.getResult().getType()) : ValueType();
  if (!source || !owner || owner.getSnapshotStorage() || !slice || !view || indices.size() > 1 ||
      !capability ||
      !mlir::isa<mlir::BlockArgument>(view.getBase()) || !fieldType ||
      fieldType.getShape().size() != 2 || source.getShape().size() != 2 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      result.getPacking() != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                                       : capability.getRhsPacking()) ||
      fragmentPartResourceGroups(result) != 1)
    return std::nullopt;
  auto access = field.getAccess();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  const auto packing = result.getPacking();
  const int64_t rows = result.getShape()[packing.getAxisOrder()[0]];
  const int64_t columns = packing.getColumnsPerTile();
  const bool packedNibbles = packing.getStorageBits() == 4;
  const int64_t rowBytes = packedNibbles ? columns / 2 : columns;
  if (packedNibbles && !nativeNibbleFieldOrder(field, owner.getRegion().getType()))
    return std::nullopt;
  const int64_t cohort = owner.getRegion().getType().getInterleaveRows();
  const int64_t rowAxis = result.getAxisIds()[packing.getAxisOrder()[0]];
  const int64_t innerAxis = result.getAxisIds()[packing.getAxisOrder()[1]];
  const bool singleNaturalRow = cohort == 0 && rows == 1 &&
      access.getMapping() == "natural";
  const bool rowWindows = cohort > 0 && access.getMapping() == "natural" &&
      access.getFieldChunkBytes() > rowBytes && access.getFieldChunkBytes() % rowBytes == 0 &&
      fieldType.getShape()[1] % access.getFieldChunkBytes() == 0;
  if (!integer || integer.isSignless() || integer.getWidth() > 8 ||
      source.getAxisIds()[0] != rowAxis || source.getAxisIds()[1] != innerAxis ||
      fieldType.getAxisIds() != source.getAxisIds() ||
      fieldType.getShape()[1] != owner.getRegion().getType().getElements() ||
      source.getShape()[1] != columns ||
      (!singleNaturalRow && (cohort < rows || cohort % rows ||
                             (access.getFieldChunkBytes() != rowBytes && !rowWindows))) ||
      access.getBitOffset() % 8 || packing.getElementOrder() != "row_major" ||
      (access.getMapping() != "natural" && access.getMapping() != "grouped_layered"))
    return std::nullopt;
  if (access.getMapping() == "natural" ? integer.getWidth() != (packedNibbles ? 4 : 8)
      : (8 % integer.getWidth() || access.getLayerSize() != columns ||
         access.getGroupSize() != columns * (8 / integer.getWidth())))
    return std::nullopt;
  PhysicalPointOp rowPoint, innerPoint;
  for (mlir::Value index : slice.getIndices())
    if (auto point = index.getDefiningOp<PhysicalPointOp>()) {
      const int64_t axis = point.getResult().getType().getDomain().getAxisId();
      if (axis == rowAxis)
        rowPoint = point;
      if (axis == innerAxis)
        innerPoint = point;
    }
  for (mlir::Value index : indices) {
    auto point = index.getDefiningOp<PhysicalPointOp>();
    if (!point || point.getResult().getType().getDomain().getAxisId() != innerAxis)
      return std::nullopt;
    innerPoint = point;
  }
  auto hasPartition = [](PhysicalPointOp point, int64_t extent) {
    auto constant = point ? point.getPartition().getDefiningOp<mlir::arith::ConstantOp>()
                          : mlir::arith::ConstantOp();
    auto value = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr();
    return value && value.getInt() == extent && knownIndexMultipleOf(point.getBase(), extent);
  };
  if (!hasPartition(rowPoint, rows) || !hasPartition(innerPoint, columns))
    return std::nullopt;
  return AccessAttr::get(input.getContext(), rowWindows ? "strided" : "unit", access.getMapping(), access.getAlignment(),
      0, 0, access.getGroupSize(), access.getLayerSize(), access.getJoinFields(),
      access.getJoinLowBits(), access.getJoinRole(), access.getBitOffset(), access.getStorageBits(),
      access.getOrder(), access.getFieldChunkBytes());
}

static std::optional<AccessAttr> fragmentRowChunkAccess(
    mlir::Value input, FragmentType result, bool nibblePack) {
  auto field = input.getDefiningOp<FieldOp>();
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  auto slice = owner ? owner.getRegion().getDefiningOp<SliceOp>() : SliceOp();
  auto memory = owner ? owner.getRegion().getType() : MemDescType();
  auto type = mlir::dyn_cast<ValueType>(input.getType());
  auto capability = field ? fragmentCapability(field, result.getFamily()) : FragmentCapabilityAttr();
  const auto *contract = findRISCVFragmentCapability(result.getFamily());
  auto element = type ? mlir::dyn_cast<mlir::IntegerType>(type.getElementType()) : mlir::IntegerType();
  const bool nibbles = element && !element.isSignless() && element.getWidth() == 4;
  const bool nativeNibbles = nibbles && result.getPacking().getStorageBits() == 4;
  if (!field || !owner || owner.getSnapshotStorage() || !slice || !type || !capability ||
      !contract || (!nibblePack && !nativeNibbles && contract->chunkSplitInstruction != "spacemit-ime2-vupack128") ||
      contract->vlenBits != 1024 || memory.getElements() != 32 ||
      type.getShape().size() != 2 || type.getAxisIds().size() != 2 ||
      type.getShape()[1] != 32 || field.getAccess().getMapping() != "natural" ||
      field.getAccess().getFieldChunkBytes() != (nibbles ? 16 : 32) || field.getAccess().getBitOffset() % 8 ||
      (!nibbles && (!integerSignednessMatches(type.getElementType(), "signed") || elementBitWidth(type.getElementType()) != 8)) ||
      (nibblePack && (nibbles || result.getPacking().getStorageBits() != 4 ||
                      field->getNumOperands() != 1 || contract->rowMergeInstruction != "spacemit-ime2-vpack-rows")) ||
      (result.getRole() != "lhs" && result.getRole() != "rhs"))
    return std::nullopt;
  if (nibbles) {
    auto parent = field->getParentOfType<KernelOp>();
    if (!parent || !llvm::is_contained(parent.getTarget().getLegalLMULEighths().asArrayRef(), 4))
      return std::nullopt;
    if (!nativeNibbleFieldOrder(field, memory)) return std::nullopt;
  }
  const auto packing = result.getPacking();
  if (packing != (result.getRole() == "lhs" ? capability.getLhsPacking() : capability.getRhsPacking()) ||
      packing.getColumnsPerTile() != (nibblePack || nativeNibbles ? 32 : 16) || fragmentPartResourceGroups(result) != 1 ||
      (packing.getRowsPerTile() != 1 && packing.getRowsPerTile() != 4 && packing.getRowsPerTile() != 8))
    return std::nullopt;
  const int64_t rows = type.getShape()[0], cohort = memory.getInterleaveRows();
  const int64_t rowAxis = result.getAxisIds()[packing.getAxisOrder()[0]];
  const int64_t innerAxis = result.getAxisIds()[packing.getAxisOrder()[1]];
  const bool splitCohorts = packing.getRowsPerTile() == 8 && cohort >= 4 &&
      cohort < rows && rows % cohort == 0 && cohort % 4 == 0;
  if (rows != result.getShape()[packing.getAxisOrder()[0]] || rows <= 0 ||
      (!splitCohorts && (cohort < rows || cohort % rows)) ||
      type.getAxisIds()[0] != rowAxis || type.getAxisIds()[1] != innerAxis)
    return std::nullopt;
  bool row = false, inner = false;
  for (mlir::Value index : slice.getIndices()) {
    auto point = index.getDefiningOp<PhysicalPointOp>();
    if (!point) continue;
    const int64_t axis = point.getResult().getType().getDomain().getAxisId();
    const int64_t size = axis == rowAxis ? rows : axis == innerAxis ? 32 : 0;
    auto partition = point.getPartition().getDefiningOp<mlir::arith::ConstantOp>();
    auto value = partition ? mlir::dyn_cast<mlir::IntegerAttr>(partition.getValue()) : mlir::IntegerAttr();
    if (!size || !value || value.getInt() != size || !knownIndexMultipleOf(point.getBase(), size))
      continue;
    row |= axis == rowAxis;
    inner |= axis == innerAxis;
  }
  return row && inner ? std::optional<AccessAttr>(field.getAccess()) : std::nullopt;
}

std::optional<AccessAttr> weft::riscv::fragmentChunkReadAccess(mlir::Value field, FragmentType result) {
  if (result.getPacking().getStorageBits() != 8) return std::nullopt;
  return fragmentRowChunkAccess(field, result, false);
}

std::optional<AccessAttr> weft::riscv::fragmentNibblePackReadAccess(mlir::Value field, FragmentType result) {
  return fragmentRowChunkAccess(field, result, true);
}

std::optional<AccessAttr> weft::riscv::fragmentNibbleCohortReadAccess(mlir::Value input, FragmentType result) {
  auto field = input.getDefiningOp<FieldOp>();
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  if (!owner || elementBitWidth(mlir::cast<ValueType>(input.getType()).getElementType()) != 4 ||
      result.getPacking().getStorageBits() != 4 || result.getPacking().getRowsPerTile() != 8 ||
      owner.getRegion().getType().getInterleaveRows() != 4)
    return std::nullopt;
  return fragmentRowChunkAccess(input, result, false);
}

bool weft::riscv::fragmentChunkNeedsRowGuard(mlir::Value input, FragmentType fragment) {
  auto field = input.getDefiningOp<FieldOp>();
  auto load = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  return load && load.getRegion().getType().getInterleaveRows() <
      fragment.getShape()[fragment.getPacking().getAxisOrder()[0]];
}

int64_t weft::riscv::fragmentChunkLoadGroups(mlir::Value fieldValue, FragmentType fragment, TargetAttr target) {
  const bool nibbles = elementBitWidth(mlir::cast<ValueType>(fieldValue.getType()).getElementType()) == 4;
  if (fragment.getPacking().getRowsPerTile() != 8) return 1;
  auto field = fieldValue.getDefiningOp<FieldOp>();
  auto load = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  if (load && load.getRegion().getType().getInterleaveRows() <
                  fragment.getShape()[fragment.getPacking().getAxisOrder()[0]])
    return 1;
  const int64_t parts = fragmentPartCount(fragment);
  if (nibbles) {
    for (int64_t groups : {4, 2})
      if (parts % groups == 0 &&
          llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), groups * 8) &&
          llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), groups * 16))
        return groups;
    return 1;
  }
  for (int64_t groups : {4, 2})
    if ((parts * 2) % groups == 0 &&
        llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), groups * 8))
      return groups;
  return 1;
}

int64_t weft::riscv::fragmentChunkTemporaryGroups(mlir::Value field, FragmentType fragment) {
  if (elementBitWidth(mlir::cast<ValueType>(field.getType()).getElementType()) == 4)
    return std::max<int64_t>(6, fragmentPartCount(fragment) * 2);
  if (fragment.getPacking().getRowsPerTile() == 1) return 0;
  return std::max<int64_t>(4, fragmentPartCount(fragment) *
      (fragment.getPacking().getRowsPerTile() == 8 ? 2 : 1));
}

llvm::StringRef weft::riscv::fragmentChunkInstruction(mlir::Value field, FragmentType fragment) {
  const auto *contract = findRISCVFragmentCapability(fragment.getFamily());
  if (!contract) return "";
  auto integer = mlir::dyn_cast<mlir::IntegerType>(mlir::cast<ValueType>(field.getType()).getElementType());
  if (!integer) return "";
  if (integer.getWidth() == 4)
    return integer.isSigned() ? "ime.fragment-nibbles-signed" : "ime.fragment-nibbles-unsigned";
  return fragment.getPacking().getRowsPerTile() == 1
      ? "rvv.fragment-chunk-slide" : llvm::StringRef(contract->chunkSplitInstruction);
}

mlir::LogicalResult IMEFragmentChunkSplitLoadOp::verify() {
  auto fragment = getLow().getType();
  const auto *contract = findRISCVFragmentCapability(fragment.getFamily());
  auto access = fragmentChunkReadAccess(getField(), fragment);
  auto target = (*this)->getParentOfType<KernelOp>().getTarget();
  const int64_t loadGroups = fragmentChunkLoadGroups(getField(), fragment, target);
  const auto parameters = getLeaf().getParameters().asArrayRef();
  if (!access || !contract || getHigh().getType() != fragment || getAccess() != *access ||
      !exactLeaf(getLeaf(), "ime", "fragment-chunk-split-load", fragmentChunkInstruction(getField(), fragment), "none", "agnostic") ||
      getLeaf().getFragmentGroups() != 2 * fragment.getResourceGroups() ||
      getLeaf().getTemporaryGroups() != fragmentChunkTemporaryGroups(getField(), fragment) || getLeaf().getLocalBytes() ||
      parameters.size() != 2 || parameters[0] != loadGroups ||
      parameters[1] != fragmentChunkNeedsRowGuard(getField(), fragment))
    return emitOpError("fragment chunk split requires complete block32 row chunks, two exact K16 results and registered packing");
  return mlir::success();
}

std::optional<AccessAttr> weft::riscv::fragmentWindowReadAccess(
    mlir::Value input, mlir::Value logicalBase, FragmentType result) {
  auto field = input.getDefiningOp<FieldOp>();
  auto owner = field ? field.getOwner().getDefiningOp<LoadOp>() : LoadOp();
  auto type = mlir::dyn_cast<ValueType>(input.getType());
  auto capability = field ? fragmentCapability(field, result.getFamily()) : FragmentCapabilityAttr();
  if (!field || !owner || owner.getSnapshotStorage() || !type || !capability ||
      type.getShape().size() != 2 || owner.getRegion().getType().getShape().size() != 2 ||
      owner.getRegion().getType().getInterleaveRows() != 0 ||
      owner.getRegion().getType().getStrides()[1] != 1 ||
      !logicalBase.getType().isIndex() || result.getLayout().getValidity() != "full" ||
      type.getLayout().getValidity() != "full" || type.getLayout().getReplicaGroup() != 1 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      result.getPacking() != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                                       : capability.getRhsPacking()) ||
      fragmentPartResourceGroups(result) != 1)
    return std::nullopt;
  auto access = field.getAccess();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  auto packing = result.getPacking();
  auto target = field->getParentOfType<KernelOp>().getTarget();
  const int64_t rowPosition = packing.getAxisOrder()[0];
  const int64_t innerPosition = packing.getAxisOrder()[1];
  const int64_t rows = result.getShape()[rowPosition];
  const int64_t columns = packing.getColumnsPerTile();
  const int64_t extent = type.getShape()[1];
  const int64_t group = access.getGroupSize(), layer = access.getLayerSize();
  auto upper = nonnegativeIndexUpperBound(logicalBase);
  if (!integer || !integer.isUnsigned() ||
      (integer.getWidth() != 1 && integer.getWidth() != 2 && integer.getWidth() != 4) || rows <= 1 ||
      rows != fragmentPartCount(result) * packing.getRowsPerTile() ||
      result.getShape()[innerPosition] != columns ||
      type.getAxisIds()[0] != result.getAxisIds()[rowPosition] ||
      type.getAxisIds()[1] != result.getAxisIds()[innerPosition] ||
      type.getShape()[0] != rows || owner.getRegion().getType().getShape() != type.getShape() ||
      extent <= 0 || extent != owner.getRegion().getType().getElements() ||
      access.getMapping() != "grouped_layered" || access.getOrder() != "lo_first" ||
      access.getFieldChunkBytes() != 1 || access.getBitOffset() % 8 ||
      group <= 0 || layer <= 0 || columns <= 0 || group % layer || extent % group ||
      layer % columns || integer.getWidth() * (group / layer) > 8 ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), columns * 64 / target.getVlenBits()) ||
      packing.getElementOrder() != "row_major" || packing.getTileOrder() != "row_major" ||
      !upper || *upper >= extent || !knownIndexMultipleOf(logicalBase, columns))
    return std::nullopt;
  const int64_t bytes = extent / group * layer;
  const int64_t offset = access.getBitOffset() / 8;
  const int64_t recordBytes = owner.getRegion().getType().getStorageBits() / 8;
  if (bytes > access.getStorageBits() / 8 || offset > recordBytes || bytes > recordBytes - offset)
    return std::nullopt;
  return access;
}

bool weft::riscv::compatibleFragmentAccumulators(FragmentType lhs, FragmentType rhs) {
  const auto *a = findRISCVFragmentCapability(lhs.getFamily());
  const auto *b = findRISCVFragmentCapability(rhs.getFamily());
  return a && b && a->extension == b->extension && lhs.getRole() == "accumulator" &&
      rhs.getRole() == "accumulator" && integerSignednessMatches(lhs.getElementType(), "signed") &&
      elementBitWidth(lhs.getElementType()) == 32 && lhs.getElementType() == rhs.getElementType() &&
      lhs.getShape() == rhs.getShape() && lhs.getAxisIds() == rhs.getAxisIds() &&
      lhs.getPacking() == rhs.getPacking() && lhs.getLayout() == rhs.getLayout() &&
      lhs.getResourceGroups() == rhs.getResourceGroups();
}

std::optional<int64_t> weft::riscv::fragmentPointwiseTemporaryGroups(
    FragmentType lhs, mlir::Type rhs) {
  if (compatibleFragmentAccumulators(lhs, lhs)) {
    auto other = mlir::dyn_cast<FragmentType>(rhs);
    if (rhs == lhs.getElementType() || (other && compatibleFragmentAccumulators(lhs, other)))
      return 0;
    return std::nullopt;
  }
  if (lhs.getRole() == "accumulator" || !integerSignednessMatches(lhs.getElementType(), "signed") ||
      elementBitWidth(lhs.getElementType()) != 8 || fragmentPartResourceGroups(lhs) != 1)
    return std::nullopt;
  if (rhs == lhs.getElementType() || rhs == lhs)
    return 0;
  auto value = mlir::dyn_cast<ValueType>(rhs);
  const int64_t rowAxis = lhs.getAxisIds()[lhs.getPacking().getAxisOrder()[0]];
  const int64_t rows = lhs.getShape()[lhs.getPacking().getAxisOrder()[0]];
  if (!value || value.getElementType() != lhs.getElementType() || value.getAxisIds().size() != 1 ||
      value.getAxisIds()[0] != rowAxis || value.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  auto layout = value.getLayout();
  const int64_t lmul = layout.getLmulEighths();
  if (layout.getLaneFactors()[0] != rows || layout.getSew() != 8 ||
      layout.getVl() != rows || layout.getValidity() == "mask" ||
      (layout.getValidity() != "full" && layout.getValidity() != lhs.getLayout().getValidity()) ||
      lmul <= 0 || lmul > 8 || (lmul & (lmul - 1)) ||
      layout.getRegisterGroups() != 1 ||
      layout.getTimeFactors()[0] != 1 || layout.getReplicaFactors()[0] != 1 ||
      layout.getFragmentFactors()[0] != 1 || layout.getLocalFactors()[0] != 1)
    return std::nullopt;
  return 2;
}

bool weft::riscv::supportsFragmentByteRepeat(
    FragmentType fragment, ValueType rows, TargetAttr target) {
  auto temporary = fragmentPointwiseTemporaryGroups(fragment, rows);
  if (!temporary || *temporary != 2 || !supportsRVVLayout(target, rows.getLayout()))
    return false;
  auto layout = rows.getLayout();
  auto packing = fragment.getPacking();
  const int64_t parts = fragmentPartCount(fragment);
  // The operand's physical rows determine coverage. A partial active M does
  // not remove any rows from a transposed RHS with a complete N-by-K packing.
  return packing.getColumnsPerTile() == 8 &&
      packing.getElementOrder() == "row_major" && packing.getTileOrder() == "row_major" &&
      fragment.getLayout().getReplicaFactors()[packing.getAxisOrder()[1]] == 1 &&
      layout.getLaneFactors()[0] == packing.getRowsPerTile() * parts &&
      packing.getRowsPerTile() * 64 == target.getVlenBits() &&
      layout.getLmulEighths() == parts && layout.getVl() * 64 == target.getVlenBits() * parts &&
      llvm::is_contained(target.getSupportedSEW().asArrayRef(), 64) &&
      llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8 * parts);
}

mlir::LogicalResult IMEFragmentRowBroadcastOp::verify() {
  auto fragment = getResult().getType();
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto temporary = fragmentPointwiseTemporaryGroups(fragment, getInput().getType());
  const int64_t parts = fragmentPartCount(fragment);
  const bool byteRepeat = getLeaf().getInstruction() == "ime.rows.byte-repeat";
  const bool preparedIndices = getLeaf().getInstruction() == "ime.rows.indices";
  if (!kernel || !fragmentCapability(*this, fragment.getFamily()) ||
      !supportsRVVLayout(kernel.getTarget(), getInput().getType().getLayout()) ||
      !temporary || *temporary != 2 || fragment.getPacking().getColumnsPerTile() != 8 ||
      !exactLeaf(getLeaf(), "ime", "fragment-broadcast",
                 byteRepeat ? "ime.rows.byte-repeat" :
                 preparedIndices ? "ime.rows.indices" : "ime.rows.local", "none", "exact") ||
      (preparedIndices && (parts <= 1 || parts > 4)) ||
      getLeaf().getTemporaryGroups() !=
          (byteRepeat || preparedIndices ? 2 * parts : parts + 1) ||
      getLeaf().getFragmentGroups() || getLeaf().getLocalBytes())
    return emitOpError("fragment row broadcast requires a complete signed-byte row projection and local index resources");
  if (byteRepeat && !supportsFragmentByteRepeat(fragment, getInput().getType(), kernel.getTarget()))
    return emitOpError("byte-repeat broadcast requires eight byte copies in each full u64 lane");
  return mlir::success();
}

bool weft::riscv::supportsFragmentWordWindows(mlir::Value field, FragmentType result) {
  auto kernel = field.getParentRegion()->getParentOfType<KernelOp>();
  if (!kernel) return false;
  auto packing = result.getPacking();
  auto target = kernel.getTarget();
  return packing.getColumnsPerTile() == 8 && packing.getRowsPerTile() * 64 == target.getVlenBits() &&
      llvm::is_contained(target.getSupportedSEW().asArrayRef(), 64) &&
      llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8);
}

namespace {
bool denseRowWindowSource(mlir::Value source, int64_t rowAxis, int64_t innerAxis,
                          int64_t rows, int64_t columns, int64_t alignment) {
  auto memory = mlir::dyn_cast<MemDescType>(source.getType());
  auto encoding = memory ? mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding())
                         : kernel::EncodingType();
  if (!memory || !encoding || encoding.getKind() != "dense" ||
      encoding.getFamily() != "i8" || memory.getStorageBits() != 8 ||
      memory.getElements() != 1 || memory.getInterleaveRows() ||
      (memory.getAccess() != "read" && memory.getAccess() != "readwrite") ||
      memory.getShape().size() != 2 || memory.getAxisIds().size() != 2 ||
      memory.getStrides().size() != 2 || rowAxis < 0 || rowAxis >= 2 ||
      innerAxis < 0 || innerAxis >= 2 || rowAxis == innerAxis ||
      !((rows >= 4 && rows <= 16 && rows % 4 == 0 && columns == 8 && alignment == 8) ||
        (rows >= 1 && rows <= 32 && columns == 16 && alignment == 1)) ||
      memory.getStrides()[innerAxis] != 1 || !isAddressableDenseDescriptor(source))
    return false;
  if (!denseDescriptorExtent(source, rowAxis) ||
      !denseDescriptorExtent(source, innerAxis))
    return false;
  return true;
}
} // namespace

std::optional<AccessAttr> weft::riscv::fragmentDenseReadAccess(
    mlir::Value source, FragmentType result) {
  auto memory = mlir::dyn_cast<MemDescType>(source.getType());
  auto *owner = source.getParentRegion()->getParentOp();
  auto kernel = mlir::dyn_cast<KernelOp>(owner);
  if (!kernel) kernel = owner->getParentOfType<KernelOp>();
  auto packing = result.getPacking();
  FragmentCapabilityAttr capability;
  if (kernel)
    for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
      auto candidate = mlir::cast<FragmentCapabilityAttr>(attribute);
      if (candidate.getInstruction() == result.getFamily()) capability = candidate;
    }
  if (!memory || !kernel || !capability || result.getAxisIds().size() != 2 ||
      result.getShape().size() != 2 || packing.getAxisOrder().size() != 2 ||
      (result.getRole() != "lhs" && result.getRole() != "rhs") ||
      packing != (result.getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      !((packing.getRowsPerTile() == 4 && packing.getColumnsPerTile() == 8) ||
        ((packing.getRowsPerTile() == 1 || packing.getRowsPerTile() == 4 || packing.getRowsPerTile() == 8) &&
         packing.getColumnsPerTile() == 16)) ||
      packing.getStorageBits() != 8 || packing.getElementOrder() != "row_major" ||
      packing.getTileOrder() != "row_major" || fragmentPartResourceGroups(result) != 1 ||
      !integerSignednessMatches(result.getElementType(), "signed") ||
      elementBitWidth(result.getElementType()) != 8)
    return std::nullopt;
  auto target = kernel.getTarget();
  const int64_t row = packing.getAxisOrder()[0], inner = packing.getAxisOrder()[1];
  const int64_t parts = fragmentPartCount(result), rows = result.getShape()[row];
  const bool bytes = packing.getColumnsPerTile() == 16;
  if (parts < 1 || parts > 4 || rows != parts * packing.getRowsPerTile() ||
      result.getShape()[inner] != packing.getColumnsPerTile() ||
      result.getLayout().getReplicaFactors()[inner] != 1 ||
      !llvm::all_of(result.getLayout().getTimeFactors().asArrayRef(), [](int64_t factor) { return factor == 1; }) ||
      (bytes ? target.getVlenBits() < packing.getRowsPerTile() * packing.getColumnsPerTile() * 8 :
               target.getVlenBits() != packing.getRowsPerTile() * 64) ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(), bytes ? 8 : 64) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 8))
    return std::nullopt;
  auto axes = memory.getAxisIds().asArrayRef();
  auto rowPosition = llvm::find(axes, result.getAxisIds()[row]);
  auto innerPosition = llvm::find(axes, result.getAxisIds()[inner]);
  if (rowPosition == axes.end() || innerPosition == axes.end() ||
      !denseRowWindowSource(source, rowPosition - axes.begin(), innerPosition - axes.begin(),
                            rows, packing.getColumnsPerTile(), bytes ? 1 : 8))
    return std::nullopt;
  return AccessAttr::get(source.getContext(), "strided", "dense", bytes ? 1 : 8,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, "none", 1);
}

mlir::LogicalResult DenseRowWindowGuardOp::verify() {
  if (!denseRowWindowSource(getSource(), getRowAxis(), getInnerAxis(),
                            getRows(), getColumns(), getAlignment()))
    return emitOpError("dense row guard requires bounded signed-byte row windows, explicit extents and unit inner stride");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentDenseLoadOp::verify() {
  auto fragment = getResult().getType();
  auto expected = fragmentDenseReadAccess(getSource(), fragment);
  const bool bytes = expected && expected->getAlignment() == 1;
  const llvm::StringRef instruction = bytes ? "ime.dense-row-bytes" : "ime.dense-row-words";
  if (!expected || getAccess() != *expected ||
      !exactLeaf(getLeaf(), "ime", "fragment-dense-load", instruction, "none", "agnostic") ||
      getLeaf().getTemporaryGroups() != (bytes ? 1 : 0) || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != fragment.getResourceGroups() ||
      !getLeaf().getParameters().empty())
    return emitOpError("dense fragment load requires its exact capability packing, word transfer and resource contract");
  if (bytes) return mlir::success();
  auto packing = fragment.getPacking();
  auto axes = getSource().getType().getAxisIds().asArrayRef();
  const int64_t row = packing.getAxisOrder()[0], inner = packing.getAxisOrder()[1];
  const int64_t rowPosition = llvm::find(axes, fragment.getAxisIds()[row]) - axes.begin();
  const int64_t innerPosition = llvm::find(axes, fragment.getAxisIds()[inner]) - axes.begin();
  for (mlir::Operation *child = getOperation(), *parent = child->getParentOp();
       parent; child = parent, parent = parent->getParentOp()) {
    auto branch = mlir::dyn_cast<mlir::scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getThenRegion()) continue;
    auto guard = branch.getCondition().getDefiningOp<DenseRowWindowGuardOp>();
    if (guard && guard.getSource() == getSource() && guard.getRowAxis() == rowPosition &&
        guard.getInnerAxis() == innerPosition && guard.getRows() == fragment.getShape()[row] &&
        guard.getColumns() == packing.getColumnsPerTile() && guard.getAlignment() == 8)
      return mlir::success();
  }
  return emitOpError("dense fragment load requires the matching row-window and alignment guard true branch");
}

mlir::LogicalResult IMEFragmentWindowLoadOp::verify() {
  auto expected = fragmentWindowReadAccess(getField(), getLogicalBase(), getResult().getType());
  bool guardedWords = supportsFragmentWordWindows(getField(), getResult().getType()) &&
      exactLeaf(getLeaf(), "ime", "fragment-window-load", "ime.field-byte-windows.guarded-vlse64", "none", "agnostic");
  if (!expected || getAccess() != *expected ||
      (!guardedWords && !exactLeaf(getLeaf(), "ime", "fragment-window-load", "ime.field-byte-windows", "none", "agnostic")) ||
      getLeaf().getTemporaryGroups() != getResult().getType().getPacking().getRowsPerTile() || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
    return emitOpError("fragment byte windows require complete record rows, aligned in-layer byte spans, and exact fragment packing");
  return mlir::success();
}

bool weft::riscv::supportsFragmentGroupedLoad(FragmentType fragment, TargetAttr target) {
  const int64_t parts = fragmentPartCount(fragment);
  auto packing = fragment.getPacking();
  return (parts == 2 || parts == 4) &&
      fragment.getLayout().getValidity() == "full" &&
      (packing.getStorageBits() == 8 || packing.getStorageBits() == 4) &&
      packing.getRowsPerTile() * packing.getColumnsPerTile() * packing.getStorageBits() == target.getVlenBits() &&
      llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), parts * 8);
}

mlir::LogicalResult IMEFragmentStorageLoadOp::verify() {
  if (getLeaf().getInstruction() == "ime.field-nibble-cohorts") {
    auto expected = fragmentNibbleCohortReadAccess(getInput(), getResult().getType());
    if (!expected || getAccess() != *expected ||
        !exactLeaf(getLeaf(), "ime", "fragment-storage-load", "ime.field-nibble-cohorts", "none", "agnostic") ||
        !getLeaf().getParameters().empty() || getLeaf().getTemporaryGroups() != 2 || getLeaf().getLocalBytes() ||
        getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
      return emitOpError("packed nibble supply requires two complete four-row cohorts per fragment");
    return mlir::success();
  }
  if (getLeaf().getInstruction() == "ime.field-nibble-pack") {
    auto expected = fragmentNibblePackReadAccess(getInput(), getResult().getType());
    auto parameters = getLeaf().getParameters().asArrayRef();
    if (!expected || getAccess() != *expected ||
        !exactLeaf(getLeaf(), "ime", "fragment-storage-load", "ime.field-nibble-pack", "none", "agnostic") ||
        parameters.size() != 1 || (parameters[0] != 0 && parameters[0] != 4) ||
        getLeaf().getTemporaryGroups() != 5 || getLeaf().getLocalBytes() ||
        getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
      return emitOpError("nibble pack read requires exact block32 byte rows and a low/high nibble selection");
    return mlir::success();
  }
  auto expected = fragmentFieldReadAccess(getInput(), getResult().getType());
  const auto instruction = getLeaf().getInstruction();
  const bool grouped = instruction == "ime.field-raw-load.grouped" ||
                       instruction == "ime.field-raw-load.grouped.scalar-prime";
  const bool prime = instruction == "ime.field-raw-load.scalar-prime" ||
                     instruction == "ime.field-raw-load.grouped.scalar-prime";
  const bool windows = expected && expected->getForm() == "strided";
  auto kernel = (*this)->getParentOfType<KernelOp>();
  const char *spelling = windows ? "ime.field-raw-load.row-windows" :
                        grouped ? (prime ? "ime.field-raw-load.grouped.scalar-prime" : "ime.field-raw-load.grouped")
                                 : (prime ? "ime.field-raw-load.scalar-prime" : "ime.field-raw-load");
  if (!expected || getAccess() != *expected ||
      !exactLeaf(getLeaf(), "ime", "fragment-storage-load", spelling, "none", "agnostic") ||
      getLeaf().getTemporaryGroups() != (grouped ? 0 : 1) || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != getResult().getType().getResourceGroups())
    return emitOpError("fragment field load requires its proven row/chunk geometry and exact transfer leaf");
  if (windows && (grouped || prime))
    return emitOpError("strided row windows cannot use a contiguous grouped or primed transfer");
  if (grouped && (!kernel || !supportsFragmentGroupedLoad(getResult().getType(), kernel.getTarget())))
    return emitOpError("grouped fragment load requires two or four complete register atoms in a legal RVV group");
  if (prime && (getResult().getType().getLayout().getValidity() != "full" ||
      getResult().getType().getPacking().getStorageBits() != 8 ||
      fragmentPartCount(getResult().getType()) < 2 || fragmentPartCount(getResult().getType()) > 4))
    return emitOpError("fragment scalar prime requires a complete two-to-four-atom byte supply");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentDecodeOp::verify() {
  auto fragment = getInput().getType();
  const int64_t bits = getElementBits();
  const bool grouped = getLeaf().getInstruction() == "ime.decode.grouped-bits";
  auto kernel = (*this)->getParentOfType<KernelOp>();
  auto maximum = nonnegativeIndexUpperBound(getShift());
  if (getResult().getType() != fragment || fragment.getRole() == "accumulator" ||
      fragment.getPacking().getStorageBits() != 8 ||
      fragmentPartResourceGroups(fragment) != 1 ||
      (bits != 1 && bits != 2 && bits != 4) || !maximum || *maximum > 8 - bits ||
      !knownIndexMultipleOf(getShift(), bits) ||
      !exactLeaf(getLeaf(), "ime", "fragment-decode",
                 grouped ? "ime.decode.grouped-bits" : "ime.decode.bits", "none", "exact") ||
      getLeaf().getTemporaryGroups() != (grouped ? 0 : 1) || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != fragment.getResourceGroups())
    return emitOpError("fragment bit projection requires a bounded byte-local shift and closed replicas");
  if (grouped && (!kernel || !supportsFragmentGroupedLoad(fragment, kernel.getTarget())))
    return emitOpError("grouped fragment decode requires complete register atoms in one legal RVV group");
  return mlir::success();
}

mlir::LogicalResult IMEFragmentPointwiseOp::verify() {
  auto temporaries = fragmentPointwiseTemporaryGroups(getLhs().getType(), getRhs().getType());
  auto rhs = mlir::dyn_cast<ValueType>(getRhs().getType());
  auto kernel = (*this)->getParentOfType<KernelOp>();
  const bool reuse = getLhsPolicy() == "reuse";
  const auto baseInstruction = ("ime.fragment." + getKind()).str();
  const bool grouped = getLeaf().getInstruction() == baseInstruction + ".grouped";
  const bool accumulator = compatibleFragmentAccumulators(getLhs().getType(), getResult().getType());
  if (!temporaries || !fragmentCapability(*this, getResult().getType().getFamily()) ||
      (rhs && (!kernel || !supportsRVVLayout(kernel.getTarget(), rhs.getLayout()))) ||
      (getResult().getType() != getLhs().getType() && !accumulator) ||
      (getKind() != "add" && getKind() != "sub" && !(accumulator && (getKind() == "mul" || getKind() == "shl"))) ||
      (getLhsPolicy() != "copy" && !reuse) || (reuse && !canReuseFragmentPointwise(*this)) ||
      !exactLeaf(getLeaf(), "ime", "fragment-pointwise",
                 baseInstruction + (grouped ? ".grouped" : ""), "none", "exact") ||
      getLeaf().getTemporaryGroups() != *temporaries || getLeaf().getLocalBytes() ||
      getLeaf().getFragmentGroups() != (reuse ? 0 : getResult().getType().getResourceGroups()))
    return emitOpError("fragment pointwise requires its exact integer domain and scalar/row broadcast");
  if (getKind() == "shl") {
    auto constant = getRhs().getDefiningOp<ConstantOp>();
    auto value = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()) : mlir::IntegerAttr();
    if (!value || value.getValue().getSExtValue() < 0 || value.getValue().getSExtValue() >= 32)
      return emitOpError("accumulator shift requires a scalar constant in [0, 31]");
  }
  if (grouped && (getRhs().getType() != getLhs().getType() || !kernel ||
                  !supportsFragmentGroupedLoad(getLhs().getType(), kernel.getTarget())))
    return emitOpError("grouped fragment pointwise requires two matching complete register groups");
  return mlir::success();
}

FragmentCapabilityAttr weft::riscv::fragmentCapabilityAttr(
    mlir::MLIRContext *context, const RISCVFragmentCapability &capability) {
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Attribute> clobbers;
  for (const std::string &clobber : capability.clobbers)
    clobbers.push_back(builder.getStringAttr(clobber));
  return FragmentCapabilityAttr::get(
      context, capability.name, signedness(capability.lhsSignedness),
      signedness(capability.rhsSignedness), capability.lhsElementBits,
      capability.rhsElementBits, capability.accumulatorElementBits,
      capability.mFactor, capability.nFactor, capability.kFactor, capability.activeMFactor,
      capability.lhsResourceGroups, capability.rhsResourceGroups,
      capability.accumulatorResourceGroups,
      packingAttr(context, capability.lhsPacking),
      packingAttr(context, capability.rhsPacking),
      packingAttr(context, capability.accumulatorPacking),
      capability.mmaGroups, capability.mmaChunks, builder.getArrayAttr(clobbers),
      capability.volatileAsm, capability.memoryClobber);
}

bool weft::riscv::fragmentAccumulatorTypeMatches(
    FragmentCapabilityAttr capability, mlir::Type element) {
  const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
  return contract && elementBitWidth(element) == capability.getAccumulatorBits() &&
         integerSignednessMatches(element, signedness(contract->accumulatorSignedness));
}

int64_t weft::riscv::fragmentPartCount(FragmentType fragment) {
  auto replicas = fragment.getLayout().getReplicaFactors();
  return replicas[0] * replicas[1];
}

int64_t weft::riscv::fragmentPartResourceGroups(FragmentType fragment) {
  return fragment.getResourceGroups() / fragmentPartCount(fragment);
}

bool weft::riscv::canReuseFragmentAccumulator(IMEFragmentMMAOp operation) {
  mlir::Value accumulator = operation.getAccumulator();
  if (!accumulator.hasOneUse())
    return false;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(accumulator))
    return argument.getOwner() == operation->getBlock();
  auto *definition = accumulator.getDefiningOp();
  return definition && definition->getBlock() == operation->getBlock();
}

bool weft::riscv::canReuseFragmentPointwise(IMEFragmentPointwiseOp operation) {
  auto lhs = operation.getLhs();
  if (!lhs.hasOneUse()) return false;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(lhs))
    return argument.getOwner() == operation->getBlock();
  auto *definition = lhs.getDefiningOp();
  return definition && definition->getBlock() == operation->getBlock();
}

int64_t weft::riscv::fragmentMMAAdditionalGroups(FragmentCapabilityAttr capability, int64_t atoms,
                                               bool reuseAccumulator) {
  llvm::SmallVector<unsigned> registers;
  for (mlir::Attribute attribute : capability.getClobbers()) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(attribute).getValue();
    unsigned number = 0;
    if (name.consume_front("v") && !name.getAsInteger(10, number) &&
        number < 32 && !llvm::is_contained(registers, number))
      registers.push_back(number);
  }
  // A tied accumulator may need a copy when its input SSA value remains live.
  // Reserve that result group as well as any fixed registers beyond operands.
  return (reuseAccumulator ? 0 : capability.getAccumulatorResourceGroups() * atoms) +
         std::max<int64_t>(0, static_cast<int64_t>(registers.size()) -
                                capability.getLhsResourceGroups() -
                                capability.getRhsResourceGroups() -
                                capability.getAccumulatorResourceGroups());
}

mlir::LogicalResult FragmentPackingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef schema, mlir::DenseI64ArrayAttr axisOrder,
    int64_t rowsPerTile, int64_t columnsPerTile,
    llvm::StringRef tileOrder, llvm::StringRef elementOrder,
    int64_t storageBits, int64_t alignment) {
  if (schema.empty() || rowsPerTile <= 0 || columnsPerTile <= 0 ||
      !axisOrder || axisOrder.size() != 2 || axisOrder[0] == axisOrder[1] ||
      llvm::any_of(axisOrder.asArrayRef(),
                   [](int64_t axis) { return axis < 0 || axis > 1; }) ||
      storageBits <= 0 || (storageBits != 4 && storageBits % 8) ||
      (rowsPerTile * columnsPerTile * storageBits) % 8 || alignment <= 0)
    return emitError()
           << "fragment packing requires one rank-two byte-addressable tiled storage mapping";
  auto validOrder = [](llvm::StringRef order) {
    return order == "row_major" || order == "column_major";
  };
  if (!validOrder(tileOrder) || !validOrder(elementOrder))
    return emitError()
           << "fragment packing requires explicit tile and in-tile orders";
  return mlir::success();
}

mlir::LogicalResult FragmentCapabilityAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef instruction, llvm::StringRef lhsSignedness,
    llvm::StringRef rhsSignedness, int64_t lhsBits, int64_t rhsBits,
    int64_t accumulatorBits, int64_t mFactor, int64_t nFactor,
    int64_t kFactor, int64_t activeMFactor, int64_t lhsResourceGroups, int64_t rhsResourceGroups,
    int64_t accumulatorResourceGroups, FragmentPackingAttr lhsPacking,
    FragmentPackingAttr rhsPacking, FragmentPackingAttr accumulatorPacking,
    int64_t mmaGroups, int64_t mmaChunks, mlir::ArrayAttr clobbers,
    bool volatileAsm, bool memoryClobber) {
  const auto *contract = findRISCVFragmentCapability(instruction);
  if (!contract)
    return emitError() << "unsupported fragment instruction " << instruction;
  if (!lhsPacking || !rhsPacking || !accumulatorPacking || !clobbers)
    return emitError() << "fragment capability requires packing and clobbers";
  auto *context = lhsPacking.getContext();
  if (lhsSignedness != signedness(contract->lhsSignedness) ||
      rhsSignedness != signedness(contract->rhsSignedness) ||
      lhsBits != contract->lhsElementBits || rhsBits != contract->rhsElementBits ||
      accumulatorBits != contract->accumulatorElementBits ||
      mFactor != contract->mFactor || nFactor != contract->nFactor ||
      kFactor != contract->kFactor || activeMFactor != contract->activeMFactor ||
      lhsResourceGroups != contract->lhsResourceGroups ||
      rhsResourceGroups != contract->rhsResourceGroups ||
      accumulatorResourceGroups != contract->accumulatorResourceGroups ||
      lhsPacking != packingAttr(context, contract->lhsPacking) ||
      rhsPacking != packingAttr(context, contract->rhsPacking) ||
      accumulatorPacking != packingAttr(context, contract->accumulatorPacking) ||
      mmaGroups != contract->mmaGroups || mmaChunks != contract->mmaChunks ||
      volatileAsm != contract->volatileAsm || memoryClobber != contract->memoryClobber ||
      clobbers.size() != contract->clobbers.size())
    return emitError() << "fragment capability disagrees with its registered ISA contract";
  for (auto [attribute, name] : llvm::zip(clobbers, contract->clobbers)) {
    auto clobber = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!clobber || clobber.getValue() != name)
      return emitError() << "fragment clobbers disagree with the registered ISA contract";
  }
  if (instruction.empty() || lhsBits <= 0 || rhsBits <= 0 ||
      accumulatorBits <= 0 || mFactor <= 0 || nFactor <= 0 || kFactor <= 0 ||
      activeMFactor <= 0 || activeMFactor > mFactor ||
      (lhsSignedness != "signed" && lhsSignedness != "unsigned") ||
      (rhsSignedness != "signed" && rhsSignedness != "unsigned") ||
      lhsResourceGroups <= 0 || rhsResourceGroups <= 0 ||
      accumulatorResourceGroups <= 0 || !lhsPacking || !rhsPacking ||
      !accumulatorPacking ||
      mmaGroups <= 0 || mmaChunks <= 0 || !volatileAsm ||
      clobbers.empty() ||
      llvm::any_of(clobbers, [](mlir::Attribute attribute) {
        auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
        return !value || value.getValue().empty();
      }))
    return emitError() << "fragment capability fields must be positive and complete";
  if (lhsPacking.getRowsPerTile() * lhsPacking.getColumnsPerTile() *
              lhsPacking.getStorageBits() / 8 <=
          0 ||
      rhsPacking.getRowsPerTile() * rhsPacking.getColumnsPerTile() *
              rhsPacking.getStorageBits() / 8 <=
          0 ||
      accumulatorPacking.getRowsPerTile() *
              accumulatorPacking.getColumnsPerTile() *
              accumulatorPacking.getStorageBits() / 8 <=
          0)
    return emitError() << "fragment packing tile geometry is incomplete";
  if (mFactor % lhsPacking.getRowsPerTile() ||
      kFactor % lhsPacking.getColumnsPerTile() ||
      nFactor % rhsPacking.getRowsPerTile() ||
      kFactor % rhsPacking.getColumnsPerTile() ||
      mFactor % accumulatorPacking.getRowsPerTile() ||
      nFactor % accumulatorPacking.getColumnsPerTile() ||
      lhsPacking.getStorageBits() < lhsBits ||
      rhsPacking.getStorageBits() < rhsBits ||
      accumulatorPacking.getStorageBits() < accumulatorBits)
    return emitError()
           << "fragment packing geometry does not cover the capability operands";
  return mlir::success();
}

std::optional<int64_t> weft::riscv::fragmentZeroPrefixWidth(FragmentType fragment, TargetAttr target) {
  if (!target || target.getSupportedSEW().empty()) return std::nullopt;
  const int64_t bits = elementBitWidth(fragment.getElementType());
  const int64_t active = fragment.getPacking().getRowsPerTile() *
      fragment.getPacking().getColumnsPerTile() * bits;
  const int64_t elen = *llvm::max_element(target.getSupportedSEW().asArrayRef());
  std::optional<int64_t> selected;
  for (int64_t width : target.getLegalLMULEighths().asArrayRef())
    if (width > 0 && width < fragmentPartResourceGroups(fragment) * 8 &&
        bits * 8 <= elen * width && active * 8 <= width * target.getVlenBits() &&
        (!selected || width < *selected))
      selected = width;
  return selected;
}

mlir::LogicalResult IMEZeroOp::verify() {
  auto fragment = getResult().getType();
  auto capability = fragmentCapability(*this, fragment.getFamily());
  if (!capability || fragment.getRole() != "accumulator" ||
      fragment.getPacking() != capability.getAccumulatorPacking() ||
      !fragmentAccumulatorTypeMatches(capability, fragment.getElementType()) ||
      (!exactLeaf(getLeaf(), "ime", "fragment-init", "ime.zero", "none", "exact") &&
       !exactLeaf(getLeaf(), "ime", "fragment-init", "ime.zero.local", "none", "exact") &&
       !exactLeaf(getLeaf(), "ime", "fragment-init", "ime.zero.local-prefix", "none", "exact")) ||
      getLeaf().getLocalBytes() || getLeaf().getTemporaryGroups() ||
      getLeaf().getFragmentGroups())
    return emitOpError("fragment zero requires a closed accumulator representation");
  if (getLeaf().getInstruction() == "ime.zero.local-prefix") {
    auto target = (*this)->getParentOfType<KernelOp>().getTarget();
    auto parameters = getLeaf().getParameters().asArrayRef();
    auto width = fragmentZeroPrefixWidth(fragment, target);
    if (parameters.size() != 1 || !width || parameters[0] != *width)
      return emitOpError("local fragment zero prefix must cover all logical cells in the selected narrower legal carrier");
  } else if (!getLeaf().getParameters().empty()) {
    return emitOpError("ordinary fragment zero has no prefix carrier parameter");
  }
  return mlir::success();
}

void IMEFragmentMMAOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  auto *storage = PrimitiveStorageResource::get();
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(0), storage);
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(1), storage);
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(2), storage);
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getOperation()->getResult(0), storage);
}

mlir::LogicalResult IMEFragmentMMAOp::verify() {
  FragmentType lhs = getLhs().getType();
  FragmentType rhs = getRhs().getType();
  FragmentType result = getResult().getType();
  auto capability = fragmentCapability(*this, getLeaf().getInstruction());
  const auto *contract = findRISCVFragmentCapability(getLeaf().getInstruction());
  const bool reuse = getAccumulatorPolicy() == "reuse";
  if ((getAccumulatorPolicy() != "copy" && !reuse) ||
      (reuse && !canReuseFragmentAccumulator(*this)))
    return emitOpError("fragment accumulator reuse requires its unique same-block final use");
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-mma",
                 capability.getInstruction(), "none", "exact") ||
      lhs.getFamily() != rhs.getFamily() || result.getFamily() != lhs.getFamily() ||
      lhs.getFamily() != capability.getInstruction() ||
      getAccumulator().getType() != result ||
      lhs.getRole() != "lhs" || rhs.getRole() != "rhs" ||
      result.getRole() != "accumulator" ||
      lhs.getPacking() != capability.getLhsPacking() ||
      rhs.getPacking() != capability.getRhsPacking() ||
      result.getPacking() != capability.getAccumulatorPacking() ||
      elementBitWidth(lhs.getElementType()) != capability.getLhsBits() ||
      elementBitWidth(rhs.getElementType()) != capability.getRhsBits() ||
      !integerSignednessMatches(lhs.getElementType(),
                                capability.getLhsSignedness()) ||
      !integerSignednessMatches(rhs.getElementType(),
                                capability.getRhsSignedness()) ||
      !fragmentAccumulatorTypeMatches(capability, result.getElementType()) ||
      getLeaf().getFragmentGroups() != fragmentMMAAdditionalGroups(capability, fragmentPartCount(result), reuse) ||
      getGroups() != capability.getMmaGroups() ||
      getChunks() != capability.getMmaChunks() ||
      getAsmClobbers() != capability.getClobbers() ||
      getVolatileAsm() != capability.getVolatileAsm() ||
      getMemoryClobber() != capability.getMemoryClobber() ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}) ||
      lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      getLeaf().getLocalBytes() != contract->mmaLocalBytes)
    return emitOpError("IME fragment MMA requires an IME leaf contract");
  if (lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      lhs.getShape()[0] != result.getShape()[0] ||
      lhs.getShape()[1] != capability.getKFactor() ||
      rhs.getShape()[0] != capability.getKFactor() ||
      rhs.getShape()[1] != result.getShape()[1] ||
      lhs.getAxisIds()[1] != rhs.getAxisIds()[0] ||
      lhs.getAxisIds()[0] != result.getAxisIds()[0] ||
      rhs.getAxisIds()[1] != result.getAxisIds()[1])
    return emitOpError(
        "IME fragment MMA axes and shapes do not match the selected capability");
  if (fragmentPartResourceGroups(lhs) != capability.getLhsResourceGroups() ||
      fragmentPartResourceGroups(rhs) != capability.getRhsResourceGroups() ||
      fragmentPartResourceGroups(result) != capability.getAccumulatorResourceGroups())
    return emitOpError(
        "IME fragment resources do not match the target capability");
  return mlir::success();
}

mlir::LogicalResult IMEPackOp::verify() {
  FragmentType result = getResult().getType();
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto capability = fragmentCapability(*this, result.getFamily());
  const auto *contract = findRISCVFragmentCapability(result.getFamily());
  if (!input || !capability || !contract ||
      (getRole() != "lhs" && getRole() != "rhs") ||
      !exactLeaf(getLeaf(), "ime", "fragment-pack",
                 getRole() == "lhs" ? "ime.pack.lhs" : "ime.pack.rhs",
                 "none", "exact") ||
      getPacking() != (getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      result.getRole() != getRole() || result.getPacking() != getPacking() ||
      fragmentPartResourceGroups(result) !=
          (getRole() == "lhs" ? capability.getLhsResourceGroups()
                              : capability.getRhsResourceGroups()) ||
      (getPacking().getStorageBits() != 8 && getPacking().getStorageBits() != 4) ||
      result.getLayout().getCarrier() != "ime" ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}))
    return emitOpError("IME pack requires one typed fragment operand contract");
  const bool lhs = getRole() == "lhs";
  auto packing = getPacking();
  const int64_t expectedBits = lhs ? capability.getLhsBits()
                                   : capability.getRhsBits();
  const llvm::StringRef expectedSignedness =
      lhs ? capability.getLhsSignedness() : capability.getRhsSignedness();
  if (elementBitWidth(result.getElementType()) != expectedBits ||
      elementBitWidth(input.getElementType()) != expectedBits ||
      !integerSignednessMatches(input.getElementType(), expectedSignedness) ||
      input.getAxisIds().size() != result.getAxisIds().size() ||
      !llvm::all_of(input.getAxisIds().asArrayRef(), [&](int64_t axis) {
        return llvm::is_contained(result.getAxisIds().asArrayRef(), axis);
      }))
    return emitOpError("IME pack input and fragment logical domains must agree");
  auto layout = input.getLayout();
  if (packing.getStorageBits() == 4) {
    auto target = (*this)->getParentOfType<KernelOp>().getTarget();
    if (layout.getLmulEighths() < 2 || !llvm::is_contained(target.getSupportedSEW().asArrayRef(), 16) ||
        !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), layout.getLmulEighths() / 2))
      return emitOpError("nibble register packing requires its half-width byte carrier and 16-bit reinterpretation");
  }
  int64_t laneAxes = 0;
  if (layout.getCarrier() != "rvv" || layout.getSew() != 8 ||
      input.getShape().size() != layout.getTimeFactors().size() ||
      input.getShape().size() != layout.getLaneFactors().size() ||
      input.getShape().size() != layout.getReplicaFactors().size() ||
      input.getShape().size() != layout.getFragmentFactors().size() ||
      input.getShape().size() != layout.getLocalFactors().size())
    return emitOpError("IME pack input requires one complete eight-bit RVV mapping");
  for (auto [axis, time, lane, replica, fragment, local] :
       llvm::zip(input.getAxisIds().asArrayRef(), layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef())) {
    const auto position = llvm::find(result.getAxisIds().asArrayRef(), axis) -
                          result.getAxisIds().asArrayRef().begin();
    const int64_t extent = result.getShape()[position];
    if (extent <= 0 || time <= 0 || lane <= 0 || replica <= 0 ||
        fragment <= 0 || local <= 0 ||
        time * lane * replica * fragment * local != extent)
      return emitOpError("IME pack input time/lane/register factors are incomplete");
    laneAxes += lane > 1;
  }
  if (laneAxes != 1)
    return emitOpError("IME pack input requires exactly one SIMD lane axis");
  if ((lhs && (result.getShape().size() != 2 ||
               result.getShape()[1] != capability.getKFactor())) ||
      (!lhs && (result.getShape().size() != 2 ||
                result.getShape()[0] != capability.getKFactor())))
    return emitOpError("IME pack logical shape disagrees with the target fragment");
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  const int64_t rows = result.getShape()[axisOrder[0]];
  const int64_t columns = result.getShape()[axisOrder[1]];
  const int64_t innerAxis = result.getAxisIds()[axisOrder[1]];
  for (auto [axis, time, lane, replica] : llvm::zip(input.getAxisIds().asArrayRef(),
           layout.getTimeFactors().asArrayRef(), layout.getLaneFactors().asArrayRef(),
           layout.getReplicaFactors().asArrayRef()))
    if (time != 1 || lane != (axis == innerAxis ? columns : 1) ||
        replica != (axis == innerAxis ? 1 : rows))
      return emitOpError("register fragment pack requires its selected contiguous inner-axis slices");
  if (layout.getLmulEighths() > 8 || rows % packing.getRowsPerTile() ||
      columns != packing.getColumnsPerTile() || packing.getElementOrder() != "row_major")
    return emitOpError("register fragment pack requires selected row-major atom tiles");
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() < expectedBits)
    return emitOpError(
        "IME operand shape and element width do not fit the selected tiled packing");
  if (getLeaf().getLocalBytes() !=
          (lhs ? contract->lhsPackLocalBytes : contract->rhsPackLocalBytes) ||
      getLeaf().getTemporaryGroups() != contract->packTemporaryGroups)
    return emitOpError(
        "IME pack leaf must account for every primitive-private byte");
  if (getLeaf().getFragmentGroups() != result.getResourceGroups())
    return emitOpError("fragment pack must account for result replicas while input slices remain live");
  if (getAccess().getMapping() == "opaque" || getAccess().getForm() == "opaque")
    return emitOpError("IME operand has no selected physical access form");
  return mlir::success();
}

mlir::LogicalResult IMEUnpackOp::verify() {
  auto capability = fragmentCapability(*this, getInput().getType().getFamily());
  const auto *contract = findRISCVFragmentCapability(getInput().getType().getFamily());
  if (!capability || !contract ||
      !exactLeaf(getLeaf(), "ime", "fragment-unpack", fragmentUnpackInstruction(getInput().getType(), getResult().getType()),
                 "none", "exact") ||
      getConversion().getKind() != "fragment_to_rvv" ||
      getConversion().getEffect() != "handoff" ||
      getInput().getType().getRole() != "accumulator" ||
      getInput().getType().getPacking() !=
          capability.getAccumulatorPacking() ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      getInput().getType().getAxisIds() != getResult().getType().getAxisIds() ||
      getInput().getType().getElementType() !=
          getResult().getType().getElementType() ||
      !fragmentAccumulatorTypeMatches(capability, getResult().getType().getElementType()) ||
      getLeaf().getLocalBytes() != 0 ||
      getLeaf().getTemporaryGroups() != fragmentUnpackTemporaryGroups(
          getInput().getType(), getResult().getType()) ||
      getConversion().getTemporaryGroups() != getLeaf().getTemporaryGroups())
    return emitOpError("IME unpack must preserve a fragment logical domain into RVV");
  auto fragment = getInput().getType();
  auto layout = getResult().getType().getLayout();
  const bool grouped = isGroupedFragmentRVVHandoff(fragment, getResult().getType());
  const bool registeredRows = fragmentUnpackInstruction(fragment, getResult().getType()) != "ime.unpack.rvv";
  if (layout.getAxisIds().size() != 2 ||
      (layout.getLaneFactors()[0] != 1 && !grouped && !registeredRows && !isDirectFragmentRVVHandoff(fragment, getResult().getType())) ||
      (!grouped && (layout.getReplicaGroup() != 1 ||
                    layout.getLmulEighths() > fragmentPartResourceGroups(fragment) * 8)) ||
      llvm::any_of(layout.getFragmentFactors().asArrayRef(), [](int64_t factor) { return factor != 1; }) ||
      llvm::any_of(layout.getLocalFactors().asArrayRef(), [](int64_t factor) { return factor != 1; }))
    return emitOpError("fragment unpack requires selected contiguous output-row slices");
  for (auto [extent, time, lane, replica, fragmentFactor, local] :
       llvm::zip(fragment.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef()))
    if (time * lane * replica * fragmentFactor * local != extent)
      return emitOpError(
          "IME unpack result layout does not cover the complete fragment domain");
  return mlir::success();
}
