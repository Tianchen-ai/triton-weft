#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Transforms/Memory/FieldStores.h"
#include "Weft/Analysis/StorageBytes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <numeric>
#include <map>

using namespace weft;

namespace {
mlir::LogicalResult materializeFieldBytes(mlir::IRRewriter &b, riscv::StoreOp op) {
  auto field = op.getRegion().getDefiningOp<riscv::FieldOp>();
  auto owner = mlir::dyn_cast<riscv::MemDescType>(field.getOwner().getType());
  auto encoding = owner ? mlir::dyn_cast<kernel::EncodingType>(owner.getEncoding()) : kernel::EncodingType();
  if (!owner || !encoding || owner.getShape().empty() || owner.getShape().asArrayRef().back() != owner.getElements())
    return op.emitError("field-byte program requires an explicit complete record along the encoded element axis");
  auto declaration = riscv_internal::findEncoding(op, riscv_internal::baseEncodingFamily(op, encoding));
  if (!declaration) return op.emitError("field-byte program has no declared base Encoding");
  unsigned ordinal = 0;
  for (; ordinal < declaration.getFieldNames().size(); ++ordinal)
    if (mlir::cast<mlir::StringAttr>(declaration.getFieldNames()[ordinal]).getValue() == field.getName()) break;
  if (ordinal == declaration.getFieldNames().size()) return op.emitError("field-byte program cannot resolve the declared field");
  auto element = mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[ordinal]).getValue();
  auto relation = analysis::fieldBitLayout(element, mlir::cast<mlir::ArrayAttr>(declaration.getFieldLayouts()[ordinal]));
  if (relation.elementBits > 64) return op.emitError("field-byte program requires an element no wider than 64 bits");
  auto shape = mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[ordinal]).asArrayRef();
  auto count = riscv_internal::staticProduct(shape);
  if (!count || *count <= 0) return op.emitError("field-byte program requires a bounded field coordinate domain");
  int64_t offset = declaration.getFieldBitOffsets()[ordinal];
  int64_t spanOffset = offset / 8;
  int64_t spanBytes = (offset % 8 + declaration.getFieldStorageBits()[ordinal] + 7) / 8;
  auto facts = riscv_internal::fieldFacts(field);
  int64_t chunk = facts.chunkBytes;
  auto valueType = mlir::dyn_cast<riscv::ValueType>(op.getValue().getType());
  unsigned outerRank = owner.getShape().size() - 1;
  if ((valueType && valueType.getShape().size() != outerRank + shape.size()) ||
      (!valueType && (outerRank || !shape.empty())))
    return op.emitError("field-byte program requires the complete record/free-axis projection");
  llvm::SmallVector<llvm::SmallVector<analysis::StorageByteFragment>> fragments;
  for (int64_t index = 0; index < *count; ++index) {
    auto pieces = analysis::fieldByteFragments(relation, offset, index, declaration.getBitOrder(), declaration.getByteOrder());
    if (!pieces) return op.emitError("field-byte program has no complete storage-bit relation");
    fragments.push_back(std::move(*pieces));
  }
  b.setInsertionPoint(op);
  auto loc = op.getLoc();
  auto u8 = mlir::IntegerType::get(b.getContext(), 8, mlir::IntegerType::Unsigned);
  auto u64 = mlir::IntegerType::get(b.getContext(), 64, mlir::IntegerType::Unsigned);
  auto index = [&](int64_t value) -> mlir::Value { return mlir::arith::ConstantIndexOp::create(b, loc, value); };
  auto number = [&](int64_t value) -> mlir::Value { return riscv::ConstantOp::create(b, loc, u64, b.getIntegerAttr(u64, value)); };
  auto binary = [&](mlir::Value left, mlir::Value right, llvm::StringRef kind) -> mlir::Value {
    return riscv::BinaryOp::create(b, loc, u64, left, right, kind, riscv_internal::unselectedLeaf(b));
  };
  llvm::SmallVector<mlir::Value> coordinates;
  auto visit = [&](auto &&visit, unsigned dimension) -> mlir::LogicalResult {
    if (dimension < outerRank) {
      auto upper = riscv::ExtentOp::create(b, loc, b.getIndexType(), field.getOwner(), dimension);
      auto loop = mlir::scf::ForOp::create(b, loc, index(0), upper, index(1));
      riscv_internal::copyOrigin(op, loop);
      loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      mlir::OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      coordinates.push_back(loop.getInductionVar());
      auto result = visit(visit, dimension + 1);
      coordinates.pop_back();
      return result;
    }
    mlir::Value record = field.getOwner();
    if (outerRank) {
      auto recordType = riscv::MemDescType::get(b.getContext(), owner.getEncoding(),
          b.getDenseI64ArrayAttr({owner.getElements()}), b.getDenseI64ArrayAttr({owner.getAxisIds().asArrayRef().back()}),
          b.getDenseI64ArrayAttr({owner.getStrides().asArrayRef().back()}), b.getDenseI64ArrayAttr({owner.getOrigins().asArrayRef().back()}),
          owner.getAlignment(), "slice", owner.getAccess(), owner.getAliasSet(), owner.getLayoutIdentity(),
          owner.getStorageBits(), owner.getElements(), owner.getInterleaveRows());
      llvm::SmallVector<mlir::Attribute> selectors(outerRank, b.getStringAttr("index"));
      selectors.push_back(b.getStringAttr("all"));
      record = riscv::SliceOp::create(b, loc, recordType, field.getOwner(), coordinates, b.getArrayAttr(selectors));
    }
    struct Byte { mlir::Value value; int64_t mask = 0; };
    std::map<int64_t, Byte> bytes;
    // Compute the numerical bits before any byte update in this record. The
    // original Value, including an encoded read snapshot, remains the supplier.
    for (int64_t member = 0; member < *count; ++member) {
      mlir::Value value = op.getValue();
      if (valueType) {
        llvm::SmallVector<mlir::Value> selected(coordinates);
        llvm::SmallVector<int64_t> position(shape.size());
        int64_t remainder = member;
        for (size_t n = shape.size(); n > 0; --n) { position[n - 1] = remainder % shape[n - 1]; remainder /= shape[n - 1]; }
        for (int64_t component : position) selected.push_back(index(component));
        value = riscv::ExtractOp::create(b, loc, element, value, selected,
            b.getArrayAttr(llvm::SmallVector<mlir::Attribute>(selected.size(), b.getStringAttr("index"))),
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
      }
      if (mlir::isa<mlir::FloatType>(element)) {
        auto bits = mlir::IntegerType::get(b.getContext(), relation.elementBits, mlir::IntegerType::Unsigned);
        value = riscv::BitcastOp::create(b, loc, bits, value, riscv_internal::unselectedLeaf(b));
      }
      if (value.getType() != u64) value = riscv::CastOp::create(b, loc, u64, value, riscv_internal::unselectedLeaf(b));
      for (const auto &piece : fragments[member]) {
        auto &byte = bytes[piece.byte];
        int64_t mask = piece.mask << piece.shift;
        if (byte.mask & mask) return op.emitError("field-byte program contains overlapping source bit fragments");
        mlir::Value part = value;
        if (piece.resultShift) part = binary(part, number(piece.resultShift), "shr");
        part = binary(part, number(piece.mask), "and");
        if (piece.shift) part = binary(part, number(piece.shift), "shl");
        byte.value = byte.value ? binary(byte.value, part, "or") : part;
        byte.mask |= mask;
      }
    }
    for (auto &[offset, byte] : bytes) {
      auto value = riscv::CastOp::create(b, loc, u8, byte.value, riscv_internal::unselectedLeaf(b));
      auto stored = riscv::FieldByteStoreOp::create(b, loc, value, record, offset, byte.mask,
          spanOffset, spanBytes, chunk, riscv_internal::leaf(b, "scalar", "field-byte-store",
              "scalar.field-byte-store", "scalar.field-byte-store", 0, 0));
      riscv_internal::copyOrigin(op, stored);
    }
    return mlir::success();
  };
  if (mlir::failed(visit(visit, 0))) return mlir::failure();
  b.eraseOp(op);
  return mlir::success();
}
}

mlir::LogicalResult riscv_internal::materializeFieldStoreView(
    mlir::IRRewriter &rewriter, riscv::StoreOp operation) {
  llvm::SmallVector<mlir::Operation *> projections;
  mlir::Value root = operation.getRegion();
  while (mlir::isa_and_nonnull<riscv::SliceOp, riscv::SubviewOp>(root.getDefiningOp())) {
    projections.push_back(root.getDefiningOp());
    root = root.getDefiningOp()->getOperand(0);
  }
  auto field = root.getDefiningOp<riscv::FieldOp>();
  if (!field || projections.empty()) return mlir::success();
  auto owner = mlir::dyn_cast<riscv::MemDescType>(field.getOwner().getType());
  auto type = mlir::cast<riscv::MemDescType>(field.getType());
  auto facts = fieldFacts(field);
  if (!owner || owner.getShape().size() != 1 || type.getShape().size() != 1 ||
      owner.getShape()[0] != owner.getElements() || type.getShape()[0] <= 0 ||
      facts.mapping != "natural" || facts.bitOffset % 8 ||
      owner.getInterleaveRows() || type.getStorageBits() % 8 ||
      type.getStorageBits() < 8 || type.getStorageBits() > 64 ||
      facts.alignment < type.getStorageBits() / 8 || owner.getElements() % type.getShape()[0])
    return operation.emitError("projected field store requires an aligned natural record-local field view");
  // Early memory planning precedes the lowering of Level point coordinates.
  mlir::Value recordBase;
  mlir::Value cursor = field.getOwner();
  while (auto slice = cursor.getDefiningOp<riscv::SliceOp>()) {
    size_t index = 0;
    for (auto [axis, selector] : llvm::zip(slice.getBase().getType().getAxisIds().asArrayRef(), slice.getSelectors())) {
      auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
      if (kind == "all") continue;
      mlir::Value selected = slice.getIndices()[index++];
      if (axis == type.getAxisIds()[0] && kind == "domain") {
        auto point = selected.getDefiningOp<riscv::PhysicalPointOp>();
        if (!point) return mlir::success();
        recordBase = point.getBase();
        break;
      }
    }
    if (recordBase) break;
    cursor = slice.getBase();
  }
  for (mlir::Value base = field.getOwner();;) {
    if (base.getDefiningOp<riscv::SubviewOp>())
      return operation.emitError("projected field store does not support a subview of the encoded record owner");
    auto slice = base.getDefiningOp<riscv::SliceOp>();
    if (!slice) break;
    base = slice.getBase();
  }
  rewriter.setInsertionPoint(operation);
  if (!field.getOwner().getDefiningOp<riscv::SliceOp>()) {
    auto sliceType = riscv::MemDescType::get(rewriter.getContext(), owner.getEncoding(), owner.getShape(),
        owner.getAxisIds(), owner.getStrides(), owner.getOrigins(), owner.getAlignment(), "slice",
        owner.getAccess(), owner.getAliasSet(), owner.getLayoutIdentity(), owner.getStorageBits(),
        owner.getElements(), owner.getInterleaveRows());
    auto record = riscv::SliceOp::create(rewriter, operation.getLoc(), sliceType, field.getOwner(),
        mlir::ValueRange{}, rewriter.getArrayAttr({rewriter.getStringAttr("all")}));
    auto projected = riscv::FieldOp::create(rewriter, field.getLoc(), type, record.getResult(),
        field.getName(), field.getAccess(), field.getLeaf());
    copyOrigin(field, projected);
    root = projected.getResult();
  }
  mlir::Value origin = recordBase;
  if (!origin) origin = mlir::arith::ConstantIndexOp::create(rewriter, operation.getLoc(), 0);
  else if (owner.getElements() != type.getShape()[0]) {
    auto divisor = mlir::arith::ConstantIndexOp::create(rewriter, operation.getLoc(), owner.getElements() / type.getShape()[0]);
    origin = mlir::arith::DivUIOp::create(rewriter, operation.getLoc(), origin, divisor);
  }
  int64_t alignment = std::min(facts.alignment, type.getStorageBits() / 8);
  auto denseType = riscv::MemDescType::get(rewriter.getContext(), type.getEncoding(), type.getShape(),
      type.getAxisIds(), rewriter.getDenseI64ArrayAttr({1}), rewriter.getDenseI64ArrayAttr({-1}),
      alignment, "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
      type.getStorageBits(), 1, 0);
  mlir::Value region = riscv::FieldViewOp::create(rewriter, operation.getLoc(), denseType, root, origin);
  copyOrigin(operation, region.getDefiningOp());
  for (auto projection : llvm::reverse(projections)) {
    auto prior = mlir::cast<riscv::MemDescType>(projection->getResult(0).getType());
    auto result = riscv::MemDescType::get(rewriter.getContext(), prior.getEncoding(), prior.getShape(),
        prior.getAxisIds(), rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(prior.getShape().size(), 1)),
        rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(prior.getShape().size(), -1)),
        alignment, "slice", prior.getAccess(), prior.getAliasSet(), prior.getLayoutIdentity(),
        prior.getStorageBits(), 1, 0);
    if (auto slice = mlir::dyn_cast<riscv::SliceOp>(projection))
      region = riscv::SliceOp::create(rewriter, slice.getLoc(), result, region,
          slice.getIndices(), slice.getSelectors());
    else {
      auto subview = mlir::cast<riscv::SubviewOp>(projection);
      region = riscv::SubviewOp::create(rewriter, subview.getLoc(), result, region,
          subview.getDynamicBounds(), subview.getOffsetsAttr(), subview.getExtentsAttr());
    }
    copyOrigin(projection, region.getDefiningOp());
  }
  operation.getRegionMutable().assign(region);
  return mlir::success();
}

mlir::LogicalResult riscv_internal::planFieldStore(
    mlir::IRRewriter &rewriter, riscv::StoreOp operation) {
  auto field = operation.getRegion().getDefiningOp<riscv::FieldOp>();
  auto facts = fieldFacts(field);
  auto owner = mlir::dyn_cast<kernel::EncodingType>(logicalElement(field.getOwner().getType()));
  if (!owner)
    return operation.emitError("encoded field store requires a byte-addressable storage field");
  if (facts.bitOffset % 8 || facts.storageBits % 8) return materializeFieldBytes(rewriter, operation);
  const int64_t rows = interleaveRows(field, owner);
  const int64_t bits = logicalBitWidth(operation.getValue().getType());
  auto value = mlir::dyn_cast<riscv::ValueType>(operation.getValue().getType());
  auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
  llvm::StringRef kind;
  int64_t storeBits = bits, lanes = 1, sourcesPerStore = 1, temporaries = 0;
  llvm::SmallVector<int64_t> offsets, sources, shifts;
  const bool scalar = !value || value.getLayout().getCarrier() == "scalar";
  if (scalar) {
    if (!facts.scalarPerRecord || facts.mapping != "natural" ||
        (bits != 8 && bits != 16 && bits != 32))
      return materializeFieldBytes(rewriter, operation);
    kind = "scalar";
    if (rows > 1 && facts.chunkBytes < bits / 8) {
      kind = "scalar_bytes";
      storeBits = 8;
      for (int64_t byte = 0; byte < bits / 8; ++byte) {
        offsets.push_back(byte);
        sources.push_back(0);
        shifts.push_back(byte * 8);
      }
    } else {
      offsets.push_back(0);
      sources.push_back(0);
      shifts.push_back(0);
    }
  } else {
    if (value.getLayout().getCarrier() != "rvv" || value.getShape().size() != 1 ||
        value.getShape()[0] <= 0 || facts.logicalRank != 1 ||
        value.getShape()[0] * bits != facts.storageBits)
      return materializeFieldBytes(rewriter, operation);
    const int64_t count = value.getShape()[0];
    if (facts.mapping == "natural" && (bits == 8 || bits == 16 || bits == 32)) {
      if (rows > 1 && facts.chunkBytes < bits / 8)
        return materializeFieldBytes(rewriter, operation);
      kind = "natural";
      lanes = rows > 1 ? std::gcd(count, facts.chunkBytes / (bits / 8))
                       : std::min(count, value.getLayout().getVl());
    } else if (facts.mapping == "grouped_layered" &&
               (bits == 1 || bits == 2 || bits == 4) && facts.layer > 0 &&
               facts.group == facts.layer * (8 / bits) && count % facts.group == 0) {
      kind = "layered";
      storeBits = 8;
      sourcesPerStore = 8 / bits;
      lanes = rows > 1 ? std::gcd(facts.layer, facts.chunkBytes) : facts.layer;
    } else {
      return materializeFieldBytes(rewriter, operation);
    }
    const int64_t sew = value.getLayout().getSew();
    auto carrier = [&](int64_t candidate) {
      return riscv::LayoutAttr::get(
          rewriter.getContext(), "rvv", value.getAxisIds(),
          rewriter.getDenseI64ArrayAttr({count / lanes}),
          rewriter.getDenseI64ArrayAttr({lanes}),
          rewriter.getDenseI64ArrayAttr({1}), rewriter.getDenseI64ArrayAttr({1}),
          rewriter.getDenseI64ArrayAttr({1}), sew, candidate, lanes,
          (candidate + 7) / 8, value.getLayout().getValidity());
    };
    int64_t lmul = 0;
    while (lanes > 0 && !lmul) {
      for (int64_t candidate : target.getLegalLMULEighths().asArrayRef())
        if (candidate * target.getVlenBits() >= lanes * sew * 8 &&
            count % lanes == 0 && riscv::supportsRVVLayout(target, carrier(candidate)) &&
            (!lmul || candidate < lmul))
          lmul = candidate;
      if (!lmul)
        lanes /= 2;
    }
    if (!lanes || count % lanes || (kind == "layered" && facts.layer % lanes))
      return operation.emitError("field store lane partition has no legal RVV carrier");
    const int64_t groups = (lmul + 7) / 8;
    auto layout = carrier(lmul);
    if (!riscv::supportsRVVLayout(target, layout))
      return operation.emitError("field store selected an unsupported RVV layout");
    auto selected = riscv::ValueType::get(rewriter.getContext(), value.getElementType(),
                                         value.getShape(), value.getAxisIds(), layout);
    if (selected != value) {
      rewriter.setInsertionPoint(operation);
      auto conversionOp = riscv::ConvertLayoutOp::create(rewriter,
          operation.getLoc(), selected, operation.getValue(),
          layoutConversion(rewriter, value.getLayout(), layout),
          riscv::AccessAttr(), unselectedLeaf(rewriter));
      copyOrigin(operation, conversionOp);
      operation.getValueMutable().assign(conversionOp.getResult());
      value = selected;
    }
    if (kind == "natural") {
      for (int64_t part = 0; part < count / lanes; ++part) {
        offsets.push_back(part * lanes * (bits / 8));
        sources.push_back(part);
        shifts.push_back(0);
      }
    } else {
      temporaries = 2 * groups;
      for (int64_t group = 0; group < count / facts.group; ++group)
        for (int64_t stripe = 0; stripe < facts.layer; stripe += lanes) {
          offsets.push_back(group * facts.layer + stripe);
          for (int64_t layer = 0; layer < sourcesPerStore; ++layer) {
            sources.push_back((group * facts.group + layer * facts.layer + stripe) / lanes);
            shifts.push_back((facts.order == "lo_first" ? layer
                                                       : sourcesPerStore - 1 - layer) * bits);
          }
        }
    }
  }
  auto access = riscv::AccessAttr::get(
      rewriter.getContext(), "unit", facts.mapping, facts.alignment, 0, 0,
      facts.group, facts.layer, facts.joinFields, facts.joinLowBits, facts.joinRole,
      facts.bitOffset, facts.storageBits, facts.order, facts.chunkBytes);
  operation.setAccessAttr(access);
  operation.setFieldStorePlanAttr(riscv::FieldStorePlanAttr::get(
      rewriter.getContext(), kind, bits, storeBits, lanes, sourcesPerStore,
      rewriter.getDenseI64ArrayAttr(offsets), rewriter.getDenseI64ArrayAttr(sources),
      rewriter.getDenseI64ArrayAttr(shifts)));
  operation.setLeafAttr(leaf(
      rewriter, scalar ? "scalar" : "rvv", "encoded-field-store",
      scalar ? "scalar.field-store" : "rvv.field-store",
      scalar ? "scalar.field-store" : "rvv.field-store",
      0, 0, temporaries, 0, "none", "exact"));
  return mlir::success();
}
