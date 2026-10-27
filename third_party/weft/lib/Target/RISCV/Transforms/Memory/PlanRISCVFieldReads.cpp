#include "Transforms/Memory/FieldReads.h"
#include "Analysis/LocalMemory.h"
#include "Transforms/Representation/NumericalValidity.h"
#include "Transforms/Support/RISCVPhysicalSupport.h"
#include "Weft/Analysis/StorageBytes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <map>

using namespace mlir;
using namespace weft;

namespace {
// An encoded load fixes the read epoch. Stage changes sharing/lifetime, not
// that epoch. Only shape-preserving wrappers can be crossed here; projections
// and control-flow values must first expose their own physical storage plan.
riscv::LoadOp encodedReadEpoch(Value owner) {
  while (true) {
    if (auto stage = owner.getDefiningOp<riscv::MaterializeOp>()) owner = stage.getInput();
    else if (auto stage = owner.getDefiningOp<riscv::RegisterMaterializeOp>()) owner = stage.getInput();
    else return owner.getDefiningOp<riscv::LoadOp>();
  }
}

bool hasOnlyFragmentSupplyUses(Value field) {
  SmallVector<Value> pending{field};
  llvm::DenseSet<Value> visited;
  bool observed = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second) continue;
    for (Operation *user : value.getUsers()) {
      if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(user);
          conversion && conversion.getConversion().getEffect() == "pure") {
        pending.push_back(conversion.getResult());
      } else if (auto projection = dyn_cast<riscv::ExtractOp>(user)) {
        pending.push_back(projection.getResult());
      } else if (isa<riscv::IMEFragmentStorageLoadOp,
                     riscv::IMEFragmentChunkSplitLoadOp,
                     riscv::IMEFragmentWindowLoadOp>(user)) {
        observed = true;
      } else if (auto implementation = user->getAttrOfType<riscv::ImplementationAttr>("implementation");
                 isa<riscv::ContractOp, riscv::OuterContractOp>(user) && implementation &&
                 implementation.getEngine() == "ime") {
        observed = true;
      } else {
        return false;
      }
    }
  }
  return observed;
}

FailureOr<Value> readField(IRRewriter &b, riscv::FieldOp field, Value region,
                           Operation *epoch, Type result, int64_t &identity) {
  auto owner = dyn_cast<riscv::MemDescType>(region.getType());
  auto encodedValue = dyn_cast<riscv::ValueType>(region.getType());
  auto encoding = cast<kernel::EncodingType>(owner ? owner.getEncoding() : encodedValue.getElementType());
  auto declaration = riscv_internal::findEncoding(field, riscv_internal::baseEncodingFamily(field, encoding));
  if (!declaration) return field.emitError("field byte read has no Encoding declaration"), failure();
  unsigned ordinal = 0;
  for (; ordinal < declaration.getFieldNames().size(); ++ordinal)
    if (cast<StringAttr>(declaration.getFieldNames()[ordinal]).getValue() == field.getName()) break;
  if (ordinal == declaration.getFieldNames().size()) return failure();
  auto element = cast<TypeAttr>(declaration.getFieldTypes()[ordinal]).getValue();
  auto relation = analysis::fieldBitLayout(element, cast<ArrayAttr>(declaration.getFieldLayouts()[ordinal]));
  auto shape = cast<DenseI64ArrayAttr>(declaration.getFieldShapes()[ordinal]).asArrayRef();
  auto count = riscv_internal::staticProduct(shape);
  auto shaped = dyn_cast<riscv::ValueType>(result);
  auto ownerShape = riscv_internal::logicalShape(region.getType());
  auto ownerAxes = riscv_internal::logicalAxes(region.getType());
  unsigned outerRank = ownerShape.size() - 1;
  if (!count || *count <= 0 || relation.elementBits <= 0 || relation.elementBits > 64 ||
      (shaped && shaped.getShape().size() != outerRank + shape.size()) ||
      (!shaped && (outerRank || !shape.empty())))
    return field.emitError("field byte read requires the complete bounded field projection"), failure();
  int64_t offset = declaration.getFieldBitOffsets()[ordinal];
  int64_t spanOffset = offset / 8;
  int64_t spanBytes = (offset % 8 + declaration.getFieldStorageBits()[ordinal] + 7) / 8;
  int64_t chunk = riscv_internal::fieldFacts(field).chunkBytes;
  SmallVector<SmallVector<analysis::StorageByteFragment>> fragments;
  for (int64_t member = 0; member < *count; ++member) {
    auto pieces = analysis::fieldByteFragments(relation, offset, member, declaration.getBitOrder(), declaration.getByteOrder());
    if (!pieces) return field.emitError("field byte read has no complete storage-bit relation"), failure();
    fragments.push_back(std::move(*pieces));
  }
  b.setInsertionPoint(epoch);
  auto loc = field.getLoc();
  SmallVector<Value> activeExtents;
  if (encodedValue) {
    auto extents = riscv_internal::activeNumericExtents(b, region);
    if (failed(extents)) return field.emitError("saved field projection requires its original active record coordinates"), failure();
    activeExtents = *extents;
  }
  auto u8 = IntegerType::get(b.getContext(), 8, IntegerType::Unsigned);
  auto machine = IntegerType::get(b.getContext(), relation.elementBits <= 32 ? 32 : 64, IntegerType::Unsigned);
  auto index = [&](int64_t n) -> Value { return arith::ConstantIndexOp::create(b, loc, n); };
  auto number = [&](int64_t n) -> Value { return riscv::ConstantOp::create(b, loc, machine, b.getIntegerAttr(machine, n)); };
  auto binary = [&](Value left, Value right, StringRef kind) -> Value {
    return riscv::BinaryOp::create(b, loc, machine, left, right, kind, riscv_internal::unselectedLeaf(b));
  };
  Value storage, output, scalar;
  Operation *birth = field.getOwner().getDefiningOp();
  bool staged = isa_and_nonnull<riscv::MaterializeOp, riscv::RegisterMaterializeOp>(birth);
  bool sameScope = staged && birth->getBlock() == epoch->getBlock();
  if (shaped) {
    auto allocated = riscv_internal::allocateLocalValue(b, shaped, sameScope ? birth : epoch, identity, sameScope);
    if (failed(allocated)) return failure();
    storage = *allocated;
    auto view = riscv_internal::localView(b, storage, epoch);
    if (failed(view)) return failure();
    output = *view;
    // Full physical capacity is initialized; absent rows never issue a read.
    Attribute zero = isa<FloatType>(element) ? Attribute(b.getFloatAttr(element, 0)) : Attribute(b.getIntegerAttr(element, 0));
    Value initial = riscv::ConstantOp::create(b, loc, element, zero);
    if (failed(riscv_internal::initializeLocalValue(b, storage, initial, epoch))) return failure();
  }
  SmallVector<Value> coordinates;
  auto visit = [&](auto &&visit, unsigned dimension) -> LogicalResult {
    if (dimension < outerRank) {
      Value upper = owner ? Value(riscv::ExtentOp::create(b, loc, b.getIndexType(), region, dimension)) : activeExtents[dimension];
      auto loop = scf::ForOp::create(b, loc, index(0), upper, index(1));
      riscv_internal::copyOrigin(field, loop);
      loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      coordinates.push_back(loop.getInductionVar());
      auto status = visit(visit, dimension + 1);
      coordinates.pop_back();
      return status;
    }
    Value record = region;
    if (outerRank) {
      SmallVector<Attribute> selectors(outerRank, b.getStringAttr("index"));
      selectors.push_back(b.getStringAttr("all"));
      if (owner) {
        auto type = riscv::MemDescType::get(b.getContext(), owner.getEncoding(),
            b.getDenseI64ArrayAttr({owner.getElements()}), b.getDenseI64ArrayAttr({ownerAxes.back()}),
            b.getDenseI64ArrayAttr({owner.getStrides().asArrayRef().back()}), b.getDenseI64ArrayAttr({owner.getOrigins().asArrayRef().back()}),
            owner.getAlignment(), "slice", owner.getAccess(), owner.getAliasSet(), owner.getLayoutIdentity(),
            owner.getStorageBits(), owner.getElements(), owner.getInterleaveRows());
        record = riscv::SliceOp::create(b, loc, type, region, coordinates, b.getArrayAttr(selectors));
      } else {
        auto axis = b.getDenseI64ArrayAttr({ownerAxes.back()});
        auto layout = encodedValue.getLayout();
        auto last = [&](DenseI64ArrayAttr factors) { return b.getDenseI64ArrayAttr({factors.asArrayRef().back()}); };
        auto projected = riscv::LayoutAttr::get(b.getContext(), layout.getCarrier(), axis,
            last(layout.getTimeFactors()), last(layout.getLaneFactors()), last(layout.getReplicaFactors()),
            last(layout.getFragmentFactors()), last(layout.getLocalFactors()), layout.getSew(),
            layout.getLmulEighths(), layout.getVl(), layout.getRegisterGroups(), layout.getValidity());
        auto type = riscv::ValueType::get(b.getContext(), encoding, b.getDenseI64ArrayAttr({static_cast<int64_t>(declaration.getElements())}),
            axis, projected);
        record = riscv::ExtractOp::create(b, loc, type, region, coordinates, b.getArrayAttr(selectors),
            riscv_internal::unassignedAccess(b), riscv_internal::unselectedLeaf(b));
      }
    }
    std::map<int64_t, Value> bytes;
    for (int64_t member = 0; member < *count; ++member) {
      Value bits;
      for (const auto &fragment : fragments[member]) {
        Value &byte = bytes[fragment.byte];
        if (!byte) byte = riscv::FieldByteLoadOp::create(b, loc, u8, record, fragment.byte,
            spanOffset, spanBytes, chunk, riscv_internal::leaf(b, "scalar", "field-byte-load", "scalar.field-byte-load", "scalar.field-byte-load", 0, 0));
        Value part = riscv::CastOp::create(b, loc, machine, byte, riscv_internal::unselectedLeaf(b));
        if (fragment.shift) part = binary(part, number(fragment.shift), "shr");
        part = binary(part, number(fragment.mask), "and");
        if (fragment.resultShift) part = binary(part, number(fragment.resultShift), "shl");
        bits = bits ? binary(bits, part, "or") : part;
      }
      Value value;
      if (isa<FloatType>(element)) {
        Type integer = IntegerType::get(b.getContext(), relation.elementBits, IntegerType::Unsigned);
        if (bits.getType() != integer) bits = riscv::CastOp::create(b, loc, integer, bits, riscv_internal::unselectedLeaf(b));
        value = riscv::BitcastOp::create(b, loc, element, bits, riscv_internal::unselectedLeaf(b));
      } else value = riscv::CastOp::create(b, loc, element, bits, riscv_internal::unselectedLeaf(b));
      if (!shaped) { scalar = value; continue; }
      SmallVector<Value> selected(coordinates);
      SmallVector<int64_t> point(shape.size());
      int64_t remainder = member;
      for (size_t n = shape.size(); n > 0; --n) { point[n - 1] = remainder % shape[n - 1]; remainder /= shape[n - 1]; }
      for (int64_t component : point) selected.push_back(index(component));
      auto full = cast<riscv::MemDescType>(output.getType());
      auto empty = b.getDenseI64ArrayAttr({});
      auto type = riscv::MemDescType::get(b.getContext(), full.getEncoding(), empty, empty, empty, empty,
          full.getAlignment(), "slice", full.getAccess(), full.getAliasSet(), full.getLayoutIdentity(), full.getStorageBits(), 1, 0);
      auto destination = riscv::SliceOp::create(b, loc, type, output, selected,
          b.getArrayAttr(SmallVector<Attribute>(selected.size(), b.getStringAttr("index"))));
      riscv::StoreOp::create(b, loc, value, destination, riscv_internal::unassignedAccess(b),
          riscv_internal::unselectedLeaf(b), riscv::FieldStorePlanAttr());
    }
    return success();
  };
  if (failed(visit(visit, 0))) return failure();
  if (staged && !sameScope && shaped) {
    // A parent read and a child stage have separate lifetime boundaries. Keep
    // the original snapshot, then initialize the child's owned representation
    // at its actual birth instead of moving that birth into the parent.
    b.setInsertionPoint(birth);
    auto copy = riscv_internal::allocateLocalValue(b, shaped, birth, identity, true);
    if (failed(copy) || failed(riscv_internal::initializeLocalValue(b, *copy, storage, birth))) return failure();
    storage = *copy;
  } else if (staged && !shaped) {
    b.setInsertionPoint(birth);
    scalar = riscv::RegisterMaterializeOp::create(b, loc, scalar.getType(), scalar,
        birth->getAttrOfType<IntegerAttr>("owner_domain_id").getInt(),
        birth->getAttrOfType<IntegerAttr>("birth_id").getInt(),
        birth->getAttrOfType<IntegerAttr>("lifetime_end_domain_id").getInt(), "share");
  }
  return shaped ? storage : scalar;
}
}

LogicalResult riscv_internal::materializeFieldReads(ModuleOp module, IRRewriter &b) {
  SmallVector<riscv::FieldOp> fields;
  module.walk([&](riscv::FieldOp field) {
    fields.push_back(field);
  });
  llvm::DenseMap<std::pair<Value, StringAttr>, Value> saved;
  int64_t identity = lastLocalIdentity(module);
  for (auto field : fields) {
    // Preserve Encoding geometry until the selected fragment materializer has
    // had the opportunity to form its exact field transfer. If it needs a
    // numerical pack instead, that pack remains a numeric consumer and the
    // subsequent memory pass completes the ordinary byte snapshot here.
    if (hasOnlyFragmentSupplyUses(field.getResult())) continue;
    auto value = dyn_cast<riscv::ValueType>(field.getResult().getType());
    if (isa<riscv::MemDescType>(field.getResult().getType())) {
      auto owner = dyn_cast<riscv::MemDescType>(field.getOwner().getType());
      auto facts = fieldFacts(field);
      if (!owner || owner.getShape().empty()) continue;
      int64_t bits = cast<riscv::MemDescType>(field.getResult().getType()).getStorageBits();
      if (!((facts.mapping == "natural" && (bits % 8 || facts.alignment < bits / 8)) ||
            (owner.getShape().size() > 1 && facts.mapping == "joined"))) continue;
      SmallVector<riscv::LoadOp> readers;
      for (auto *user : field.getResult().getUsers())
        if (auto load = dyn_cast<riscv::LoadOp>(user)) readers.push_back(load);
      for (auto load : readers) {
        if (owner.getShape().asArrayRef().back() != owner.getElements())
          return load.emitError("field byte read requires one complete record along the encoded element axis");
        auto read = readField(b, field, field.getOwner(), load, load.getResult().getType(), identity);
        if (failed(read)) return failure();
        b.setInsertionPoint(load);
        Value result = *read;
        if (result.getType() != load.getResult().getType()) {
          auto actual = cast<riscv::ValueType>(result.getType());
          auto expected = cast<riscv::ValueType>(load.getResult().getType());
          result = riscv::ConvertLayoutOp::create(b, load.getLoc(), expected, result,
              layoutConversion(b, actual.getLayout(), expected.getLayout()), riscv::AccessAttr(), unselectedLeaf(b));
        }
        b.replaceOp(load, result);
      }
      continue;
    }
    auto load = encodedReadEpoch(field.getOwner());
    if (!load) {
      auto owner = dyn_cast<riscv::ValueType>(field.getOwner().getType());
      auto facts = fieldFacts(field);
      int64_t bits = logicalBitWidth(value ? value.getElementType() : field.getResult().getType());
      auto encoding = owner ? dyn_cast<kernel::EncodingType>(owner.getElementType()) : kernel::EncodingType();
      // Native scalar reads and the selected interleaved/joined transfers
      // already realize their layouts. Close the base-record vector bit
      // projection here, without replacing those existing transfer programs.
      bool general = value && value.getLayout().getCarrier() != "scalar" &&
          facts.mapping == "natural" && bits % 8;
      if (!encoding || encoding.getKind() != "base" || !general) continue;
      auto declaration = findEncoding(field, baseEncodingFamily(field, encoding));
      if (!declaration || owner.getShape().empty() || owner.getShape().asArrayRef().back() != declaration.getElements())
        return field.emitError("saved field byte read requires one complete record along the encoded element axis");
      auto read = readField(b, field, field.getOwner(), field, field.getResult().getType(), identity);
      if (failed(read)) return failure();
      b.setInsertionPoint(field);
      Value result = *read;
      if (result.getType() != field.getResult().getType()) {
        auto actual = cast<riscv::ValueType>(result.getType());
        result = riscv::ConvertLayoutOp::create(b, field.getLoc(), value, result,
            layoutConversion(b, actual.getLayout(), value.getLayout()), riscv::AccessAttr(), unselectedLeaf(b));
      }
      b.replaceOp(field, result);
      continue;
    }
    if (load.getSnapshotStorage()) continue;
    auto owner = load.getRegion().getType();
    auto facts = fieldFacts(field);
    int64_t bits = logicalBitWidth(value ? value.getElementType() : field.getResult().getType());
    // Complete encoded snapshots, including multidimensional records, belong
    // to MaterializeRISCVReadSnapshots. Preserve the field edge so its normal
    // vector/fragment consumer reads that same saved record. Only layouts whose
    // byte projection is not covered by an existing transfer need expansion.
    // A local whole-byte field is a storage projection, not a numeric read.
    // Its consumers select explicit indexed-entry byte loads with their real
    // windows and alignment. An actual numeric field still needs its complete
    // byte assembly. Interfering writes on a deferred projection go through
    // the common encoded-read snapshot owner.
    // Scalar and scalar-replica field transfers already read natural words
    // through alignment-safe byte copies. Keep the encoded storage edge for
    // the shared snapshot owner instead of expanding these into local arrays.
    auto encoding = cast<kernel::EncodingType>(owner.getEncoding());
    auto declaration = findEncoding(field, baseEncodingFamily(field, encoding));
    bool naturalWord = declaration && declaration.getByteOrder() == "little" &&
        facts.bitOffset % 8 == 0 && (bits == 16 || bits == 32 || bits == 64);
    bool scalarWord = naturalWord &&
        (!value || value.getLayout().getCarrier() == "scalar");
    // A scalar field distributed across a declared interleave cohort already
    // has unit-byte and indexed-byte RVV transfers. Their selected byte loads
    // preserve the original alignment and avoid a scalar assembly buffer.
    bool cohortWord = naturalWord && bits <= 32 && facts.scalarPerRecord && value &&
        value.getLayout().getCarrier() == "rvv" && interleaveRows(field, encoding) > 0;
    bool numericUnaligned = facts.alignment < bits / 8 && !scalarWord && !cohortWord &&
        (!value || value.getLayout().getCarrier() != "local");
    bool general = (facts.mapping == "natural" && (bits % 8 || numericUnaligned)) ||
        (owner.getShape().size() > 1 && facts.mapping == "joined");
    if (!general) continue;
    if (owner.getShape().empty() || owner.getShape().asArrayRef().back() != owner.getElements())
      return field.emitError("field byte read requires one complete record along the encoded element axis");
    auto key = std::make_pair(field.getOwner(), field.getNameAttr());
    Value result = saved.lookup(key);
    if (!result) {
      auto read = readField(b, field, load.getRegion(), load, field.getResult().getType(), identity);
      if (failed(read)) return failure();
      result = *read;
      saved[key] = result;
    }
    b.setInsertionPoint(field);
    if (result.getType() != field.getResult().getType()) {
      auto actual = cast<riscv::ValueType>(result.getType());
      result = riscv::ConvertLayoutOp::create(b, field.getLoc(), value, result,
          layoutConversion(b, actual.getLayout(), value.getLayout()), riscv::AccessAttr(), unselectedLeaf(b));
    }
    b.replaceOp(field, result);
  }
  return success();
}
