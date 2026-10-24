#include "Weft/Target/RISCVPasses.h"
#include "Weft/Analysis/LogicalAxes.h"
#include "Weft/Analysis/StorageBytes.h"
#include "LocalMemory.h"
#include "NumericalValidity.h"
#include "RISCVPhysicalSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

namespace {
struct RecordValue {
  riscv::LoadOp root;
  Value bytes, valid;
  SmallVector<riscv::ReshapeOp> reshapes;
  int64_t elements, recordBytes;
};

class EncodedValues {
  ModuleOp module;
  IRRewriter b;
  int64_t identity, nextAxis = 0;
  DenseMap<Value, RecordValue> records;
  DenseMap<Value, Value> replacements, masks;
  SmallVector<Operation *> removed;
public:
  explicit EncodedValues(ModuleOp module) : module(module), b(module.getContext()), identity(lastLocalIdentity(module)) {
    module.walk([&](Operation *op) {
      auto account = [&](Type type) {
        for (int64_t axis : logicalAxes(type)) nextAxis = std::max(nextAxis, axis + 1);
        if (auto point = dyn_cast<riscv::PointType>(type)) nextAxis = std::max(nextAxis, point.getDomain().getAxisId() + 1);
        if (auto domain = dyn_cast<riscv::DomainType>(type)) nextAxis = std::max(nextAxis, domain.getAxisId() + 1);
      };
      for (Type type : op->getOperandTypes()) account(type);
      for (Type type : op->getResultTypes()) account(type);
    });
  }

  Value index(Location loc, int64_t n) { return arith::ConstantIndexOp::create(b, loc, n); }
  Value number(Location loc, Type type, int64_t n) { return riscv::ConstantOp::create(b, loc, type, b.getIntegerAttr(type, n)); }
  riscv::ValueType numeric(Type element, ArrayRef<int64_t> shape, ArrayRef<int64_t> axes) {
    return riscv::ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape), b.getDenseI64ArrayAttr(axes), unassignedLayout(b, axes));
  }
  Value scalarAddress(Value view, ArrayRef<Value> coordinates, Location loc) {
    auto type = cast<riscv::MemDescType>(view.getType());
    auto empty = b.getDenseI64ArrayAttr({});
    auto selected = riscv::MemDescType::get(b.getContext(), type.getEncoding(), empty, empty, empty, empty,
        type.getAlignment(), "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(), type.getStorageBits(), 1, 0);
    return riscv::SliceOp::create(b, loc, selected, view, coordinates,
        b.getArrayAttr(SmallVector<Attribute>(coordinates.size(), b.getStringAttr("index"))));
  }
  Value read(Value view, ArrayRef<Value> coordinates, Type element, Location loc) {
    return riscv::LoadOp::create(b, loc, element, scalarAddress(view, coordinates, loc), Value(), unassignedAccess(b), unselectedLeaf(b));
  }
  void write(Value value, Value view, ArrayRef<Value> coordinates, Location loc) {
    riscv::StoreOp::create(b, loc, value, scalarAddress(view, coordinates, loc), unassignedAccess(b), unselectedLeaf(b), riscv::FieldStorePlanAttr());
  }
  Value linear(ArrayRef<Value> coordinates, ArrayRef<int64_t> shape, Location loc) {
    Value result = index(loc, 0);
    for (auto [coordinate, extent] : llvm::zip(coordinates, shape))
      result = b.createOrFold<arith::AddIOp>(loc, b.createOrFold<arith::MulIOp>(loc, result, index(loc, extent)), coordinate);
    return result;
  }
  SmallVector<Value> coordinates(Value linear, ArrayRef<int64_t> shape, Location loc) {
    SmallVector<Value> result(shape.size());
    for (size_t i = shape.size(); i > 0; --i) {
      result[i - 1] = b.createOrFold<arith::RemUIOp>(loc, linear, index(loc, shape[i - 1]));
      linear = b.createOrFold<arith::DivUIOp>(loc, linear, index(loc, shape[i - 1]));
    }
    return result;
  }
  Value sourceRecord(const RecordValue &record, Value outputRecord, Location loc) {
    Value position = b.createOrFold<arith::MulIOp>(loc, outputRecord, index(loc, record.elements));
    for (auto reshape : llvm::reverse(record.reshapes)) {
      auto input = reshape.getInput().getType();
      auto order = *analysis::axisPermutation(input.getAxisIds().asArrayRef(), reshape.getOrder());
      SmallVector<Value> point(order.size());
      for (unsigned axis : llvm::reverse(order)) {
        Value extent = index(loc, input.getShape()[axis]);
        point[axis] = b.createOrFold<arith::RemUIOp>(loc, position, extent);
        position = b.createOrFold<arith::DivUIOp>(loc, position, extent);
      }
      position = linear(point, input.getShape().asArrayRef(), loc);
    }
    return b.createOrFold<arith::DivUIOp>(loc, position, index(loc, record.elements));
  }
  // Project a complete record through explicit descriptor coordinates. No
  // pointer arithmetic, traversal or record permutation is left to the emitter.
  Value recordRegion(Value view, ArrayRef<Value> point, int64_t elements, Location loc) {
    auto source = cast<riscv::MemDescType>(view.getType());
    unsigned last = source.getShape().size() - 1;
    auto tail = [&](DenseI64ArrayAttr values) { return b.getDenseI64ArrayAttr({values[last]}); };
    auto selected = riscv::MemDescType::get(b.getContext(), source.getEncoding(), b.getDenseI64ArrayAttr({source.getShape()[last]}),
        tail(source.getAxisIds()), tail(source.getStrides()), tail(source.getOrigins()), source.getAlignment(), "slice",
        source.getAccess(), source.getAliasSet(), source.getLayoutIdentity(), source.getStorageBits(), elements, 0);
    SmallVector<Attribute> selectors(last, b.getStringAttr("index"));
    selectors.push_back(b.getStringAttr("all"));
    Value row = riscv::SliceOp::create(b, loc, selected, view, point.drop_back(), b.getArrayAttr(selectors));
    auto window = riscv::MemDescType::get(b.getContext(), source.getEncoding(), b.getDenseI64ArrayAttr({elements}),
        selected.getAxisIds(), selected.getStrides(), selected.getOrigins(), source.getAlignment(), "slice",
        source.getAccess(), source.getAliasSet(), source.getLayoutIdentity(), source.getStorageBits(), elements, 0);
    Value offset = b.createOrFold<arith::MulIOp>(loc, point.back(), index(loc, elements));
    return riscv::SubviewOp::create(b, loc, window, row, ValueRange{offset}, b.getDenseI64ArrayAttr({-1}), b.getDenseI64ArrayAttr({elements}));
  }
  SmallVector<int64_t> recordShape(riscv::ValueType type, int64_t elements) {
    SmallVector<int64_t> result(type.getShape().asArrayRef());
    result.back() /= elements;
    return result;
  }
  FailureOr<Value> within(Value view, ArrayRef<Value> point, int64_t elements, Operation *anchor) {
    auto loc = anchor->getLoc();
    SmallVector<Value> first(point);
    first.back() = b.createOrFold<arith::MulIOp>(loc, first.back(), index(loc, elements));
    auto begin = memoryContains(b, view, first, anchor);
    first.back() = b.createOrFold<arith::AddIOp>(loc, first.back(), index(loc, elements - 1));
    auto end = memoryContains(b, view, first, anchor);
    if (failed(begin) || failed(end)) return failure();
    return b.createOrFold<arith::AndIOp>(loc, *begin, *end);
  }
  LogicalResult forRecords(riscv::ValueType type, int64_t elements, Operation *anchor,
                           function_ref<LogicalResult(Value, ArrayRef<Value>)> body) {
    auto shape = recordShape(type, elements);
    auto count = staticProduct(shape);
    if (!count || *count <= 0) return anchor->emitError("encoded program requires a bounded complete record domain");
    auto loc = anchor->getLoc();
    auto loop = scf::ForOp::create(b, loc, index(loc, 0), index(loc, *count), index(loc, 1));
    loop->setAttr("weft.riscv.direction", b.getStringAttr("ascending"));
    loop->setAttr("weft.riscv.system_unroll", b.getStringAttr("disable"));
    copyOrigin(anchor, loop);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto point = coordinates(loop.getInductionVar(), shape, loc);
    return body(loop.getInductionVar(), point);
  }
  FailureOr<RecordValue> snapshot(riscv::LoadOp load) {
    auto type = cast<riscv::ValueType>(load.getResult().getType());
    auto memory = load.getRegion().getType();
    auto encoding = cast<kernel::EncodingType>(memory.getEncoding());
    auto count = staticProduct(type.getShape().asArrayRef());
    if (encoding.getKind() != "base" || !count || *count <= 0 || memory.getElements() <= 0 ||
        type.getShape().empty() || type.getShape().asArrayRef().back() % memory.getElements() || memory.getStorageBits() % 8)
      return load.emitError("encoded reshape snapshot requires a bounded domain of complete base records"), failure();
    int64_t records = *count / memory.getElements(), bytes = memory.getStorageBits() / 8;
    if (bytes <= 0 || records > load->getParentOfType<riscv::KernelOp>().getTarget().getMaxPrivateStackBytes() / bytes)
      return load.emitError("encoded reshape snapshot exceeds target local storage"), failure();
    b.setInsertionPoint(load);
    auto u8 = IntegerType::get(b.getContext(), 8, IntegerType::Unsigned);
    auto payloadType = numeric(u8, {records, bytes}, {nextAxis++, nextAxis++});
    auto validType = numeric(b.getI1Type(), {records}, {nextAxis++});
    auto payload = allocateLocalValue(b, payloadType, load, identity);
    auto validity = allocateLocalValue(b, validType, load, identity);
    if (failed(payload) || failed(validity)) return failure();
    auto data = localView(b, *payload, load), mask = localView(b, *validity, load);
    if (failed(data) || failed(mask)) return failure();
    if (failed(forRecords(type, memory.getElements(), load, [&](Value record, ArrayRef<Value> point) {
      auto loc = load.getLoc();
      auto bounds = within(load.getRegion(), point, memory.getElements(), load);
      if (failed(bounds)) return failure();
      Value valid = *bounds;
      write(valid, *mask, {record}, loc);
      auto branch = scf::IfOp::create(b, loc, valid, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      Value region = recordRegion(load.getRegion(), point, memory.getElements(), loc);
      for (int64_t byte = 0; byte < bytes; ++byte) {
        Value value = riscv::FieldByteLoadOp::create(b, loc, u8, region, byte, 0, bytes, 1,
            leaf(b, "scalar", "field-byte-load", "scalar.field-byte-load", "scalar.field-byte-load", 0, 0));
        write(value, *data, {record, index(loc, byte)}, loc);
      }
      return success();
    }))) return failure();
    return RecordValue{load, *payload, *validity, {}, memory.getElements(), bytes};
  }
  LogicalResult store(riscv::StoreOp op, const RecordValue &record) {
    auto type = cast<riscv::ValueType>(op.getValue().getType());
    auto memory = op.getRegion().getType();
    if (memory.getEncoding() != type.getElementType() || memory.getInterleaveRows() ||
        memory.getShape() != type.getShape() || memory.getAxisIds() != type.getAxisIds())
      return op.emitError("encoded record transfer requires matching base layout and logical domain");
    b.setInsertionPoint(op);
    auto bytes = localView(b, record.bytes, op), mask = localView(b, record.valid, op);
    if (failed(bytes) || failed(mask)) return failure();
    return forRecords(type, record.elements, op, [&](Value linear, ArrayRef<Value> point) {
      auto loc = op.getLoc();
      Value source = sourceRecord(record, linear, loc);
      Value valid = read(*mask, {source}, b.getI1Type(), loc);
      auto bounds = within(op.getRegion(), point, record.elements, op);
      if (failed(bounds)) return failure();
      valid = b.createOrFold<arith::AndIOp>(loc, valid, *bounds);
      auto branch = scf::IfOp::create(b, loc, valid, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      Value region = recordRegion(op.getRegion(), point, record.elements, loc);
      for (int64_t byte = 0; byte < record.recordBytes; ++byte) {
        Value value = read(*bytes, {source, index(loc, byte)}, IntegerType::get(b.getContext(), 8, IntegerType::Unsigned), loc);
        riscv::FieldByteStoreOp::create(b, loc, value, region, byte, 255, 0, record.recordBytes, 1,
            leaf(b, "scalar", "field-byte-store", "scalar.field-byte-store", "scalar.field-byte-store", 0, 0));
      }
      return success();
    });
  }
  LogicalResult field(riscv::FieldOp op, const RecordValue &record) {
    auto owner = cast<riscv::ValueType>(op.getOwner().getType());
    auto type = dyn_cast<riscv::ValueType>(op.getResult().getType());
    if (!type || owner.getShape().asArrayRef().back() != record.elements)
      return op.emitError("encoded field program requires explicit complete-record prefix axes");
    auto encoding = cast<kernel::EncodingType>(owner.getElementType());
    auto declaration = findEncoding(op, baseEncodingFamily(op, encoding));
    unsigned ordinal = 0;
    for (; ordinal < declaration.getFieldNames().size(); ++ordinal)
      if (cast<StringAttr>(declaration.getFieldNames()[ordinal]).getValue() == op.getName()) break;
    if (ordinal == declaration.getFieldNames().size()) return op.emitError("encoded field is absent from its declaration");
    auto element = cast<TypeAttr>(declaration.getFieldTypes()[ordinal]).getValue();
    auto relation = analysis::fieldBitLayout(element, cast<ArrayAttr>(declaration.getFieldLayouts()[ordinal]));
    auto shape = cast<DenseI64ArrayAttr>(declaration.getFieldShapes()[ordinal]).asArrayRef();
    auto count = staticProduct(shape);
    if (!count || *count <= 0 || relation.elementBits > 64) return op.emitError("encoded field has no bounded numerical domain");
    SmallVector<SmallVector<analysis::StorageByteFragment>> fragments;
    for (int64_t i = 0; i < *count; ++i) {
      auto pieces = analysis::fieldByteFragments(relation, declaration.getFieldBitOffsets()[ordinal], i,
          declaration.getBitOrder(), declaration.getByteOrder());
      if (!pieces) return op.emitError("encoded field has no complete storage-bit relation");
      fragments.push_back(std::move(*pieces));
    }
    b.setInsertionPoint(op);
    auto output = allocateLocalValue(b, type, op, identity);
    SmallVector<int64_t> privateAxes;
    for (size_t i = 0; i < type.getShape().size(); ++i) privateAxes.push_back(nextAxis++);
    auto byteType = IntegerType::get(b.getContext(), 8, IntegerType::Unsigned);
    auto maskType = numeric(byteType, type.getShape().asArrayRef(), privateAxes);
    auto active = allocateLocalValue(b, maskType, op, identity);
    if (failed(output) || failed(active)) return failure();
    auto loc = op.getLoc();
    Value zero = riscv::ConstantOp::create(b, loc, element,
        isa<FloatType>(element) ? Attribute(b.getFloatAttr(element, 0)) : Attribute(b.getIntegerAttr(element, 0)));
    if (failed(initializeLocalValue(b, *output, zero, op)) ||
        failed(initializeLocalValue(b, *active, number(loc, byteType, 0), op))) return failure();
    auto data = localView(b, *output, op), predicate = localView(b, *active, op);
    auto bytes = localView(b, record.bytes, op), validity = localView(b, record.valid, op);
    if (failed(data) || failed(predicate) || failed(bytes) || failed(validity)) return failure();
    auto machine = IntegerType::get(b.getContext(), relation.elementBits <= 32 ? 32 : 64, IntegerType::Unsigned);
    if (failed(forRecords(owner, record.elements, op, [&](Value linear, ArrayRef<Value> point) {
      Value source = sourceRecord(record, linear, loc);
      Value valid = read(*validity, {source}, b.getI1Type(), loc);
      auto branch = scf::IfOp::create(b, loc, valid, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      std::map<int64_t, Value> loaded;
      auto binary = [&](Value a, Value c, StringRef kind) -> Value {
        return riscv::BinaryOp::create(b, loc, machine, a, c, kind, unselectedLeaf(b));
      };
      for (int64_t member = 0; member < *count; ++member) {
        Value bits;
        for (const auto &fragment : fragments[member]) {
          auto &byte = loaded[fragment.byte];
          if (!byte) byte = read(*bytes, {source, index(loc, fragment.byte)}, IntegerType::get(b.getContext(), 8, IntegerType::Unsigned), loc);
          Value part = riscv::CastOp::create(b, loc, machine, byte, unselectedLeaf(b));
          if (fragment.shift) part = binary(part, number(loc, machine, fragment.shift), "shr");
          part = binary(part, number(loc, machine, fragment.mask), "and");
          if (fragment.resultShift) part = binary(part, number(loc, machine, fragment.resultShift), "shl");
          bits = bits ? binary(bits, part, "or") : part;
        }
        Value value;
        if (isa<FloatType>(element)) {
          Type integer = IntegerType::get(b.getContext(), relation.elementBits, IntegerType::Unsigned);
          if (integer != bits.getType()) bits = riscv::CastOp::create(b, loc, integer, bits, unselectedLeaf(b));
          value = riscv::BitcastOp::create(b, loc, element, bits, unselectedLeaf(b));
        } else value = riscv::CastOp::create(b, loc, element, bits, unselectedLeaf(b));
        SmallVector<Value> position(point.drop_back());
        auto local = coordinates(index(loc, member), shape, loc);
        position.append(local);
        write(value, *data, position, loc);
        write(number(loc, byteType, 1), *predicate, position, loc);
      }
      return success();
    }))) return failure();
    // Private predicates have byte storage. Convert their complete initialized
    // contents to numeric Boolean values before common mask propagation; never
    // reinterpret a byte array as RVV's packed predicate representation.
    Value maskBytes = riscv::LoadOp::create(b, loc, maskType, completeLocalSlice(b, *predicate),
        Value(), unassignedAccess(b), unselectedLeaf(b));
    Value maskValue = riscv::CompareOp::create(b, loc,
        numeric(b.getI1Type(), type.getShape().asArrayRef(), privateAxes), maskBytes,
        number(loc, byteType, 0), "ne", unselectedLeaf(b));
    auto publicMask = numeric(b.getI1Type(), type.getShape().asArrayRef(), type.getAxisIds().asArrayRef());
    masks[op.getResult()] = riscv::ReshapeOp::create(b, loc, publicMask, maskValue,
        maskType.getAxisIds(), DenseI64ArrayAttr());
    replacements[op.getResult()] = riscv::LoadOp::create(b, loc, type, completeLocalSlice(b, *data),
        Value(), unassignedAccess(b), unselectedLeaf(b));
    return success();
  }
  LogicalResult run() {
    SmallVector<Operation *> original;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) { original.push_back(op); });
    llvm::SmallPtrSet<Operation *, 32> selected;
    for (Operation *op : original) {
      auto load = dyn_cast<riscv::LoadOp>(op);
      auto type = load ? dyn_cast<riscv::ValueType>(load.getResult().getType()) : riscv::ValueType();
      if (!type || !isa<kernel::EncodingType>(type.getElementType()) ||
          !llvm::any_of(load.getResult().getUsers(), [](Operation *user) { return isa<riscv::ReshapeOp>(user); })) continue;
      SmallVector<Value> pending{load.getResult()};
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        for (Operation *user : value.getUsers()) {
          if (!selected.insert(user).second) continue;
          if (auto reshape = dyn_cast<riscv::ReshapeOp>(user)) pending.push_back(reshape.getResult());
          else if (!isa<riscv::FieldOp, riscv::StoreOp>(user))
            return user->emitError("encoded reshape consumer needs an explicit record program");
        }
      }
      auto saved = snapshot(load);
      if (failed(saved)) return failure();
      records[load.getResult()] = *saved;
      removed.push_back(load);
    }
    for (Operation *op : original) {
      if (!selected.contains(op)) continue;
      if (auto reshape = dyn_cast<riscv::ReshapeOp>(op)) {
        auto before = reshape.getInput().getType(), after = reshape.getResult().getType();
        auto found = records.find(reshape.getInput());
        if (found == records.end()) return reshape.emitError("encoded reshape lost its source snapshot");
        auto order = analysis::axisPermutation(before.getAxisIds().asArrayRef(), reshape.getOrder());
        auto n = staticProduct(before.getShape().asArrayRef()), m = staticProduct(after.getShape().asArrayRef());
        int64_t elements = found->second.elements;
        if (!order || order->empty() || order->back() + 1 != before.getShape().size() ||
            !n || !m || *n <= 0 || n != m || after.getShape().empty() ||
            before.getShape().asArrayRef().back() % elements || after.getShape().asArrayRef().back() % elements)
          return reshape.emitError("encoded reshape must permute complete records, preserving their internal logical elements");
        auto next = found->second;
        next.reshapes.push_back(reshape);
        records[reshape.getResult()] = std::move(next);
      } else if (auto write = dyn_cast<riscv::StoreOp>(op)) {
        auto found = records.find(write.getValue());
        if (found == records.end() || failed(store(write, found->second))) return failure();
      } else if (auto read = dyn_cast<riscv::FieldOp>(op)) {
        auto found = records.find(read.getOwner());
        if (found == records.end() || failed(field(read, found->second))) return failure();
      }
      removed.push_back(op);
    }
    if (failed(propagateNumericValidity(b, original, masks, nextAxis))) return failure();
    for (auto [source, replacement] : replacements) source.replaceAllUsesWith(replacement);
    for (Operation *op : llvm::reverse(removed)) {
      if (!op->use_empty()) return op->emitError("encoded program retains an unresolved consumer");
      b.eraseOp(op);
    }
    return success();
  }
};

class MaterializeRISCVEncodedValuesPass : public PassWrapper<MaterializeRISCVEncodedValuesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeRISCVEncodedValuesPass)
  StringRef getArgument() const override { return "weft-riscv-materialize-encoded-values"; }
  StringRef getDescription() const override { return "Materialize encoded record snapshots, coordinate maps and valid transfers"; }
  void runOnOperation() override { if (failed(EncodedValues(getOperation()).run())) signalPassFailure(); }
};
}

std::unique_ptr<Pass> weft::createMaterializeRISCVEncodedValuesPass() {
  return std::make_unique<MaterializeRISCVEncodedValuesPass>();
}
