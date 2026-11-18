#include "Transforms/Memory/RowSnapshots.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/TypeQueries.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

std::pair<Value, Value> weft::x86::loadRowSnapshot(
    OpBuilder &b, Location loc, VectorType flat, int64_t columns, Value offset,
    Value guard, ValueRange strides, ValueRange limits,
    llvm::function_ref<Value(Value, Value)> loadPacket) {
  const int64_t count = flat.getNumElements();
  auto indices = VectorType::get({columns}, b.getIndexType());
  auto predicates = indices.clone(b.getI1Type());
  auto maskBytes = indices.clone(b.getI8Type());
  auto splat = [&](Type type, Value value) {
    return b.createOrFold<vector::BroadcastOp>(loc, type, value);
  };
  SmallVector<APInt> lanes;
  for (int64_t lane = 0; lane < columns; ++lane)
    lanes.emplace_back(64, lane);
  Value positions = arith::ConstantOp::create(b, loc, indices, DenseIntElementsAttr::get(indices, lanes));
  Value columnOffsets = b.createOrFold<arith::MulIOp>(loc, positions, splat(indices, strides[1]));
  Value validColumns = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, positions, splat(indices, limits[1]));
  Value payload = memref::AllocOp::create(b, loc, MemRefType::get({count}, flat.getElementType()));
  Value validity = memref::AllocOp::create(b, loc, MemRefType::get({count}, b.getI8Type()));
  auto index = [&](int64_t n) { return arith::ConstantIndexOp::create(b, loc, n); };
  Value zero = index(0), width = index(columns);
  auto loop = scf::ForOp::create(b, loc, zero, index(count / columns), index(1));
  {
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(loop.getBody());
    Value row = loop.getInductionVar();
    Value rowStart = b.createOrFold<arith::AddIOp>(loc, offset, b.createOrFold<arith::MulIOp>(loc, row, strides[0]));
    Value addresses = b.createOrFold<arith::AddIOp>(loc, splat(indices, rowStart), columnOffsets);
    Value validRow = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row, limits[0]);
    validRow = b.createOrFold<arith::AndIOp>(loc, guard, validRow);
    Value mask = b.createOrFold<arith::AndIOp>(loc, splat(predicates, validRow), validColumns);
    Value packet = loadPacket(b.createOrFold<arith::IndexCastOp>(loc, indices.clone(b.getI64Type()), addresses), mask);
    Value destination = b.createOrFold<arith::MulIOp>(loc, row, width);
    vector::StoreOp::create(b, loc, packet, payload, destination);
    vector::StoreOp::create(b, loc, b.createOrFold<arith::ExtUIOp>(loc, maskBytes, mask), validity, destination);
  }
  Value result = vector::LoadOp::create(b, loc, flat, payload, zero);
  Value bytes = vector::LoadOp::create(b, loc, flat.clone(b.getI8Type()), validity, zero);
  return {result, b.createOrFold<arith::TruncIOp>(loc, flat.clone(b.getI1Type()), bytes)};
}

std::optional<std::pair<Value, Value>> weft::x86::updateRowWindow(OpBuilder &b,
    weft::kernel::UpdateOp update, Value indices, Value indexMask,
    Value owner, Value ownerMask, Value values, Value valid) {
  Location loc = update.getLoc();
  auto domain = update.getInput().getType(), patch = update.getValue().getType();
  auto axes = weft::kernel::logicalAxes(domain);
  auto dimensions = weft::kernel::logicalShape(domain);
  auto patchShape = weft::kernel::logicalShape(patch);
  auto selectors = update.getSelectors().getValue();
  if (axes.empty() || update.getIndices().size() != 1 || selectors.size() != axes.size() ||
      axes != weft::kernel::logicalAxes(patch) || dimensions.size() != patchShape.size() ||
      dimensions.drop_back() != patchShape.drop_back() ||
      cast<StringAttr>(selectors.back()).getValue() != "gather" ||
      !llvm::all_of(selectors.drop_back(), [](Attribute s) { return cast<StringAttr>(s).getValue() == "all"; }))
    return {};
  Value sourceIndices = update.getIndices().front();
  if (weft::kernel::logicalAxes(sourceIndices.getType()) != axes.take_back()) return {};
  int64_t columns = dimensions.back();
  auto offsets = dyn_cast<VectorType>(indices.getType());
  auto tile = dyn_cast<VectorType>(owner.getType());
  auto incoming = dyn_cast<VectorType>(values.getType());
  if (!offsets || !tile || !incoming || tile.getElementType().isInteger(1)) return {};
  int64_t width = offsets.getNumElements(), rows = tile.getNumElements() / columns;
  if (width < 4 || width > columns || incoming.getNumElements() != rows * width) return {};
  // An iota plus a uniform offset is injective even when unsigned addition
  // wraps, provided a complete period is at least as wide as the packet.
  Value ramp = sourceIndices;
  if (auto add = ramp.getDefiningOp<weft::kernel::BinaryOp>(); add && add.getKind() == "add") {
    if (!isa<weft::kernel::ValueType>(add.getLhs().getType())) ramp = add.getRhs();
    else if (!isa<weft::kernel::ValueType>(add.getRhs().getType())) ramp = add.getLhs();
  }
  auto iota = ramp.getDefiningOp<weft::kernel::IotaOp>();
  if (!iota || ramp.getType() != sourceIndices.getType() || offsets.getElementTypeBitWidth() > 64 ||
      offsets.getElementTypeBitWidth() < llvm::Log2_64_Ceil(columns + 1))
    return {};
  auto index = [&](int64_t n) { return arith::ConstantIndexOp::create(b, loc, n); };
  Value zero = index(0);
  auto bytes = tile.clone(b.getI8Type());
  Value buffer = memref::AllocOp::create(b, loc, MemRefType::get(tile.getShape(), tile.getElementType()));
  Value validity = memref::AllocOp::create(b, loc, MemRefType::get(tile.getShape(), b.getI8Type()));
  vector::StoreOp::create(b, loc, owner, buffer, zero);
  vector::StoreOp::create(b, loc, b.createOrFold<arith::ExtUIOp>(loc, bytes, ownerMask), validity, zero);
  Value relative = b.createOrFold<arith::IndexCastUIOp>(loc, offsets.clone(b.getIndexType()), indices);
  Value first = vector::ExtractOp::create(b, loc, relative, ArrayRef<int64_t>{0});
  Value fits = b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, first, index(columns - width));
  Value end = vector::BroadcastOp::create(b, loc, relative.getType(), index(columns));
  Value inRange = b.createOrFold<arith::AndIOp>(loc, indexMask,
      b.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, relative, end));
  for (int64_t row = 0; row < rows; ++row) {
    auto packet = [&](Value flat) -> Value {
      return vector::ExtractStridedSliceOp::create(b, loc, flat, ArrayRef<int64_t>{row * width},
          ArrayRef<int64_t>{width}, ArrayRef<int64_t>{1});
    };
    Value payload = packet(values);
    Value live = b.createOrFold<arith::ExtUIOp>(loc, offsets.clone(b.getI8Type()), packet(valid));
    Value start = index(row * columns);
    auto branch = scf::IfOp::create(b, loc, TypeRange{}, fits, true);
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(branch.thenBlock());
    Value dest = b.createOrFold<arith::AddIOp>(loc, start, first);
    vector::MaskedStoreOp::create(b, loc, buffer, ValueRange{dest}, indexMask, payload);
    vector::MaskedStoreOp::create(b, loc, validity, ValueRange{dest}, indexMask, live);
    b.setInsertionPointToStart(branch.elseBlock());
    // A partially out-of-range or wrapped window can still select lanes at
    // either end. Use the original coordinates and predicate for those lanes.
    vector::ScatterOp::create(b, loc, Type{}, buffer, ValueRange{start}, relative, inRange, payload);
    vector::ScatterOp::create(b, loc, Type{}, validity, ValueRange{start}, relative, inRange, live);
  }
  Value result = vector::LoadOp::create(b, loc, tile, buffer, zero);
  Value mask = vector::LoadOp::create(b, loc, bytes, validity, zero);
  return std::make_pair(result, b.createOrFold<arith::TruncIOp>(loc, ownerMask.getType(), mask));
}
