#include "AMXContractions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
namespace {
Value index(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
Value integer(OpBuilder &b, Location loc, int64_t n, unsigned bits) {
  return arith::ConstantIntOp::create(b, loc, n, bits);
}
void intrinsic(ModuleOp module, OpBuilder &b, Location loc, StringRef name, ValueRange operands) {
  auto function = module.lookupSymbol<func::FuncOp>(name);
  if (!function) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    function = func::FuncOp::create(b, loc, name, b.getFunctionType(operands.getTypes(), {}));
    function.setPrivate();
  }
  func::CallOp::create(b, loc, function, operands);
}

// Keep the MemRef visible at the call site until ownership lowering has placed
// releases. Only this always-inline leaf exposes its borrowed pointer to LLVM.
void transfer(ModuleOp module, OpBuilder &b, Location loc, const weft::X86Target &target,
              Value storage, Value offset, int64_t strideBytes, unsigned tile, bool store,
              int64_t rows = 0, int64_t rowBytes = 0) {
  auto type = cast<MemRefType>(storage.getType());
  unsigned bytes = type.getElementTypeBitWidth() / 8;
  std::string name = "__weft_amx_" + std::string(store ? "store_" : "load_") +
      std::to_string(tile) + "_" + std::to_string(bytes);
  auto function = module.lookupSymbol<func::FuncOp>(name);
  if (!function) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    function = func::FuncOp::create(b, loc, name,
        b.getFunctionType({type, b.getIndexType(), b.getI64Type()}, {}));
    function.setPrivate();
    function->setAttr("weft.x86.cpu", b.getStringAttr(target.cpu));
    // The address is consumed by one synchronous transfer and is never retained.
    function.setArgAttr(0, "weft.borrowed", b.getUnitAttr());
    function->setAttr("weft.x86.vector_bits", b.getI32IntegerAttr(target.vectorBits));
    function->setAttr("weft.x86.amx_transfer", b.getUnitAttr());
    function->setAttr("weft.x86.helper", b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr(name)), b.getNamedAttr("noinline", b.getBoolAttr(false))}));
    auto *body = function.addEntryBlock();
    b.setInsertionPointToStart(body);
    Value address = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, body->getArgument(0));
    Value displacement = arith::MulIOp::create(b, loc, body->getArgument(1), index(b, loc, bytes));
    address = arith::AddIOp::create(b, loc, address, displacement);
    address = arith::IndexCastUIOp::create(b, loc, b.getI64Type(), address);
    Value pointer = LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(b.getContext()), address);
    intrinsic(module, b, loc, store ? "llvm.x86.tilestored64" : "llvm.x86.tileloadd64",
        ValueRange{integer(b, loc, tile, 8), pointer, body->getArgument(2)});
    func::ReturnOp::create(b, loc);
  }
  auto call = func::CallOp::create(b, loc, function, ValueRange{storage, offset, integer(b, loc, strideBytes, 64)});
  if (!store)
    call->setAttr("weft.x86.amx_load", b.getDenseI64ArrayAttr({rows, rowBytes, int64_t(tile)}));
}
}

Value weft::x86::guardAMXBFloat(OpBuilder &b, Location loc, const ArrayStorage &input,
                               int64_t count, int64_t terms) {
  Value safe = integer(b, loc, 1, 1);
  // Two bits of headroom make every partial sum finite. The lower bound makes
  // each product an integer multiple of the smallest normal F32 value.
  int64_t maximumExponent = (378 - llvm::Log2_64_Ceil(terms)) / 2;
  auto bitsType = VectorType::get({32}, b.getI16Type());
  auto payload = bitsType.clone(b.getBF16Type());
  auto validType = bitsType.clone(b.getI8Type());
  auto literal = [&](int64_t n) -> Value {
    return arith::ConstantOp::create(b, loc, bitsType,
        DenseElementsAttr::get(bitsType, b.getI16IntegerAttr(n)));
  };
  {
    auto loop = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, count), index(b, loc, 32), ValueRange{safe});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value data = vector::LoadOp::create(b, loc, payload, input.data, loop.getInductionVar());
      Value bits = arith::BitcastOp::create(b, loc, bitsType, data);
      Value magnitude = arith::AndIOp::create(b, loc, bits, literal(0x7fff));
      Value lower = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, magnitude, literal(71 * 128));
      Value upper = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, magnitude, literal((maximumExponent + 1) * 128));
      Value good = arith::AndIOp::create(b, loc, lower, upper);
      good = arith::OrIOp::create(b, loc, good,
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, magnitude, literal(0)));
      Value validity = vector::LoadOp::create(b, loc, validType, input.valid, loop.getInductionVar());
      validity = arith::TruncIOp::create(b, loc, bitsType.clone(b.getI1Type()), validity);
      good = arith::AndIOp::create(b, loc, good, validity);
      Value all = vector::ReductionOp::create(b, loc, vector::CombiningKind::AND, good);
      scf::YieldOp::create(b, loc, arith::AndIOp::create(b, loc, loop.getRegionIterArgs()[0], all).getResult());
    }
    safe = loop.getResult(0);
  }
  return safe;
}

static void materializeAMX(ModuleOp module, OpBuilder &b, Location loc,
                           const weft::X86Target &target, const weft::x86::ArrayStorage &left,
                           const weft::x86::ArrayStorage &right, const weft::x86::ArrayStorage &output,
                           int64_t rows, int64_t columns, int64_t terms,
                           bool integerDot, bool leftUnsigned, bool rightUnsigned, Value rightSnapshot,
                           bool tiledOutput = false, bool leftReductionFirst = false,
                           bool rightReductionFirst = false, bool maskInputs = false) {
  int64_t group = integerDot ? 4 : 2;
  int64_t bytesPerElement = integerDot ? 1 : 2;
  int64_t reductionTile = !integerDot ? 32 : terms % 64 == 0 ? 64 : terms % 32 == 0 ? 32 : 16;
  int64_t tileColumns = std::min<int64_t>(columns, 16);
  auto maskPayload = [&](Value payload, const weft::x86::ArrayStorage &source, Value offset) -> Value {
    if (!maskInputs) return payload;
    auto type = cast<VectorType>(payload.getType());
    Value bytes = vector::LoadOp::create(b, loc, type.clone(b.getI8Type()), source.valid, offset);
    Value mask = arith::TruncIOp::create(b, loc, type.clone(b.getI1Type()), bytes);
    Value zero = arith::ConstantOp::create(b, loc, type,
        DenseElementsAttr::get(type, b.getZeroAttr(type.getElementType())));
    return arith::SelectOp::create(b, loc, mask, payload, zero);
  };
  Value leftData = left.data;
  if (leftReductionFirst) {
    // Transpose the immutable snapshot in bounded register rectangles. Reads
    // stay at the contraction, and no author or externally aliased storage is
    // changed. Partial snapshots are zeroed within the same transpose.
    Type element = cast<MemRefType>(left.data.getType()).getElementType();
    auto allocation = memref::AllocOp::create(b, loc,
        MemRefType::get({ShapedType::kDynamic}, element), index(b, loc, rows * terms));
    allocation->setAttr("alignment", b.getI64IntegerAttr(64));
    allocation->setAttr("weft.x86.matrix_operand", b.getStringAttr("left-reduction-last"));
    leftData = allocation;
    int64_t rowPacket = std::gcd(rows, int64_t(8));
    int64_t termPacket = std::gcd(terms, int64_t(8));
    auto rowLoop = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, rows), index(b, loc, rowPacket));
    OpBuilder::InsertionGuard rowGuard(b);
    b.setInsertionPointToStart(rowLoop.getBody());
    auto termLoop = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, terms), index(b, loc, termPacket));
    b.setInsertionPointToStart(termLoop.getBody());
    Type bits = integerDot ? Type(b.getI8Type()) : Type(b.getI16Type());
    auto rectangleType = VectorType::get({rowPacket * termPacket}, bits);
    Value rectangle = arith::ConstantOp::create(b, loc, rectangleType,
        DenseElementsAttr::get(rectangleType, b.getIntegerAttr(bits, 0)));
    for (int64_t term = 0; term < termPacket; ++term) {
      Value k = arith::AddIOp::create(b, loc, termLoop.getInductionVar(), index(b, loc, term));
      Value offset = arith::AddIOp::create(b, loc,
          arith::MulIOp::create(b, loc, k, index(b, loc, rows)), rowLoop.getInductionVar());
      Value payload = vector::LoadOp::create(b, loc, VectorType::get({rowPacket}, element), left.data, offset);
      payload = maskPayload(payload, left, offset);
      if (!integerDot) payload = arith::BitcastOp::create(b, loc, VectorType::get({rowPacket}, bits), payload);
      rectangle = vector::InsertStridedSliceOp::create(b, loc, payload, rectangle,
          ArrayRef<int64_t>{term * rowPacket}, ArrayRef<int64_t>{1});
    }
    for (int64_t row = 0; row < rowPacket; ++row) {
      SmallVector<int64_t> order;
      for (int64_t term = 0; term < termPacket; ++term) order.push_back(term * rowPacket + row);
      Value payload = vector::ShuffleOp::create(b, loc, rectangle, rectangle, order);
      if (!integerDot) payload = arith::BitcastOp::create(b, loc, VectorType::get({termPacket}, element), payload);
      Value m = arith::AddIOp::create(b, loc, rowLoop.getInductionVar(), index(b, loc, row));
      Value offset = arith::AddIOp::create(b, loc,
          arith::MulIOp::create(b, loc, m, index(b, loc, terms)), termLoop.getInductionVar());
      vector::StoreOp::create(b, loc, payload, leftData, offset);
    }
  } else if (maskInputs) {
    Type element = cast<MemRefType>(left.data.getType()).getElementType();
    auto allocation = memref::AllocOp::create(b, loc,
        MemRefType::get({ShapedType::kDynamic}, element), index(b, loc, rows * terms));
    allocation->setAttr("alignment", b.getI64IntegerAttr(64));
    leftData = allocation;
    int64_t packet = std::gcd(rows * terms, int64_t(target.vectorBits / 8));
    auto loop = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, rows * terms), index(b, loc, packet));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value offset = loop.getInductionVar();
    Value payload = vector::LoadOp::create(b, loc, VectorType::get({packet}, element), left.data, offset);
    vector::StoreOp::create(b, loc, maskPayload(payload, left, offset), leftData, offset);
  }
  auto packedType = MemRefType::get({ShapedType::kDynamic}, b.getI32Type());
  auto allocation = memref::AllocOp::create(b, loc, packedType, index(b, loc, columns * terms / group));
  allocation->setAttr("alignment", b.getI64IntegerAttr(64));
  if (rightSnapshot) allocation->setAttr("weft.x86.amx_pack", b.getUnitAttr());
  Value packed = allocation;
  if (rightSnapshot) {
    // Project the saved SSA operand directly into tile rows. Materializing its
    // logical row order and immediately loading it back for this transpose
    // hides storage/decode permutations from packet projection and adds a
    // redundant private-memory round trip.
    auto wordsType = VectorType::get({columns * terms / group}, b.getI32Type());
    Value words = vector::BitCastOp::create(b, loc, wordsType, rightSnapshot);
    int64_t packet = target.vectorBits / 32, count = columns * terms / group;
    for (int64_t first = 0; first < count; first += packet) {
      SmallVector<int64_t> order;
      for (int64_t position = first; position < std::min(first + packet, count); ++position) {
        int64_t panel = position / (tileColumns * (terms / group));
        int64_t word = position / tileColumns % (terms / group);
        int64_t column = panel * tileColumns + position % tileColumns;
        order.push_back(column * (terms / group) + word);
      }
      Value selected = vector::ShuffleOp::create(b, loc, words, words, order);
      auto store = vector::StoreOp::create(b, loc, selected, packed, index(b, loc, first));
      store->setAttr("weft.x86.amx_pack", b.getUnitAttr());
    }
  } else if (rightReductionFirst) {
    // Source [K,N] is already contiguous along B's tile row. Interleave a
    // bounded group of K rows directly into the packed AMX row: a full [N,K]
    // intermediate would add a second memory round trip and longer residency.
    auto packing = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, terms), index(b, loc, group));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(packing.getBody());
    auto columnLoop = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, columns), index(b, loc, tileColumns));
    b.setInsertionPointToStart(columnLoop.getBody());
    Type bits = integerDot ? Type(b.getI8Type()) : Type(b.getI16Type());
    Type payloadType = cast<MemRefType>(right.data.getType()).getElementType();
    auto rectangleType = VectorType::get({tileColumns * group}, bits);
    Value rectangle = arith::ConstantOp::create(b, loc, rectangleType,
        DenseElementsAttr::get(rectangleType, b.getIntegerAttr(bits, 0)));
    for (int64_t member = 0; member < group; ++member) {
      Value k = arith::AddIOp::create(b, loc, packing.getInductionVar(), index(b, loc, member));
      Value offset = arith::AddIOp::create(b, loc,
          arith::MulIOp::create(b, loc, k, index(b, loc, columns)), columnLoop.getInductionVar());
      Value payload = vector::LoadOp::create(b, loc, VectorType::get({tileColumns}, payloadType), right.data, offset);
      payload = maskPayload(payload, right, offset);
      if (!integerDot) payload = arith::BitcastOp::create(b, loc, VectorType::get({tileColumns}, bits), payload);
      rectangle = vector::InsertStridedSliceOp::create(b, loc, payload, rectangle,
          ArrayRef<int64_t>{member * tileColumns}, ArrayRef<int64_t>{1});
    }
    SmallVector<int64_t> order;
    for (int64_t column = 0; column < tileColumns; ++column)
      for (int64_t member = 0; member < group; ++member) order.push_back(member * tileColumns + column);
    Value interleaved = vector::ShuffleOp::create(b, loc, rectangle, rectangle, order);
    Value words = vector::BitCastOp::create(b, loc, VectorType::get({tileColumns}, b.getI32Type()), interleaved);
    Value offset = arith::AddIOp::create(b, loc,
        arith::MulIOp::create(b, loc, columnLoop.getInductionVar(), index(b, loc, terms / group)),
        arith::MulIOp::create(b, loc,
            arith::DivUIOp::create(b, loc, packing.getInductionVar(), index(b, loc, group)), index(b, loc, tileColumns)));
    vector::StoreOp::create(b, loc, words, packed, offset);
    allocation->setAttr("weft.x86.matrix_operand", b.getStringAttr("right-reduction-first-to-packed"));
  } else {
    // A memory word holds adjacent K elements. Columns are contiguous in the
    // tile's VNNI row; all packing remains inside this contraction invocation.
    // Consume a contiguous cache-line fragment from each input row before
    // visiting the next K packet. A scalar column walk otherwise repeatedly
    // touches more same-set cache lines than the L1 can retain.
    int64_t pairsPerPacket = std::gcd(int64_t(8), terms / group);
    auto packing = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, terms / group), index(b, loc, pairsPerPacket));
    {
      OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(packing.getBody());
      for (int64_t first = 0; first < columns; first += 8) {
        int64_t width = std::min<int64_t>(8, columns - first);
        auto tileType = VectorType::get({width * pairsPerPacket}, b.getI32Type());
        Value tile = arith::ConstantOp::create(b, loc, tileType, DenseElementsAttr::get(tileType, b.getI32IntegerAttr(0)));
        for (int64_t column = 0; column < width; ++column) {
          Value source = arith::AddIOp::create(b, loc, index(b, loc, (first + column) * terms),
              arith::MulIOp::create(b, loc, packing.getInductionVar(), index(b, loc, group)));
          Type inputType = integerDot ? Type(b.getI8Type()) : Type(b.getBF16Type());
          Value pairs = vector::LoadOp::create(b, loc, VectorType::get({pairsPerPacket * group}, inputType), right.data, source);
          pairs = maskPayload(pairs, right, source);
          Value bits = integerDot ? pairs : Value(arith::BitcastOp::create(b, loc,
              VectorType::get({pairsPerPacket * group}, b.getI16Type()), pairs));
          Value words = vector::BitCastOp::create(b, loc, VectorType::get({pairsPerPacket}, b.getI32Type()), bits);
          tile = vector::InsertStridedSliceOp::create(b, loc, words, tile,
              ArrayRef<int64_t>{column * pairsPerPacket}, ArrayRef<int64_t>{1});
        }
        for (int64_t pair = 0; pair < pairsPerPacket; ++pair) {
          SmallVector<int64_t> indices;
          for (int64_t column = 0; column < width; ++column) indices.push_back(column * pairsPerPacket + pair);
          Value words = vector::ShuffleOp::create(b, loc, tile, tile, indices);
          Value row = arith::AddIOp::create(b, loc, packing.getInductionVar(), index(b, loc, pair));
          Value destination = arith::AddIOp::create(b, loc,
              index(b, loc, (first / tileColumns) * tileColumns * (terms / group) + first % tileColumns),
              arith::MulIOp::create(b, loc, row, index(b, loc, tileColumns)));
          vector::StoreOp::create(b, loc, words, packed, destination);
        }
      }
    }
  }
  int64_t tileRows = std::min<int64_t>(rows, 16);
  int64_t rowCopies = rows % 32 == 0 ? 2 : 1;
  int64_t columnCopies = columns % 32 == 0 ? 2 : 1;
  // Two A tiles, two B tiles and four accumulators fill the eight-tile file.
  // Smaller domains use only their legal rectangle, without padded reads.
  int64_t firstB = rowCopies, firstC = rowCopies + columnCopies;
  std::string name = "__weft_amx_config_" + std::to_string(tileRows) + "_" +
      std::to_string(tileColumns) + "_" + std::to_string(rowCopies) + "_" + std::to_string(columnCopies) +
      "_" + std::to_string(reductionTile * bytesPerElement);
  auto configuration = module.lookupSymbol<LLVM::GlobalOp>(name);
  if (!configuration) {
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToEnd(module.getBody());
    SmallVector<APInt> bytes(64, APInt(8, 0));
    bytes[0] = APInt(8, 1);
    auto configure = [&](int64_t tile, int64_t height, int64_t rowBytes) {
      bytes[16 + tile * 2] = APInt(8, rowBytes);
      bytes[48 + tile] = APInt(8, height);
    };
    for (int64_t row = 0; row < rowCopies; ++row) configure(row, tileRows, reductionTile * bytesPerElement);
    for (int64_t column = 0; column < columnCopies; ++column) configure(firstB + column, reductionTile / group, tileColumns * 4);
    for (int64_t result = 0; result < rowCopies * columnCopies; ++result)
      configure(firstC + result, tileRows, tileColumns * 4);
    auto type = VectorType::get({64}, b.getI8Type());
    configuration = LLVM::GlobalOp::create(b, loc, type, true, LLVM::Linkage::Private, name,
        DenseIntElementsAttr::get(type, bytes), 64);
  }
  intrinsic(module, b, loc, "llvm.x86.ldtilecfg", ValueRange{LLVM::AddressOfOp::create(b, loc, configuration)});
  auto rowTiles = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, rows), index(b, loc, tileRows * rowCopies));
  {
    OpBuilder::InsertionGuard rowGuard(b); b.setInsertionPointToStart(rowTiles.getBody());
    auto columnTiles = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, columns), index(b, loc, tileColumns * columnCopies));
    OpBuilder::InsertionGuard columnGuard(b); b.setInsertionPointToStart(columnTiles.getBody());
    for (int64_t result = 0; result < rowCopies * columnCopies; ++result)
      intrinsic(module, b, loc, "llvm.x86.tilezero", ValueRange{integer(b, loc, firstC + result, 8)});
    auto reduction = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, terms), index(b, loc, reductionTile));
    reduction->setAttr("weft.x86.amx_contraction", b.getDictionaryAttr({
        b.getNamedAttr("rows", b.getI64IntegerAttr(tileRows * rowCopies)),
        b.getNamedAttr("columns", b.getI64IntegerAttr(tileColumns * columnCopies)),
        b.getNamedAttr("terms", b.getI64IntegerAttr(terms)),
        b.getNamedAttr("tiles", b.getI64IntegerAttr(firstC + rowCopies * columnCopies))}));
    {
      OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(reduction.getBody());
      for (int64_t row = 0; row < rowCopies; ++row) {
        Value coordinate = arith::AddIOp::create(b, loc, rowTiles.getInductionVar(), index(b, loc, row * tileRows));
        Value offset = arith::AddIOp::create(b, loc, reduction.getInductionVar(),
            arith::MulIOp::create(b, loc, coordinate, index(b, loc, terms)));
        transfer(module, b, loc, target, leftData, offset, terms * bytesPerElement, row, false,
                 tileRows, reductionTile * bytesPerElement);
      }
      for (int64_t column = 0; column < columnCopies; ++column) {
        Value offset = arith::MulIOp::create(b, loc,
            arith::DivUIOp::create(b, loc, reduction.getInductionVar(), index(b, loc, group)), index(b, loc, tileColumns));
        Value first = arith::AddIOp::create(b, loc, columnTiles.getInductionVar(), index(b, loc, column * tileColumns));
        offset = arith::AddIOp::create(b, loc, offset,
            arith::MulIOp::create(b, loc, first, index(b, loc, terms / group)));
        transfer(module, b, loc, target, packed, offset, tileColumns * 4, firstB + column, false,
                 reductionTile / group, tileColumns * 4);
      }
      for (int64_t row = 0; row < rowCopies; ++row)
        for (int64_t column = 0; column < columnCopies; ++column)
          intrinsic(module, b, loc, !integerDot ? "llvm.x86.tdpbf16ps" :
              leftUnsigned ? (rightUnsigned ? "llvm.x86.tdpbuud" : "llvm.x86.tdpbusd") :
                             (rightUnsigned ? "llvm.x86.tdpbsud" : "llvm.x86.tdpbssd"), ValueRange{
              integer(b, loc, firstC + row * columnCopies + column, 8),
              integer(b, loc, row, 8), integer(b, loc, firstB + column, 8)});
    }
    for (int64_t row = 0; row < rowCopies; ++row)
      for (int64_t column = 0; column < columnCopies; ++column) {
        Value coordinate = arith::AddIOp::create(b, loc, rowTiles.getInductionVar(), index(b, loc, row * tileRows));
        Value offset;
        if (tiledOutput) {
          // Fixed SSA consumers can project the result from native tile order.
          // Keep each tile contiguous instead of interleaving its cache lines
          // with the neighboring tile's store.
          Value first = arith::AddIOp::create(b, loc, columnTiles.getInductionVar(),
                                            index(b, loc, column * tileColumns));
          offset = arith::AddIOp::create(b, loc,
              arith::MulIOp::create(b, loc, coordinate, index(b, loc, columns)),
              arith::MulIOp::create(b, loc, first, index(b, loc, tileRows)));
        } else {
          offset = arith::MulIOp::create(b, loc, coordinate, index(b, loc, columns));
          offset = arith::AddIOp::create(b, loc, offset, columnTiles.getInductionVar());
          offset = arith::AddIOp::create(b, loc, offset, index(b, loc, column * tileColumns));
        }
        transfer(module, b, loc, target, output.data, offset, (tiledOutput ? tileColumns : columns) * 4,
            firstC + row * columnCopies + column, true);
      }
  }
  intrinsic(module, b, loc, "llvm.x86.tilerelease", {});
  if (maskInputs) return;
  auto valid = scf::ForOp::create(b, loc, index(b, loc, 0), index(b, loc, rows * columns), index(b, loc, 1));
  {
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(valid.getBody());
    memref::StoreOp::create(b, loc, integer(b, loc, 1, 8), output.valid, valid.getInductionVar());
  }
}

void weft::x86::materializeAMXBFloat(ModuleOp module, OpBuilder &b, Location loc,
                                    const X86Target &target, const ArrayStorage &left,
                                    const ArrayStorage &right, const ArrayStorage &output,
                                    int64_t rows, int64_t columns, int64_t terms,
                                    bool leftReductionFirst, bool rightReductionFirst) {
  materializeAMX(module, b, loc, target, left, right, output, rows, columns, terms, false, false, false, {},
                 false, leftReductionFirst, rightReductionFirst);
}

void weft::x86::materializeAMXInteger(ModuleOp module, OpBuilder &b, Location loc,
                                    const X86Target &target, const ArrayStorage &left,
                                    const ArrayStorage &right, const ArrayStorage &output,
                                    int64_t rows, int64_t columns, int64_t terms,
                                    bool leftUnsigned, bool rightUnsigned,
                                    bool leftReductionFirst, bool rightReductionFirst, bool maskInputs) {
  materializeAMX(module, b, loc, target, left, right, output, rows, columns, terms, true, leftUnsigned, rightUnsigned, {},
                 false, leftReductionFirst, rightReductionFirst, maskInputs);
}

Value weft::x86::materializeAMXIntegerFixed(ModuleOp module, OpBuilder &b, Location loc,
                                         const X86Target &target, const ArrayStorage &left,
                                         Value right, const ArrayStorage &output,
                                         int64_t rows, int64_t columns, int64_t terms,
                                         bool leftUnsigned, bool rightUnsigned, bool tiledOutput,
                                         bool leftReductionFirst, bool maskInputs) {
  materializeAMX(module, b, loc, target, left, {}, output, rows, columns, terms,
                 true, leftUnsigned, rightUnsigned, right, tiledOutput, leftReductionFirst, false, maskInputs);
  auto type = VectorType::get({rows * columns}, b.getI32Type());
  Value result = vector::LoadOp::create(b, loc, type, output.data, index(b, loc, 0));
  if (!tiledOutput) return result;
  int64_t tileRows = std::min<int64_t>(rows, 16), tileColumns = std::min<int64_t>(columns, 16);
  SmallVector<int64_t> order;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column)
      order.push_back((row / tileRows) * tileRows * columns +
                      (column / tileColumns) * tileRows * tileColumns +
                      (row % tileRows) * tileColumns + column % tileColumns);
  return vector::ShuffleOp::create(b, loc, result, result, order);
}

namespace {
class ReuseAMXConfiguration : public PassWrapper<ReuseAMXConfiguration, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseAMXConfiguration)
  StringRef getArgument() const override { return "weft-x86-reuse-amx-configuration"; }
  StringRef getDescription() const override { return "Reuse one closed tile configuration across an ordered traversal"; }
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    for (auto loop : loops) {
      SmallVector<func::CallOp> configurations, releases;
      StringAttr configuration;
      bool eligible = true;
      loop.walk([&](Operation *op) {
        if (isa<func::CallIndirectOp, LLVM::CallOp, LLVM::InlineAsmOp>(op)) { eligible = false; return; }
        auto call = dyn_cast<func::CallOp>(op);
        if (!call) return;
        auto name = call.getCallee();
        if (name == "llvm.x86.ldtilecfg") {
          auto address = call.getOperand(0).getDefiningOp<LLVM::AddressOfOp>();
          if (!address || (configuration && configuration.getValue() != address.getGlobalName())) {
            eligible = false; return;
          }
          configuration = StringAttr::get(&getContext(), address.getGlobalName());
          configurations.push_back(call);
        } else if (name == "llvm.x86.tilerelease") {
          releases.push_back(call);
        } else if (name != "llvm.x86.tilezero" && name != "llvm.x86.tileloadd64" && name != "llvm.x86.tdpbf16ps" &&
                   name != "llvm.x86.tdpbssd" && name != "llvm.x86.tdpbsud" &&
                   name != "llvm.x86.tdpbusd" && name != "llvm.x86.tdpbuud") {
          auto function = getOperation().lookupSymbol<func::FuncOp>(name);
          if (!function || (!function->hasAttr("weft.x86.amx_transfer") &&
                            !function->hasAttr("weft.x86.preserves_tile_state"))) eligible = false;
        }
      });
      if (!eligible || configurations.empty() || configurations.size() != releases.size()) continue;
      for (auto [begin, end] : llvm::zip(configurations, releases))
        if (begin->getBlock() != end->getBlock() || !begin->isBeforeInBlock(end)) eligible = false;
      if (!eligible) continue;
      OpBuilder b(loop);
      auto global = getOperation().lookupSymbol<LLVM::GlobalOp>(configuration.getValue());
      intrinsic(getOperation(), b, loop.getLoc(), "llvm.x86.ldtilecfg",
          ValueRange{LLVM::AddressOfOp::create(b, loop.getLoc(), global)});
      b.setInsertionPointAfter(loop);
      intrinsic(getOperation(), b, loop.getLoc(), "llvm.x86.tilerelease", {});
      loop->setAttr("weft.x86.amx_configurations_reused", b.getI64IntegerAttr(configurations.size()));
      for (auto call : configurations) call.erase();
      for (auto call : releases) call.erase();
    }
  }
};
}

std::unique_ptr<mlir::Pass> weft::x86::createAMXConfigurationReusePass() {
  return std::make_unique<ReuseAMXConfiguration>();
}
