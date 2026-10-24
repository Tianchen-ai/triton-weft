#include "Emitter.h"
#include "../IME/FragmentEmission.h"
#include "Weft/Target/RISCVFragment.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace weft::riscv_emission {

std::optional<std::string> fragmentVectorType(riscv::FragmentType fragment) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(fragment.getElementType());
  if (!integer || integer.isSignless() || !findRISCVFragmentCapability(fragment.getFamily()))
    return std::nullopt;
  const int64_t groups = riscv::fragmentPartResourceGroups(fragment);
  if (groups != 1 && groups != 2 && groups != 4 && groups != 8)
    return std::nullopt;
  return "vint" + std::to_string(std::max(8U, integer.getWidth())) + "m" + std::to_string(groups) + "_t";
}

mlir::LogicalResult Emitter::compileIMEZero(riscv::IMEZeroOp operation) {
  auto fragment = operation.getResult().getType();
  auto type = fragmentVectorType(fragment);
  if (!type)
    return fail(operation, "fragment zero has no selected register carrier");
  const int64_t bits = riscv_internal::logicalBitWidth(fragment.getElementType());
  const int64_t groups = riscv::fragmentPartResourceGroups(fragment);
  const int64_t elements = fragment.getPacking().getRowsPerTile() *
                           fragment.getPacking().getColumnsPerTile();
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const bool prefix = operation.getLeaf().getInstruction() == "ime.zero.local-prefix";
  const bool local = prefix || operation.getLeaf().getInstruction() == "ime.zero.local";
  if (!local && operation.getLeaf().getInstruction() != "ime.zero")
    return fail(operation, "fragment zero has no exact selected local realization");
  if (local) {
    for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
      result.parts.push_back(fresh("ime_zero"));
      line(*type + " " + result.parts.back() + ";");
    }
    line("__asm__ volatile(");
    const std::string lmul = prefix ? lmulSpelling(operation.getLeaf().getParameters()[0])
                                    : "m" + std::to_string(groups);
    line("  \"vsetvli zero, %[vl], e" + std::to_string(bits) + ", " + lmul + ", ta, ma\\n\\t\"");
    std::string outputs = "  : ";
    for (auto [part, name] : llvm::enumerate(result.parts)) {
      line("  \"vmv.v.i %[part" + std::to_string(part) + "], 0\\n\\t\"");
      if (part) outputs += ", ";
      outputs += "[part" + std::to_string(part) + "] \"=vr\"(" + name + ")";
    }
    line(outputs);
    line("  : [vl] \"r\"((size_t)" + std::to_string(elements) + ") : \"vl\", \"vtype\");");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    result.parts.push_back(fresh("ime_zero"));
    line(*type + " " + result.parts.back() + " = __riscv_vmv_v_x_i" +
         std::to_string(bits) + "m" + std::to_string(groups) +
         "(0, " + std::to_string(elements) + ");");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEPack(riscv::IMEPackOp operation) {
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input))
    return mlir::failure();
  auto inputType = mlir::cast<riscv::ValueType>(operation.getInput().getType());
  auto fragment = operation.getResult().getType();
  auto type = fragmentVectorType(fragment);
  const int64_t rows = operation.getPacking().getRowsPerTile();
  const int64_t columns = operation.getPacking().getColumnsPerTile();
  const int64_t parts = riscv::fragmentPartCount(fragment);
  if (!type || input->kind != Binding::Kind::Vector ||
      static_cast<int64_t>(input->parts.size()) != rows * parts ||
      riscv::fragmentPartResourceGroups(fragment) != 1)
    return fail(operation, "fragment pack has no complete selected register slices");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const std::string suffix = vectorSuffix(operation.getInput());
  const bool full = inputType.getLayout().getValidity() == "full";
  if (operation.getPacking().getStorageBits() == 4) {
    const std::string sourceLMUL = lmulSpelling(inputType.getLayout().getLmulEighths());
    const std::string narrowLMUL = lmulSpelling(inputType.getLayout().getLmulEighths() / 2);
    for (int64_t atom = 0; atom < parts; ++atom) {
      const std::string packed = fresh("ime_nibble_pack");
      line("vint8m1_t " + packed + " = __riscv_vmv_v_x_i8m1(0, 128);");
      for (int64_t row = 0; row < rows; ++row) {
        const int64_t sourcePart = atom * rows + row;
        const std::string prefix = "u16" + sourceLMUL;
        const std::string raw = fresh("ime_nibble_pairs"), joined = fresh("ime_nibble_bytes");
        const bool signedInput = mlir::cast<mlir::IntegerType>(inputType.getElementType()).isSigned();
        std::string bytes = input->parts[sourcePart];
        if (signedInput) bytes = "__riscv_vreinterpret_v_" + suffix + "_u8" + sourceLMUL + "(" + bytes + ")";
        if (!full) {
          const std::string active = partVL(operation.getInput(), sourcePart);
          const std::string mask = "__riscv_vmsgeu_vx_u8" + sourceLMUL + "_b" +
              std::to_string(64 / inputType.getLayout().getLmulEighths()) +
              "(__riscv_vid_v_u8" + sourceLMUL + "(" + std::to_string(columns) + "), " + active + ", " + std::to_string(columns) + ")";
          bytes = "__riscv_vmerge_vxm_u8" + sourceLMUL + "(" + bytes + ", 0, " + mask + ", " + std::to_string(columns) + ")";
        }
        line("vuint16" + sourceLMUL + "_t " + raw + " = __riscv_vreinterpret_v_u8" + sourceLMUL + "_u16" + sourceLMUL + "(" + bytes + ");");
        const std::string count = std::to_string(columns / 2);
        const std::string combined = "__riscv_vor_vv_" + prefix + "(__riscv_vand_vx_" + prefix + "(" + raw + ", 15, " + count + "), __riscv_vand_vx_" + prefix + "(__riscv_vsrl_vx_" + prefix + "(" + raw + ", 4, " + count + "), 240, " + count + "), " + count + ")";
        line("vuint8" + narrowLMUL + "_t " + joined + " = __riscv_vncvt_x_x_w_u8" + narrowLMUL + "(" + combined + ", " + count + ");");
        const std::string expanded = "__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vlmul_ext_v_u8" + narrowLMUL + "_u8m1(" + joined + "))";
        const std::string vl = count;
        line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + expanded + ", " + std::to_string(row * columns / 2) + ", " + std::to_string(row * columns / 2) + " + " + vl + ");");
      }
      result.parts.push_back(packed);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t atom = 0; atom < parts; ++atom) {
    const std::string packed = fresh("ime_pack");
    if (full) {
      std::string first = input->parts[atom * rows];
      if (suffix != "i8m1")
        first = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + first + ")";
      line(*type + " " + packed + " = " + first + ";");
    } else {
      line(*type + " " + packed + " = __riscv_vmv_v_x_i8m1(0, " +
           std::to_string(kernel.getTarget().getVlenBits() / 8) + ");");
    }
    for (int64_t row = full ? 1 : 0; row < rows; ++row) {
      const int64_t sourcePart = atom * rows + row;
      std::string part = input->parts[sourcePart];
      if (suffix != "i8m1")
        part = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + part + ")";
      const std::string offset = std::to_string(row * columns);
      // Full slices overwrite every logical cell by the final insert. Tail
      // slices keep zero padding outside their explicitly active prefix.
      const std::string vl = full ? std::to_string(rows * columns)
          : offset + " + " + partVL(operation.getInput(), sourcePart);
      line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + part +
           ", " + offset + ", " + vl + ");");
    }
    result.parts.push_back(packed);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentStorageLoad(riscv::IMEFragmentStorageLoadOp operation) {
  Binding projection = bindings.lookup(operation.getInput());
  auto field = fieldFor(projection);
  if (projection.kind != Binding::Kind::Field || !field || !projection.field.relativeIndices.empty())
    return fail(operation, "fragment load requires its selected field projection binding");
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record))
      return mlir::failure();
    owner = std::move(*record);
  }
  if (owner.kind != Binding::Kind::Record || owner.recordElements <= 0)
    return fail(operation, "fragment field load has no declared record binding");
  std::string logicalIndex = "0";
  if (projection.field.index) {
    Binding point = bindings.lookup(*projection.field.index);
    if (point.kind != Binding::Kind::Point)
      return fail(operation, "fragment field load has no selected inner point");
    logicalIndex = "(" + point.point.base + " % " + std::to_string(owner.recordElements) + ")";
  }
  const int64_t bits = riscv_internal::logicalBitWidth(field->type);
  auto storage = singleStorageFragment(*field, logicalIndex, bits);
  if (!storage)
    return fail(operation, "fragment field load has no closed byte projection");
  auto fragment = operation.getResult().getType();
  const int64_t cells = fragment.getPacking().getRowsPerTile() * fragment.getPacking().getColumnsPerTile() *
                        fragment.getPacking().getStorageBits() / 8;
  const auto instruction = operation.getLeaf().getInstruction();
  if (instruction == "ime.field-nibble-cohorts") {
    const auto source = operation.getInput().getDefiningOp<riscv::FieldOp>()
                            .getOwner().getDefiningOp<riscv::LoadOp>().getRegion();
    auto rowExtent = descriptorExtent(source, fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[0]]);
    if (!rowExtent) return fail(operation, "nibble cohort read lost its active row extent");
    Binding result;
    result.kind = Binding::Kind::Fragment;
    result.fragmentFamily = fragment.getFamily().str();
    for (int64_t atom = 0; atom < riscv::fragmentPartCount(fragment); ++atom) {
      const std::string packed = fresh("ime_nibble_cohorts");
      line("vint8m1_t " + packed + " = __riscv_vmv_v_x_i8m1(0, 128);");
      for (int64_t half = 0; half < 2; ++half) {
        const int64_t first = atom * 8 + half * 4;
        if (first) line("if ((" + *rowExtent + ") > " + std::to_string(first) + ") {");
        const std::string chunk = fresh("ime_nibble_cohort");
        line("vint8m1_t " + chunk + " = __riscv_vle8_v_i8m1((const int8_t *)" +
             fieldStorageAddress(owner, *field, storage->byte, std::to_string(first)) + ", 64);");
        line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + chunk + ", " +
             std::to_string(half * 64) + ", " + std::to_string((half + 1) * 64) + ");");
        if (first) line("}");
      }
      result.parts.push_back(packed);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (instruction == "ime.field-nibble-pack") {
    const auto source = operation.getInput().getDefiningOp<riscv::FieldOp>()
                            .getOwner().getDefiningOp<riscv::LoadOp>().getRegion();
    const int64_t rows = fragment.getPacking().getRowsPerTile();
    auto rowExtent = descriptorExtent(source, fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[0]]);
    if (!rowExtent) return fail(operation, "nibble packing read lost its active row extent");
    Binding result;
    result.kind = Binding::Kind::Fragment;
    result.fragmentFamily = fragment.getFamily().str();
    for (int64_t atom = 0; atom < riscv::fragmentPartCount(fragment); ++atom) {
      llvm::SmallVector<std::string> chunks;
      for (int64_t half = 0; half < (rows == 8 ? 2 : 1); ++half) {
        const int64_t first = atom * rows + half * 4;
        const std::string chunk = fresh("ime_narrow_chunk");
        line("vint8m1_t " + chunk + " = __riscv_vmv_v_x_i8m1(0, 128);");
        if (first) line("if ((" + *rowExtent + ") > " + std::to_string(first) + ") {");
        line(chunk + " = __riscv_vle8_v_i8m1((const int8_t *)" +
             fieldStorageAddress(owner, *field, storage->byte, std::to_string(first)) + ", " +
             std::to_string(std::min<int64_t>(rows, 4) * 32) + ");");
        if (first) line("}");
        if (operation.getLeaf().getParameters()[0])
          line(chunk + " = __riscv_vsra_vx_i8m1(" + chunk + ", 4, 128);");
        chunks.push_back(chunk);
      }
      const std::string pair = fresh("ime_narrow_order"), packed = fresh("ime_narrow_fragment");
      line("vint8m2_t " + pair + ";");
      line("__asm__ volatile(\"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
      // vnpack4's 256-bit output chunks consume 512 input bits. Split
      // pairs of source rows so narrowing restores contiguous row order.
      line("  \".insn r 0x2b, 7, 0x33, %0, %1, %2\\n\\t\"");
      line("  : \"=&vr\"(" + pair + ") : \"vr\"(" + chunks[0] + "), \"vr\"(" + chunks[rows == 8 ? 1 : 0] + ") : \"t0\", \"vl\", \"vtype\");");
      line("vint8m1_t " + packed + ";");
      line("__asm__ volatile(\"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
      line("  \".insn r 0x2b, 3, 0x21, %0, %1, %2\\n\\t\"");
      line("  : \"=&vr\"(" + packed + ") : \"vr\"(__riscv_vget_v_i8m2_i8m1(" + pair + ", 0)),");
      line("    \"vr\"(__riscv_vget_v_i8m2_i8m1(" + pair + ", 1)) : \"t0\", \"vl\", \"vtype\");");
      result.parts.push_back(packed);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const bool grouped = instruction == "ime.field-raw-load.grouped" ||
                       instruction == "ime.field-raw-load.grouped.scalar-prime";
  if (instruction == "ime.field-raw-load.scalar-prime" ||
      instruction == "ime.field-raw-load.grouped.scalar-prime") {
    const int64_t finalByte = cells * riscv::fragmentPartCount(fragment) - 1;
    line("__asm__ volatile(\"lb zero, " + std::to_string(finalByte) +
         "(%0)\" : : \"r\"((const uint8_t *)" +
         fieldStorageAddress(owner, *field, storage->byte) + ") : \"memory\");");
  }
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (instruction == "ime.field-raw-load.row-windows") {
    const int64_t rows = fragment.getPacking().getRowsPerTile();
    const int64_t columns = fragment.getPacking().getColumnsPerTile() * fragment.getPacking().getStorageBits() / 8;
    const int64_t stride = operation.getAccess().getFieldChunkBytes();
    const std::string first = fieldStorageAddress(owner, *field, storage->byte);
    for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
      const std::string packed = fresh("ime_row_windows");
      for (int64_t row = 0; row < rows; ++row) {
        const std::string at = "((const int8_t *)" + first + " + " +
                               std::to_string((part * rows + row) * stride) + ")";
        const std::string value = "__riscv_vle8_v_i8m1(" + at + ", " +
                                  std::to_string(columns) + ")";
        if (!row)
          line("vint8m1_t " + packed + " = " + value + ";");
        else
          line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + value +
               ", " + std::to_string(row * columns) + ", " + std::to_string((row + 1) * columns) + ");");
      }
      result.parts.push_back(packed);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (grouped) {
    const int64_t parts = riscv::fragmentPartCount(fragment);
    const std::string suffix = "i8m" + std::to_string(parts);
    const std::string raw = fresh("ime_field_group");
    line("vint8m" + std::to_string(parts) + "_t " + raw + " = __riscv_vle8_v_" + suffix +
         "((const int8_t *)" + fieldStorageAddress(owner, *field, storage->byte) + ", " +
         std::to_string(cells * parts) + ");");
    for (int64_t part = 0; part < parts; ++part) {
      result.parts.push_back(fresh("ime_field"));
      line("vint8m1_t " + result.parts.back() + " = __riscv_vget_v_" + suffix + "_i8m1(" +
           raw + ", " + std::to_string(part) + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
  const std::string raw = fresh("ime_field_bytes");
  const std::string address = "((const uint8_t *)" + fieldStorageAddress(owner, *field, storage->byte) +
                              " + " + std::to_string(part * cells) + ")";
  line("vuint8m1_t " + raw + " = __riscv_vle8_v_u8m1((const uint8_t *)" + address + ", " + std::to_string(cells) + ");");
  const std::string expression = "__riscv_vreinterpret_v_u8m1_i8m1(" + raw + ")";
  result.parts.push_back(fresh("ime_field"));
  line("vint8m1_t " + result.parts.back() + " = " + expression + ";");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentChunkSplitLoad(riscv::IMEFragmentChunkSplitLoadOp operation) {
  Binding projection = bindings.lookup(operation.getField());
  auto field = fieldFor(projection);
  const auto instruction = operation.getLeaf().getInstruction();
  const bool nibbles = instruction == "ime.fragment-nibbles-signed" ||
                       instruction == "ime.fragment-nibbles-unsigned";
  if (projection.kind != Binding::Kind::Field || !field || projection.field.index ||
      !projection.field.relativeIndices.empty() ||
      (instruction != "spacemit-ime2-vupack128" &&
       instruction != "rvv.fragment-chunk-slide" && !nibbles))
    return fail(operation, "chunk split requires its selected full field and local permutation");
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record)) return mlir::failure();
    owner = std::move(*record);
  }
  auto storage = singleStorageFragment(*field, "0", 8);
  if (owner.kind != Binding::Kind::Record || !storage)
    return fail(operation, "chunk split lost the declared record storage");
  auto fragment = operation.getLow().getType();
  const int64_t rows = fragment.getPacking().getRowsPerTile();
  const bool rowGuard = operation.getLeaf().getParameters()[1];
  const auto source = operation.getField().getDefiningOp<riscv::FieldOp>()
                          .getOwner().getDefiningOp<riscv::LoadOp>().getRegion();
  const auto rowExtent = rowGuard ? descriptorExtent(source,
      fragment.getAxisIds()[fragment.getPacking().getAxisOrder()[0]]) : std::optional<std::string>();
  if (rowGuard && !rowExtent)
    return fail(operation, "guarded chunk supply lost its active row extent");
  const std::string first = fieldStorageAddress(owner, *field, storage->byte);
  Binding low, high;
  low.kind = high.kind = Binding::Kind::Fragment;
  low.fragmentFamily = high.fragmentFamily = fragment.getFamily().str();
  // Each byte contains adjacent logical elements. Widening before separating
  // its nibbles produces one little-endian i16 lane containing two i8 cells.
  // Decode at most four declared rows; a second cohort retains its own address.
  auto decodeNibbles = [&](const std::string &address, int64_t rowCount, int64_t packedGroups = 0) {
    const std::string count = std::to_string(rowCount * 16);
    const std::string inputLMUL = packedGroups ? "m" + std::to_string(packedGroups) : "mf2";
    const std::string outputLMUL = packedGroups ? "m" + std::to_string(packedGroups * 2) : "m1";
    const std::string bytes = fresh("ime_nibbles"), wide = fresh("ime_nibbles_wide");
    const std::string lower = fresh("ime_nibbles_low"), upper = fresh("ime_nibbles_high");
    const std::string paired = fresh("ime_nibbles_pair"), decoded = fresh("ime_nibbles_decoded");
    line("vuint8" + inputLMUL + "_t " + bytes + " = __riscv_vle8_v_u8" + inputLMUL + "((const uint8_t *)" + address + ", " + count + ");");
    line("vuint16" + outputLMUL + "_t " + wide + " = __riscv_vzext_vf2_u16" + outputLMUL + "(" + bytes + ", " + count + ");");
    line("vuint16" + outputLMUL + "_t " + upper + " = __riscv_vsll_vx_u16" + outputLMUL + "(" + wide + ", 4, " + count + ");");
    line("vuint16" + outputLMUL + "_t " + lower + " = __riscv_vor_vv_u16" + outputLMUL + "(" + wide + ", " + upper + ", " + count + ");");
    line("vuint16" + outputLMUL + "_t " + paired + " = __riscv_vand_vx_u16" + outputLMUL + "(" + lower + ", 3855, " + count + ");");
    line("vint8" + outputLMUL + "_t " + decoded + " = __riscv_vreinterpret_v_u8" + outputLMUL + "_i8" + outputLMUL + "(__riscv_vreinterpret_v_u16" + outputLMUL + "_u8" + outputLMUL + "(" + paired + "));");
    if (instruction == "ime.fragment-nibbles-signed") {
      const std::string lanes = std::to_string(rowCount * 32);
      line(decoded + " = __riscv_vsra_vx_i8" + outputLMUL + "(__riscv_vsll_vx_i8" + outputLMUL + "(" + decoded + ", 4, " + lanes + "), 4, " + lanes + ");");
    }
    return decoded;
  };
  auto readChunk = [&](const std::string &address, int64_t firstRow) {
    const std::string result = fresh("ime_chunk_read");
    const bool guarded = rowGuard && firstRow > 0;
    line("vint8m1_t " + result + ";");
    if (guarded)
      line("if ((" + *rowExtent + ") > " + std::to_string(firstRow) + ") {");
    if (nibbles)
      line(result + " = " + decodeNibbles(address, std::min<int64_t>(rows, 4)) + ";");
    else
      line(result + " = __riscv_vle8_v_i8m1((const int8_t *)" + address + ", " +
           std::to_string(std::min<int64_t>(rows, 4) * 32) + ");");
    if (guarded) {
      line("} else {");
      line("  " + result + " = __riscv_vmv_v_x_i8m1(0, 128);");
      line("}");
    }
    return result;
  };
  if (rows == 1) {
    const std::string value = fresh("ime_chunk"), upper = fresh("ime_chunk_high");
    if (nibbles)
      line("vint8m1_t " + value + " = " + decodeNibbles(first, 1) + ";");
    else
      line("vint8m1_t " + value + " = __riscv_vle8_v_i8m1((const int8_t *)" + first + ", 32);");
    line("vint8m1_t " + upper + " = __riscv_vslidedown_vx_i8m1(" + value + ", 16, 16);");
    low.parts.push_back(value);
    high.parts.push_back(upper);
    bindings[operation.getLow()] = std::move(low);
    bindings[operation.getHigh()] = std::move(high);
    return mlir::success();
  }
  const int64_t loadGroups = operation.getLeaf().getParameters()[0];
  const int64_t atomsPerLoad = rows == 8 && loadGroups > 1 ? (nibbles ? loadGroups : loadGroups / 2) : 1;
  const bool groupedNibbles = nibbles && loadGroups > 1;
  const int64_t decodedGroups = groupedNibbles ? loadGroups * 2 : loadGroups;
  std::string loaded;
  llvm::SmallVector<std::string> pairs, lefts, rights;
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    const std::string at = "((const int8_t *)" +
        fieldStorageAddress(owner, *field, storage->byte, std::to_string(part * rows)) + ")";
    if (groupedNibbles) {
      if (part % atomsPerLoad == 0)
        loaded = decodeNibbles(at, loadGroups * 8, loadGroups);
    } else if (nibbles || rowGuard) {
      loaded = readChunk(at, part * rows);
    } else if (part % atomsPerLoad == 0) {
      loaded = fresh("ime_chunk_load");
      const std::string suffix = "i8m" + std::to_string(loadGroups);
      line("vint8m" + std::to_string(loadGroups) + "_t " + loaded +
           " = __riscv_vle8_v_" + suffix + "(" + at + ", " +
           std::to_string(loadGroups > 1 ? loadGroups * 128 : std::min<int64_t>(rows, 4) * 32) + ");");
    }
    const int64_t subregister = (part % atomsPerLoad) * 2;
    const std::string left = decodedGroups == 1 ? loaded : "__riscv_vget_v_i8m" +
        std::to_string(decodedGroups) + "_i8m1(" + loaded + ", " + std::to_string(subregister) + ")";
    std::string right = left;
    if (decodedGroups > 1) {
      right = "__riscv_vget_v_i8m" + std::to_string(decodedGroups) + "_i8m1(" +
          loaded + ", " + std::to_string(subregister + 1) + ")";
    } else if (rows == 8) {
      const std::string next = fieldStorageAddress(owner, *field, storage->byte, std::to_string(part * rows + 4));
      if (nibbles || rowGuard) {
        right = readChunk(next, part * rows + 4);
      } else {
        right = fresh("ime_chunk_right");
        line("vint8m1_t " + right + " = __riscv_vle8_v_i8m1((const int8_t *)" + next + ", 128);");
      }
    }
    const std::string pair = fresh("ime_chunk_pair");
    line("vint8m2_t " + pair + ";");
    pairs.push_back(pair);
    lefts.push_back(left);
    rights.push_back(right);
  }
  line("__asm__ volatile(\"vsetvli t0, zero, e8, m1, ta, ma\\n\\t\"");
  std::string outputs, inputs;
  for (auto [part, pair] : llvm::enumerate(pairs)) {
    const std::string p = std::to_string(part);
    line("  \".insn r 0x2b, 5, 0x33, %[pair" + p + "], %[left" + p + "], %[right" + p + "]\\n\\t\"");
    if (part) { outputs += ", "; inputs += ", "; }
    outputs += "[pair" + p + "] \"=&vr\"(" + pair + ")";
    inputs += "[left" + p + "] \"vr\"(" + lefts[part] + "), [right" + p + "] \"vr\"(" + rights[part] + ")";
  }
  line("  : " + outputs + " : " + inputs + " : \"t0\", \"vl\", \"vtype\");");
  for (const std::string &pair : pairs) {
    low.parts.push_back(fresh("ime_chunk_low"));
    high.parts.push_back(fresh("ime_chunk_high"));
    line("vint8m1_t " + low.parts.back() + " = __riscv_vget_v_i8m2_i8m1(" + pair + ", 0);");
    line("vint8m1_t " + high.parts.back() + " = __riscv_vget_v_i8m2_i8m1(" + pair + ", 1);");
  }
  bindings[operation.getLow()] = std::move(low);
  bindings[operation.getHigh()] = std::move(high);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentWindowLoad(riscv::IMEFragmentWindowLoadOp operation) {
  Binding projection = bindings.lookup(operation.getField());
  Binding base = bindings.lookup(operation.getLogicalBase());
  auto field = fieldFor(projection);
  if (projection.kind != Binding::Kind::Field || !field || base.kind != Binding::Kind::Scalar)
    return fail(operation, "fragment byte windows require their selected field and logical base");
  Binding owner = bindings.lookup(projection.field.owner);
  if (owner.kind == Binding::Kind::Slice) {
    auto record = recordForSlice(projection.field.owner);
    if (mlir::failed(record)) return mlir::failure();
    owner = std::move(*record);
  }
  auto fragment = operation.getResult().getType();
  auto packing = fragment.getPacking();
  const int64_t rowAxis = fragment.getAxisIds()[packing.getAxisOrder()[0]];
  auto stride = llvm::find_if(owner.recordByteStrides,
      [&](const auto &entry) { return entry.first == rowAxis; });
  if (owner.kind != Binding::Kind::Record || owner.interleaveRows ||
      stride == owner.recordByteStrides.end())
    return fail(operation, "fragment byte windows lost their independent record stride");
  const int64_t rows = packing.getRowsPerTile(), columns = packing.getColumnsPerTile();
  const std::string lmul = lmulSpelling(columns * 64 / kernel.getTarget().getVlenBits());
  const int64_t group = operation.getAccess().getGroupSize(), layer = operation.getAccess().getLayerSize();
  const std::string byte = "(" + std::to_string(operation.getAccess().getBitOffset() / 8) +
      " + (" + base.scalar + ") / " + std::to_string(group) + " * " + std::to_string(layer) +
      " + (" + base.scalar + ") % " + std::to_string(layer) + ")";
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    const std::string packed = fresh("ime_byte_windows");
    bool words = operation.getLeaf().getInstruction() == "ime.field-byte-windows.guarded-vlse64";
    if (words) {
      const std::string address = "(" + owner.recordPointer + " + " +
          std::to_string(part * rows) + " * " + stride->second + " + " + byte + ")";
      line("vint8m1_t " + packed + ";");
      line("if ((((uintptr_t)" + address + " | (uintptr_t)(" + stride->second + ")) & 7u) == 0) {");
      const std::string loaded = fresh("ime_word_windows");
      line("  vuint64m1_t " + loaded + " = __riscv_vlse64_v_u64m1((const uint64_t *)" + address +
          ", (ptrdiff_t)(" + stride->second + "), " + std::to_string(rows) + ");");
      line("  " + packed + " = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(" + loaded + "));");
      line("} else {");
    }
    llvm::SmallVector<std::string> windows;
    for (int64_t row = 0; row < rows; ++row) {
      const std::string address = "((const int8_t *)(" + owner.recordPointer + " + " +
          std::to_string(part * rows + row) + " * " + stride->second + " + " + byte + "))";
      const std::string window = fresh("ime_byte_window");
      line("vint8" + lmul + "_t " + window + " = __riscv_vle8_v_i8" + lmul +
          "(" + address + ", " + std::to_string(columns) + ");");
      windows.push_back("__riscv_vlmul_ext_v_i8" + lmul + "_i8m1(" + window + ")");
    }
    line((words ? "" : "vint8m1_t ") + packed + " = " + windows.front() + ";");
    for (int64_t row = 1; row < rows; ++row)
      line(packed + " = __riscv_vslideup_vx_i8m1_tu(" + packed + ", " + windows[row] + ", " +
          std::to_string(row * columns) + ", " + std::to_string(rows * columns) + ");");
    if (words) line("}");
    result.parts.push_back(packed);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentDenseLoad(
    riscv::IMEFragmentDenseLoadOp operation) {
  auto fragment = operation.getResult().getType();
  auto packing = fragment.getPacking();
  int64_t rowAxis = fragment.getAxisIds()[packing.getAxisOrder()[0]];
  auto address = byteReadAddress(operation.getSource());
  auto stride = descriptorStride(operation.getSource(), rowAxis);
  if (operation.getLeaf().getInstruction() == "ime.dense-row-bytes") {
    const int64_t columns = packing.getColumnsPerTile(), rows = packing.getRowsPerTile();
    const int64_t innerAxis = fragment.getAxisIds()[packing.getAxisOrder()[1]];
    auto rowExtent = descriptorExtent(operation.getSource(), rowAxis);
    auto columnExtent = descriptorExtent(operation.getSource(), innerAxis);
    if (!address || !stride || !rowExtent || !columnExtent)
      return fail(operation, "dense byte fragment load lost its explicit extents or row stride");
    Binding result;
    result.kind = Binding::Kind::Fragment;
    result.fragmentFamily = fragment.getFamily().str();
    const std::string active = fresh("ime_columns");
    line("const size_t " + active + " = (" + *columnExtent + ") < " + std::to_string(columns) +
         " ? (" + *columnExtent + ") : " + std::to_string(columns) + ";");
    for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
      const std::string value = fresh("ime_dense_fragment");
      line("vint8m1_t " + value + " = __riscv_vmv_v_x_i8m1(0, " + std::to_string(rows * columns) + ");");
      for (int64_t row = 0; row < rows; ++row) {
        const std::string at = "((const int8_t *)(" + *address + ") + " +
            std::to_string(part * rows + row) + " * (ptrdiff_t)(" + *stride + "))";
        line("if (" + std::to_string(part * rows + row) + " < (" + *rowExtent + ") && " + active + ") {");
        if (row == 0) {
          line("  " + value + " = __riscv_vle8_v_i8m1_tu(" + value + ", " + at + ", " + active + ");");
        } else {
          const std::string temporary = fresh("ime_dense_row");
          line("  vint8m1_t " + temporary + " = __riscv_vle8_v_i8m1(" + at + ", " + active + ");");
          line("  " + value + " = __riscv_vslideup_vx_i8m1_tu(" + value + ", " + temporary +
               ", " + std::to_string(row * columns) + ", " + std::to_string(row * columns) + " + " + active + ");");
        }
        line("}");
      }
      result.parts.push_back(value);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (!address || !stride ||
      operation.getLeaf().getInstruction() != "ime.dense-row-words")
    return fail(operation, "dense fragment transfer lost its exact selected memory binding");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  const int64_t rows = packing.getRowsPerTile();
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
    const std::string words = fresh("ime_dense_words");
    const std::string at = "((const uint8_t *)(" + *address + ") + " +
        std::to_string(part * rows) + " * (" + *stride + "))";
    line("vuint64m1_t " + words + " = __riscv_vlse64_v_u64m1((const uint64_t *)" +
         at + ", (ptrdiff_t)(" + *stride + "), " + std::to_string(rows) + ");");
    result.parts.push_back(fresh("ime_dense_fragment"));
    line("vint8m1_t " + result.parts.back() +
         " = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(" +
         words + "));");
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentDecode(riscv::IMEFragmentDecodeOp operation) {
  const Binding input = bindings.lookup(operation.getInput());
  const Binding shift = bindings.lookup(operation.getShift());
  auto fragment = operation.getResult().getType();
  if (input.kind != Binding::Kind::Fragment || shift.kind != Binding::Kind::Scalar ||
      input.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)))
    return fail(operation, "fragment decode requires its raw-byte SSA and bounded shift");
  const int64_t bits = operation.getElementBits();
  auto constant = operation.getShift().getDefiningOp<mlir::arith::ConstantIndexOp>();
  const std::optional<int64_t> knownShift = constant
      ? std::optional<int64_t>(constant.value()) : std::nullopt;
  const bool grouped = operation.getLeaf().getInstruction() == "ime.decode.grouped-bits";
  const int64_t parts = riscv::fragmentPartCount(fragment);
  const std::string group = "m" + std::to_string(grouped ? parts : 1);
  const std::string vl = std::to_string(fragment.getPacking().getRowsPerTile() *
      fragment.getPacking().getColumnsPerTile() * (grouped ? parts : 1));
  llvm::SmallVector<std::string> operands(input.parts.begin(), input.parts.end());
  if (grouped) {
    const std::string packed = fresh("ime_decode_input");
    line("vint8" + group + "_t " + packed + " = __riscv_vundefined_i8" + group + "();");
    for (auto [part, value] : llvm::enumerate(input.parts))
      line(packed + " = __riscv_vset_v_i8m1_i8" + group + "(" + packed + ", " +
           std::to_string(part) + ", " + value + ");");
    operands = {packed};
  }
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  for (const std::string &part : operands) {
    std::string expression = "__riscv_vreinterpret_v_i8" + group + "_u8" + group + "(" + part + ")";
    if (knownShift != 0)
      expression = "__riscv_vsrl_vx_u8" + group + "(" + expression + ", " + shift.scalar + ", " + vl + ")";
    if (knownShift != 8 - bits)
      expression = "__riscv_vand_vx_u8" + group + "(" + expression + ", " +
                   std::to_string((1u << bits) - 1) + ", " + vl + ")";
    expression = "__riscv_vreinterpret_v_u8" + group + "_i8" + group + "(" + expression + ")";
    if (operation.getSignedElements())
      expression = "__riscv_vsra_vx_i8" + group + "(__riscv_vsll_vx_i8" + group + "(" + expression + ", " +
                   std::to_string(8 - bits) + ", " + vl + "), " + std::to_string(8 - bits) + ", " + vl + ")";
    const std::string decoded = fresh("ime_decode");
    line("vint8" + group + "_t " + decoded + " = " + expression + ";");
    if (grouped) {
      for (int64_t index = 0; index < parts; ++index) {
        result.parts.push_back(fresh("ime_decode_part"));
        line("vint8m1_t " + result.parts.back() + " = __riscv_vget_v_i8" + group +
             "_i8m1(" + decoded + ", " + std::to_string(index) + ");");
      }
    } else {
      result.parts.push_back(decoded);
    }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentRowBroadcast(riscv::IMEFragmentRowBroadcastOp operation) {
  auto input = materializeNumeric(operation.getInput(), bindings.lookup(operation.getInput()));
  if (mlir::failed(input) || input->kind != Binding::Kind::Vector || input->parts.size() != 1)
    return fail(operation, "fragment broadcast has no selected row input or local index sequence");
  auto fragment = operation.getResult().getType();
  const int64_t rows = fragment.getPacking().getRowsPerTile();
  const int64_t parts = riscv::fragmentPartCount(fragment);
  const std::string suffix = vectorSuffix(operation.getInput());
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (operation.getLeaf().getInstruction() == "ime.rows.byte-repeat") {
    const std::string byteSuffix = "u8" + lmulSpelling(operation.getInput().getType().getLayout().getLmulEighths());
    const std::string wordSuffix = "u64" + lmulSpelling(parts * 8);
    const std::string wideByteSuffix = "u8" + lmulSpelling(parts * 8);
    const std::string repeatedSuffix = "i8" + lmulSpelling(parts * 8);
    const std::string vl = std::to_string(rows * parts);
    const std::string words = fresh("row_words");
    line("vuint64" + lmulSpelling(parts * 8) + "_t " + words + " = __riscv_vzext_vf8_" +
         wordSuffix + "(__riscv_vreinterpret_v_" + suffix + "_" + byteSuffix + "(" +
         input->parts.front() + "), " + vl + ");");
    const std::string repeated = fresh("row_bytes");
    line("vint8" + lmulSpelling(parts * 8) + "_t " + repeated + " = __riscv_vreinterpret_v_" +
         wideByteSuffix + "_" + repeatedSuffix + "(__riscv_vreinterpret_v_" + wordSuffix + "_" +
         wideByteSuffix + "(__riscv_vmul_vx_" + wordSuffix + "(" + words +
         ", UINT64_C(0x0101010101010101), " + vl + ")));");
    for (int64_t part = 0; part < parts; ++part) {
      result.parts.push_back(fresh("fragment_rows"));
      const std::string value = parts == 1 ? repeated : "__riscv_vget_v_" + repeatedSuffix +
          "_i8m1(" + repeated + ", " + std::to_string(part) + ")";
      line("vint8m1_t " + result.parts.back() + " = " + value + ";");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (operation.getLeaf().getInstruction() == "ime.rows.indices") {
    std::string source = input->parts.front();
    if (suffix != "i8m1")
      source = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + source + ")";
    const std::string vl = std::to_string(rows * 8);
    const std::string base = fresh("row_index");
    line("vuint8m1_t " + base + " = __riscv_vsrl_vx_u8m1(__riscv_vid_v_u8m1(" +
         vl + "), 3, " + vl + ");");
    for (int64_t part = 0; part < parts; ++part) {
      std::string index = base;
      if (part) {
        index = fresh("row_index");
        line("vuint8m1_t " + index + " = __riscv_vadd_vx_u8m1(" + base + ", " +
             std::to_string(part * rows) + ", " + vl + ");");
      }
      result.parts.push_back(fresh("fragment_rows"));
      line("vint8m1_t " + result.parts.back() + " = __riscv_vrgather_vv_i8m1(" +
           source + ", " + index + ", " + vl + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (operation.getLeaf().getInstruction() != "ime.rows.local")
    return fail(operation, "fragment broadcast has no selected local realization");
  std::string source = input->parts.front();
  if (suffix != "i8m1") source = "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + source + ")";
  for (int64_t part = 0; part < parts; ++part) {
    result.parts.push_back(fresh("fragment_rows"));
    line("vint8m1_t " + result.parts.back() + ";");
  }
  auto index = fresh("row_index");
  line("vuint8m1_t " + index + ";");
  line("__asm__ volatile(");
  line("  \"vsetvli zero, %[vl], e8, m1, ta, ma\\n\\tvid.v %[index]\\n\\tvsrl.vi %[index], %[index], 3\\n\\t\"");
  std::string outputs = "  : [index] \"=&vr\"(" + index + ")";
  for (int64_t part = 0; part < parts; ++part) {
    if (part)
      line("  \"vadd.vi %[index], %[index], " + std::to_string(rows) + "\\n\\t\"");
    line("  \"vrgather.vv %[part" + std::to_string(part) + "], %[source], %[index]\\n\\t\"");
    outputs += ", [part" + std::to_string(part) + "] \"=&vr\"(" + result.parts[part] + ")";
  }
  line(outputs);
  line("  : [source] \"vr\"(" + source + "), [vl] \"r\"((size_t)" + std::to_string(rows * 8) + ") : \"vl\", \"vtype\");");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentPointwise(riscv::IMEFragmentPointwiseOp operation) {
  auto lhs = bindings.lookup(operation.getLhs());
  auto rhs = materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  if (lhs.kind != Binding::Kind::Fragment || mlir::failed(rhs))
    return fail(operation, "fragment pointwise requires its selected numeric operands");
  auto fragment = operation.getResult().getType();
  const int64_t columns = fragment.getPacking().getColumnsPerTile();
  const std::string vl = std::to_string(fragment.getPacking().getRowsPerTile() * columns);
  if (lhs.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)))
    return fail(operation, "fragment pointwise lost its atom replicas");
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = fragment.getFamily().str();
  if (fragment.getRole() == "accumulator") {
    const std::string suffix = "i32m" + std::to_string(riscv::fragmentPartResourceGroups(fragment));
    for (size_t part = 0; part < lhs.parts.size(); ++part) {
      const bool scalar = rhs->kind == Binding::Kind::Scalar;
      if (!scalar && (rhs->kind != Binding::Kind::Fragment || rhs->parts.size() != lhs.parts.size()))
        return fail(operation, "accumulator pointwise lost its exact fragment/scalar mapping");
      const bool reuse = operation.getLhsPolicy() == "reuse";
      const std::string output = reuse ? lhs.parts[part] : fresh("ime_acc_pointwise");
      line((reuse ? "" : "vint32m" + std::to_string(riscv::fragmentPartResourceGroups(fragment)) + "_t ") + output + " = __riscv_v" +
           (operation.getKind() == "shl" ? "sll" : operation.getKind().str()) + (scalar ? "_vx_" : "_vv_") + suffix + "(" +
           lhs.parts[part] + ", " + (scalar ? rhs->scalar : rhs->parts[part]) + ", " + vl + ");");
      result.parts.push_back(output);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (operation.getLeaf().getInstruction() == ("ime.fragment." + operation.getKind() + ".grouped").str()) {
    if (rhs->kind != Binding::Kind::Fragment || rhs->parts.size() != lhs.parts.size())
      return fail(operation, "grouped fragment pointwise lost its matching fragment operand");
    const int64_t parts = riscv::fragmentPartCount(fragment);
    const std::string group = "m" + std::to_string(parts);
    auto pack = [&](llvm::ArrayRef<std::string> values) {
      auto packed = fresh("ime_pointwise_input");
      line("vint8" + group + "_t " + packed + " = __riscv_vundefined_i8" + group + "();");
      for (auto [part, value] : llvm::enumerate(values))
        line(packed + " = __riscv_vset_v_i8m1_i8" + group + "(" + packed + ", " +
             std::to_string(part) + ", " + value + ");");
      return packed;
    };
    const std::string left = pack(lhs.parts), right = pack(rhs->parts);
    line(left + " = __riscv_v" + operation.getKind().str() + "_vv_i8" + group + "(" +
         left + ", " + right + ", " + std::to_string(fragment.getPacking().getRowsPerTile() * columns * parts) + ");");
    for (int64_t part = 0; part < parts; ++part) {
      const bool reuse = operation.getLhsPolicy() == "reuse";
      result.parts.push_back(reuse ? lhs.parts[part] : fresh("ime_pointwise"));
      line((reuse ? "" : "vint8m1_t ") + result.parts.back() + " = __riscv_vget_v_i8" + group +
           "_i8m1(" + left + ", " + std::to_string(part) + ");");
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  for (int64_t part = 0; part < riscv::fragmentPartCount(fragment); ++part) {
  std::string expression;
  if (rhs->kind == Binding::Kind::Scalar) {
    expression = "__riscv_v" + operation.getKind().str() + "_vx_i8m1(" + lhs.parts[part] + ", " + rhs->scalar + ", " + vl + ")";
  } else if (rhs->kind == Binding::Kind::Fragment && rhs->fragmentFamily == lhs.fragmentFamily &&
             rhs->parts.size() == lhs.parts.size()) {
    expression = "__riscv_v" + operation.getKind().str() + "_vv_i8m1(" + lhs.parts[part] + ", " + rhs->parts[part] + ", " + vl + ")";
  } else if (rhs->kind == Binding::Kind::Vector && rhs->parts.size() == 1) {
    const std::string index = fresh("ime_row_index");
    const std::string row = fresh("ime_rows");
    line("vuint8m1_t " + index + " = __riscv_vdivu_vx_u8m1(__riscv_vid_v_u8m1(" + vl + "), " + std::to_string(columns) + ", " + vl + ");");
    if (part)
      line(index + " = __riscv_vadd_vx_u8m1(" + index + ", " +
           std::to_string(part * fragment.getPacking().getRowsPerTile()) + ", " + vl + ");");
    const std::string suffix = vectorSuffix(operation.getRhs());
    const std::string values = suffix == "i8m1" ? rhs->parts.front()
        : "__riscv_vlmul_ext_v_" + suffix + "_i8m1(" + rhs->parts.front() + ")";
    line("vint8m1_t " + row + " = __riscv_vrgather_vv_i8m1(" + values + ", " + index + ", " + vl + ");");
    expression = "__riscv_v" + operation.getKind().str() + "_vv_i8m1(" + lhs.parts[part] + ", " + row + ", " + vl + ")";
  } else {
    return fail(operation, "fragment pointwise has no selected scalar/row broadcast binding");
  }
  if (operation.getLhsPolicy() == "reuse") {
    result.parts.push_back(lhs.parts[part]);
    line(result.parts.back() + " = " + expression + ";");
  } else if (operation.getLhsPolicy() == "copy") {
    result.parts.push_back(fresh("ime_pointwise"));
    line("vint8m1_t " + result.parts.back() + " = " + expression + ";");
  } else {
    return fail(operation, "fragment pointwise requires a selected lhs reuse policy");
  }
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEFragmentMMA(riscv::IMEFragmentMMAOp operation) {
  Binding lhs = bindings.lookup(operation.getLhs());
  Binding rhs = bindings.lookup(operation.getRhs());
  Binding accumulator = bindings.lookup(operation.getAccumulator());
  auto family = operation.getResult().getType().getFamily();
  auto valid = [&](const Binding &binding, riscv::FragmentType type) {
    return binding.kind == Binding::Kind::Fragment &&
           binding.parts.size() == static_cast<size_t>(riscv::fragmentPartCount(type)) &&
           binding.fragmentFamily == family;
  };
  if (!valid(lhs, operation.getLhs().getType()) || !valid(rhs, operation.getRhs().getType()) ||
      !valid(accumulator, operation.getAccumulator().getType()))
    return fail(operation, "IME MMA operands do not match the selected fragment leaf");
  auto freshName = [&](llvm::StringRef prefix) { return fresh(prefix); };
  auto emitLine = [&](llvm::StringRef text) { line(text.str()); };
  Binding result;
  result.kind = Binding::Kind::Fragment;
  result.fragmentFamily = family.str();
  auto output = emitFragmentMMA(operation, lhs.parts, rhs.parts,
                                accumulator.parts, {freshName, emitLine});
  if (mlir::failed(output))
    return mlir::failure();
  llvm::append_range(result.parts, *output);
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIMEUnpack(riscv::IMEUnpackOp operation) {
  Binding input = bindings.lookup(operation.getInput());
  auto fragment = operation.getInput().getType();
  if (input.kind != Binding::Kind::Fragment ||
      input.parts.size() != static_cast<size_t>(riscv::fragmentPartCount(fragment)) ||
      input.fragmentFamily != fragment.getFamily())
    return fail(operation, "IME unpack input has no typed fragment binding");
  Binding result;
  result.kind = Binding::Kind::Vector;
  auto layout = operation.getResult().getType().getLayout();
  const std::string sourceSuffix = "i32m" + std::to_string(riscv::fragmentPartResourceGroups(fragment));
  const std::string suffix = vectorSuffix(operation.getResult());
  if (operation.getLeaf().getInstruction() == "spacemit-ime2-vpack-rows") {
    const int64_t rows = fragment.getShape()[0];
    const int64_t atomRows = fragment.getPacking().getRowsPerTile();
    if (fragment.getShape()[1] == 16) {
      for (int64_t atom = 0; atom < rows / atomRows; ++atom) {
        for (int64_t group = 0; group < (atomRows + 3) / 4; ++group) {
          const std::string pair = fresh("ime_row_pairs");
          line("vint32m2_t " + pair + ";");
          line("__asm__ volatile(\"vsetvli t0, zero, e32, m1, ta, ma\\n\\t\"");
          line("  \".insn r 0x2b, 2, 0x33, %0, %1, %2\\n\\t\"");
          line("  : \"=&vr\"(" + pair + ") : \"vr\"(__riscv_vget_v_i32m2_i32m1(" + input.parts[atom * 2] + ", " + std::to_string(group) + ")), ");
          line("    \"vr\"(__riscv_vget_v_i32m2_i32m1(" + input.parts[atom * 2 + 1] + ", " + std::to_string(group) + ")) : \"t0\", \"vl\", \"vtype\");");
          if (layout.getLaneFactors()[0] == 2) {
            for (int64_t half = 0; half < std::min<int64_t>(4, atomRows - group * 4) / 2; ++half)
              result.parts.push_back("__riscv_vget_v_i32m2_i32m1(" + pair + ", " + std::to_string(half) + ")");
            continue;
          }
          for (int64_t row = 0; row < std::min<int64_t>(4, atomRows - group * 4); ++row) {
            std::string value = "__riscv_vget_v_i32m2_i32m1(" + pair + ", " + std::to_string(row / 2) + ")";
            if (row % 2) value = "__riscv_vslidedown_vx_i32m1(" + value + ", 16, 16)";
            if (suffix != "i32m1") value = "__riscv_vlmul_trunc_v_i32m1_" + suffix + "(" + value + ")";
            const std::string name = fresh("ime_output_row");
            line(vectorType(operation.getResult()) + " " + name + " = " + value + ";");
            result.parts.push_back(name);
          }
        }
      }
      bindings[operation.getResult()] = std::move(result);
      return mlir::success();
    }
    for (int64_t atom = 0; atom < rows / atomRows; ++atom) {
    const int64_t activeRows = atomRows;
    for (int64_t group = 0; group < (activeRows + 3) / 4; ++group) {
      llvm::SmallVector<std::string> inputs;
      for (const auto &part : llvm::ArrayRef(input.parts).slice(atom * 4, 4))
        inputs.push_back("__riscv_vget_v_i32m2_i32m1(" + part + ", " + std::to_string(group) + ")");
      const std::string first = fresh("ime_join_columns"), second = fresh("ime_join_columns");
      line("vint32m2_t " + first + ", " + second + ";");
      llvm::SmallVector<std::string> rowPairs;
      for (int64_t half = 0; half < 2 && group * 4 + half * 2 < activeRows; ++half) {
        rowPairs.push_back(fresh("ime_output_rows"));
        line("vint32m2_t " + rowPairs.back() + ";");
      }
      line("__asm__ volatile(\"vsetvli t0, zero, e32, m1, ta, ma\\n\\t\"");
      line("  \".insn r 0x2b, 2, 0x33, %[first], %[in0], %[in1]\\n\\t\"");
      line("  \".insn r 0x2b, 2, 0x33, %[second], %[in2], %[in3]\\n\\t\"");
      // Only the first halves are named operands; the registered pair geometry
      // fixes the adjacent upper halves. Separate scalar names cannot express
      // that adjacency, so keep the upper-row operation in its own typed block.
      line("  \".insn r 0x2b, 3, 0x33, %[row0], %[first], %[second]\\n\\t\"");
      line("  : [first] \"=&vr\"(" + first + "), [second] \"=&vr\"(" + second +
           "), [row0] \"=&vr\"(" + rowPairs[0] + ") : [in0] \"vr\"(" + inputs[0] +
           "), [in1] \"vr\"(" + inputs[1] + "), [in2] \"vr\"(" + inputs[2] +
           "), [in3] \"vr\"(" + inputs[3] + ") : \"t0\", \"vl\", \"vtype\");");
      for (int64_t half = 0; half < int64_t(rowPairs.size()); ++half) {
        const std::string &pair = rowPairs[half];
        if (half) {
          line("__asm__ volatile(\"vsetvli t0, zero, e32, m1, ta, ma\\n\\t\"");
          line("  \".insn r 0x2b, 3, 0x33, %0, %1, %2\\n\\t\"");
          line("  : \"=&vr\"(" + pair + ") : \"vr\"(__riscv_vget_v_i32m2_i32m1(" + first + ", 1)),");
          line("    \"vr\"(__riscv_vget_v_i32m2_i32m1(" + second + ", 1)) : \"t0\", \"vl\", \"vtype\");");
        }
        for (int64_t row = 0; row < 2 && group * 4 + half * 2 + row < activeRows; ++row)
          result.parts.push_back("__riscv_vget_v_i32m2_i32m1(" + pair + ", " + std::to_string(row) + ")");
      }
    }
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (riscv::isDirectFragmentRVVHandoff(fragment, operation.getResult().getType())) {
    for (const std::string &part : input.parts)
      result.parts.push_back(sourceSuffix == suffix ? part
          : "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" + suffix + "(" + part + ")");
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (riscv::isGroupedFragmentRVVHandoff(fragment, operation.getResult().getType())) {
    const int64_t group = layout.getReplicaGroup();
    for (size_t part = 0; part < input.parts.size(); part += group) {
      const std::string name = fresh("ime_group");
      line(vectorType(operation.getResult()) + " " + name +
           " = __riscv_vlmul_ext_v_" + sourceSuffix + "_" + suffix + "(" + input.parts[part] + ");");
      for (int64_t offset = 1; offset < group; ++offset)
        line(name + " = __riscv_vset_v_" + sourceSuffix + "_" + suffix + "(" + name + ", " +
             std::to_string(offset) + ", " + input.parts[part + offset] + ");");
      result.parts.push_back(name);
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  const int64_t streams = streamPartCount(operation.getResult());
  const int64_t atomRows = fragment.getPacking().getRowsPerTile();
  const int64_t atomColumns = fragment.getPacking().getColumnsPerTile();
  const int64_t columnAtoms = fragment.getShape()[1] / atomColumns;
  const int64_t lanes = layout.getLaneFactors()[1];
  for (int64_t part = 0; part < vectorPartCount(operation.getResult()); ++part) {
    int64_t time = part % streams;
    int64_t replica = part / streams;
    llvm::SmallVector<int64_t, 2> coordinates(2);
    for (int64_t dimension = fragment.getShape().size() - 1; dimension >= 0; --dimension) {
      coordinates[dimension] = (time % layout.getTimeFactors()[dimension] *
          layout.getReplicaFactors()[dimension] + replica % layout.getReplicaFactors()[dimension]) *
          layout.getLaneFactors()[dimension];
      time /= layout.getTimeFactors()[dimension];
      replica /= layout.getReplicaFactors()[dimension];
    }
    const std::string name = fresh("ime_unpack");
    const std::string vl = partVL(operation.getResult(), part);
    for (int64_t inserted = 0; inserted < lanes;) {
      const int64_t column = coordinates[1] + inserted;
      const int64_t sourcePart = coordinates[0] / atomRows * columnAtoms + column / atomColumns;
      const int64_t offset = coordinates[0] % atomRows * atomColumns + column % atomColumns;
      const int64_t count = std::min(lanes - inserted, atomColumns - column % atomColumns);
      std::string expression = input.parts[sourcePart];
      if (offset)
        expression = "__riscv_vslidedown_vx_" + sourceSuffix + "(" + expression + ", " +
                     std::to_string(offset) + ", " + std::to_string(count) + ")";
      if (sourceSuffix != suffix)
        expression = "__riscv_vlmul_trunc_v_" + sourceSuffix + "_" + suffix + "(" + expression + ")";
      if (!inserted)
        line(vectorType(operation.getResult()) + " " + name + " = " + expression + ";");
      else
        line(name + " = __riscv_vslideup_vx_" + suffix + "_tu(" + name + ", " + expression +
             ", " + std::to_string(inserted) + ", " + vl + ");");
      inserted += count;
    }
    result.parts.push_back(name);
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
