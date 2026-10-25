#include "Emitter.h"
#include "Weft/Dialect/RISCV/IR/FMAIssue.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::compileRVVIntegerMAC(riscv::RVVIntegerMACOp operation) {
  auto vector = materializeNumeric(operation.getVector(), bindings.lookup(operation.getVector()));
  auto multiplier = materializeNumeric(operation.getMultiplier(), bindings.lookup(operation.getMultiplier()));
  auto accumulator = materializeNumeric(operation.getAccumulator(), bindings.lookup(operation.getAccumulator()));
  if (mlir::failed(vector) || mlir::failed(multiplier) || mlir::failed(accumulator)) return mlir::failure();
  auto instruction = operation.getLeaf().getInstruction();
  const bool rows = instruction == "rvv.vmacc.row.vx";
  const auto rowParts = rows ? riscv::integerMACRowParts(operation.getVector().getType(),
      mlir::dyn_cast<riscv::ValueType>(operation.getMultiplier().getType()), operation.getResult().getType())
      : std::optional<llvm::SmallVector<std::pair<int64_t, int64_t>>>();
  const bool wideScalar = instruction == "rvv.vwmacc.vx" || instruction == "rvv.vwmaccu.vx";
  const bool scalarMultiplier = instruction == "rvv.vmacc.vx" || wideScalar || rows;
  const bool unsignedWide = instruction == "rvv.vwmaccu.vv";
  const bool mixed = instruction == "rvv.vwmaccsu.vv" || instruction == "rvv.vwmaccsu.vv.swap";
  if ((!scalarMultiplier && !unsignedWide && !mixed && instruction != "rvv.vwmacc.vv" && instruction != "rvv.vmacc.vv") ||
      vector->kind != Binding::Kind::Vector || accumulator->kind != Binding::Kind::Vector ||
      (rows ? !rowParts || rowParts->size() != accumulator->parts.size() :
              vector->parts.size() != accumulator->parts.size()) ||
      (rows ? multiplier->kind != Binding::Kind::ScalarTuple && multiplier->kind != Binding::Kind::Scalar :
       scalarMultiplier ? multiplier->kind != Binding::Kind::Scalar :
          multiplier->kind != Binding::Kind::Vector || multiplier->parts.size() != vector->parts.size()))
    return fail(operation, "integer MAC lost its selected numeric bindings");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < accumulator->parts.size(); ++part) {
    const int64_t vectorPart = rows ? (*rowParts)[part].first : part;
    const int64_t multiplierPart = rows ? (*rowParts)[part].second : part;
    if (vectorPart >= int64_t(vector->parts.size()) ||
        (rows && (multiplier->kind == Binding::Kind::Scalar ? multiplierPart != 0 :
                   multiplierPart >= int64_t(multiplier->parts.size()))))
      return fail(operation, "integer MAC row mapping exceeds its numeric bindings");
    const std::string scale = rows && multiplier->kind == Binding::Kind::ScalarTuple
        ? multiplier->parts[multiplierPart] : scalarMultiplier ? multiplier->scalar : multiplier->parts[part];
    auto name = fresh("integer_mac");
    std::string intrinsic = wideScalar
        ? (instruction == "rvv.vwmaccu.vx" ? "vwmaccu_vx_" : "vwmacc_vx_")
        : scalarMultiplier ? "vmacc_vx_" :
        instruction == "rvv.vmacc.vv" ? "vmacc_vv_" : mixed ? "vwmaccsu_vv_" :
        unsignedWide ? "vwmaccu_vv_" : "vwmacc_vv_";
    const std::string operands = instruction == "rvv.vwmaccsu.vv"
        ? vector->parts[vectorPart] + ", " + scale : scale + ", " + vector->parts[vectorPart];
    line(vectorType(operation.getResult()) + " " + name + " = __riscv_" + intrinsic +
         vectorSuffix(operation.getResult()) + "(" + accumulator->parts[part] + ", " +
         operands + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFMA(riscv::FMAOp operation) {
  auto lhs = materializeNumeric(operation.getLhs(), bindings.lookup(operation.getLhs()));
  auto rhs = materializeNumeric(operation.getRhs(), bindings.lookup(operation.getRhs()));
  auto acc = materializeNumeric(operation.getAccumulator(), bindings.lookup(operation.getAccumulator()));
  if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(acc)) return mlir::failure();
  auto instruction = operation.getLeaf().getInstruction();
  if (instruction == "scalar.fma") {
    if (lhs->kind != Binding::Kind::Scalar || rhs->kind != Binding::Kind::Scalar ||
        acc->kind != Binding::Kind::Scalar) return fail(operation, "scalar FMA requires scalar bindings");
    auto type = scalarCType(operation.getResult().getType());
    if (!type) return fail(operation, "scalar FMA has no C type");
    std::string builtin = operation.getResult().getType().isF32() ? "__builtin_fmaf" : "__builtin_fma";
    std::string left = operation.getKind() == "subtract_product" ? "-(" + lhs->scalar + ")" : lhs->scalar;
    std::string addend = operation.getKind() == "subtract_accumulator" ? "-(" + acc->scalar + ")" : acc->scalar;
    bindings[operation.getResult()] = scalar(materializeScalarRead(*type,
        builtin + "(" + left + ", " + rhs->scalar + ", " + addend + ")", "fma"));
    return mlir::success();
  }
  bool rows = instruction.ends_with(".row.vf");
  bool vf = instruction.ends_with(".vf");
  const bool overwriteLHS = instruction == "rvv.vfmadd.vf";
  const auto mnemonic = instruction.split('.').second.split('.').first;
  if (mnemonic != "vfmacc" && mnemonic != "vfnmsac" && mnemonic != "vfmsac" && !overwriteLHS)
    return fail(operation, "pointwise FMA has no selected arithmetic instruction");
  std::optional<riscv::FMAIssueGeometry> geometry;
  if (rows)
    geometry = riscv::FMAIssueGeometry::get(
        mlir::cast<riscv::ValueType>(operation.getLhs().getType()),
        operation.getRhs().getType(),
        mlir::cast<riscv::ValueType>(operation.getResult().getType()));
  if ((!vf && !instruction.ends_with(".vv")) ||
      lhs->kind != Binding::Kind::Vector || acc->kind != Binding::Kind::Vector ||
      lhs->parts.size() != acc->parts.size() ||
      (rows ? !geometry || (rhs->kind != Binding::Kind::ScalarTuple && rhs->kind != Binding::Kind::Scalar) :
       vf ? rhs->kind != Binding::Kind::Scalar :
            rhs->kind != Binding::Kind::Vector || rhs->parts.size() != acc->parts.size()))
    return fail(operation, "pointwise FMA lost its selected numeric bindings");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (size_t part = 0; part < acc->parts.size(); ++part) {
    int64_t row = rows ? geometry->multiplierParts[part] : 0;
    if (rows && (rhs->kind == Binding::Kind::Scalar ? row != 0 : row >= int64_t(rhs->parts.size())))
      return fail(operation, "pointwise FMA row mapping exceeds its scalar binding");
    std::string multiplier = rows && rhs->kind == Binding::Kind::ScalarTuple
        ? rhs->parts[row] : vf ? rhs->scalar : rhs->parts[part];
    std::string name = fresh("fma");
    line(vectorType(operation.getResult()) + " " + name + " = __riscv_" + mnemonic.str() + "_" +
         (vf ? "vf_" : "vv_") + vectorSuffix(operation.getResult()) + "(" +
         (overwriteLHS ? lhs->parts[part] : acc->parts[part]) +
         ", " + multiplier + ", " + (overwriteLHS ? acc->parts[part] : lhs->parts[part]) + ", " +
         partVL(operation.getResult(), part) + ");");
    result.parts.push_back(std::move(name));
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileRVVFMAIssue(riscv::RVVFMAIssueOp operation) {
  auto geometry = riscv::FMAIssueGeometry::get(operation.getVector().getType(),
      operation.getMultiplier().getType(), operation.getResult().getType());
  if (!geometry) return fail(operation, "FMA issue lost its verified numeric mapping");
  const auto vector = bindings.lookup(operation.getVector());
  const auto multiplier = bindings.lookup(operation.getMultiplier());
  const auto accumulator = bindings.lookup(operation.getAccumulator());
  if (vector.kind != Binding::Kind::Vector || accumulator.kind != Binding::Kind::Vector ||
      accumulator.parts.size() != size_t(geometry->parts()) ||
      (geometry->scalar && multiplier.kind != Binding::Kind::Scalar && multiplier.kind != Binding::Kind::ScalarTuple) ||
      (!geometry->scalar && multiplier.kind != Binding::Kind::Vector))
    return fail(operation, "FMA issue requires already materialized numeric operands");
  Binding result;
  result.kind = Binding::Kind::Vector;
  for (const std::string &part : accumulator.parts) {
    if (operation.getAccumulatorPolicy() == "reuse") result.parts.push_back(part);
    else {
      result.parts.push_back(fresh("fma_acc"));
      line(vectorType(operation.getResult()) + " " + result.parts.back() + " = " + part + ";");
    }
  }
  std::string instruction = geometry->widening ? "vfwmacc" : "vfmacc";
  instruction += geometry->scalar ? "_vf_" : "_vv_";
  instruction += vectorSuffix(operation.getResult());
  instruction += operation.getTail() == "agnostic" ? "(" : "_tu(";
  int64_t begin = 0;
  for (int64_t end : operation.getGroupEnds()) {
    std::string outputs;
    for (int64_t part = begin; part < end; ++part) {
      int64_t v = geometry->vectorParts[part], m = geometry->multiplierParts[part];
      if (v >= int64_t(vector.parts.size()) ||
          (multiplier.kind != Binding::Kind::Scalar && m >= int64_t(multiplier.parts.size())))
        return fail(operation, "FMA issue part mapping exceeds its numeric binding");
      mlir::Value length = operation.getLengths()[part];
      auto vl = bindings.lookup(length);
      if (vl.kind != Binding::Kind::Scalar)
        return fail(operation, "FMA issue VL requires an explicit scalar binding");
      std::string value = multiplier.kind == Binding::Kind::Scalar ? multiplier.scalar : multiplier.parts[m];
      line(result.parts[part] + " = __riscv_" + instruction + result.parts[part] +
           ", " + value + ", " + vector.parts[v] + ", " + vl.scalar + ");");
      if (!outputs.empty()) outputs += ", ";
      outputs += "\"+vr\"(" + result.parts[part] + ")";
    }
    // Finish these numeric results before subsequent independent loads can
    // enter the issue frontier. This is a compiler scheduling boundary, not
    // a hardware memory fence or a memory read/write. Leaving arithmetic as
    // typed intrinsics preserves the system compiler's exact VL/VTYPE state.
    line("__asm__ volatile(\"\" : " + outputs + " : : \"memory\");");
    begin = end;
  }
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
