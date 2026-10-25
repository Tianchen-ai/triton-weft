#include "RISCVResidencyPlanning.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVPasses.h"
#include "Weft/Analysis/PhysicalDependence.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>

namespace {

using namespace weft;

void copySharedIdentity(mlir::Operation *source, mlir::Operation *target) {
  for (llvm::StringRef name : {"source_origin", "canonical_op"})
    if (auto attribute = source->getAttr(name)) target->setAttr(name, attribute);
}

bool isDenseLoadBacked(riscv::MaterializeOp materialize) {
  auto load = riscv_internal::stripRepresentationConversions(materialize.getInput())
                  .getDefiningOp<riscv::LoadOp>();
  if (!load)
    return false;
  auto source = mlir::dyn_cast<riscv::ValueType>(load.getResult().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  if (!source || !result || source.getElementType() != result.getElementType() ||
      source.getShape() != result.getShape() || source.getAxisIds() != result.getAxisIds())
    return false;
  auto encoding =
      mlir::dyn_cast<kernel::EncodingType>(load.getRegion().getType().getEncoding());
  return encoding && encoding.getKind() == "dense";
}

bool requiresAddressableReload(riscv::MaterializeOp materialize,
                               riscv::TargetAttr target) {
  auto value = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  if (!value || value.getLayout().getCarrier() != "rvv" ||
      !isDenseLoadBacked(materialize))
    return false;

  bool hasExtract = false;
  bool allExtracts = true;
  bool registerProjectionUnsupported = false;
  for (mlir::Operation *user : materialize.getResult().getUsers()) {
    auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
    if (!extract || extract.getInput() != materialize.getResult()) {
      allExtracts = false;
      continue;
    }
    hasExtract = true;
    int64_t selectedAxes = 0;
    for (mlir::Attribute selector : extract.getSelectors())
      if (mlir::cast<mlir::StringAttr>(selector).getValue() != "all")
        ++selectedAxes;
    registerProjectionUnsupported |= selectedAxes != 1;
  }
  if (!hasExtract || !allExtracts)
    return false;

  auto timeParts = riscv_internal::staticProduct(
      value.getLayout().getTimeFactors().asArrayRef());
  if (!timeParts)
    return true;
  const int64_t residentGroups =
      *timeParts * value.getLayout().getRegisterGroups();
  return registerProjectionUnsupported ||
         residentGroups > target.getVectorRegisters();
}

std::optional<riscv::LocalPackPlanAttr>
planEncodedLocalPack(mlir::Builder &builder,
                     riscv::MaterializeOp materialize) {
  auto input = mlir::dyn_cast<riscv::ValueType>(materialize.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  auto encoding = result
                      ? mlir::dyn_cast<kernel::EncodingType>(result.getElementType())
                      : kernel::EncodingType();
  auto load = riscv_internal::sourceLoad(materialize.getInput());
  if (!input || !result || !encoding || encoding.getKind() != "base" || !load ||
      result.getLayout().getCarrier() != "local" ||
      result.getShape().size() != 2 || result.getAxisIds().size() != 2) {
    return std::nullopt;
  }

  auto declaration =
      riscv_internal::findEncoding(materialize, encoding.getFamily());
  if (!declaration || declaration.getStorageBits() <= 0 ||
      declaration.getStorageBits() % 8 || declaration.getElements() <= 0) {
    return std::nullopt;
  }

  const int64_t rowAxis = result.getAxisIds()[0];
  const int64_t recordAxis = result.getAxisIds()[1];
  int64_t interleaveRows = 0;
  bool hasConsumer = false;
  llvm::SmallVector<mlir::Value> worklist{materialize.getResult()};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (mlir::Operation *user : current.getUsers()) {
      if (auto convert = mlir::dyn_cast<riscv::ConvertLayoutOp>(user);
          convert && convert.getInput() == current &&
          convert.getConversion().getEffect() == "pure") {
        worklist.push_back(convert.getResult());
        continue;
      }
      auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
      auto extracted =
          extract && extract.getInput() == current
              ? mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType())
              : riscv::ValueType();
      if (!extracted) {
        return std::nullopt;
      }
      auto row = llvm::find(extracted.getAxisIds().asArrayRef(), rowAxis);
      auto record = llvm::find(extracted.getAxisIds().asArrayRef(), recordAxis);
      if (row == extracted.getAxisIds().asArrayRef().end() ||
          record == extracted.getAxisIds().asArrayRef().end()) {
        return std::nullopt;
      }
      const int64_t rows = extracted.getShape()[static_cast<size_t>(
          row - extracted.getAxisIds().asArrayRef().begin())];
      if (rows <= 1 || (interleaveRows != 0 && interleaveRows != rows)) {
        return std::nullopt;
      }
      interleaveRows = rows;
      hasConsumer = true;
    }
  }
  if (!hasConsumer || interleaveRows <= 1) {
    return std::nullopt;
  }

  auto representedExtent = [&](size_t axis) -> std::optional<int64_t> {
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             result.getLayout().getTimeFactors(),
             result.getLayout().getLaneFactors(),
             result.getLayout().getReplicaFactors(),
             result.getLayout().getFragmentFactors(),
             result.getLayout().getLocalFactors()}) {
      const int64_t factor = factors[axis];
      if (factor <= 0 || extent > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      extent *= factor;
    }
    return extent;
  };
  auto localRows = representedExtent(0);
  auto localElements = representedExtent(1);
  const int64_t recordElements = declaration.getElements();
  const int64_t recordBytes = declaration.getStorageBits() / 8;
  if (!localRows || !localElements || *localRows % interleaveRows ||
      *localElements % recordElements) {
    return std::nullopt;
  }
  const int64_t records = *localRows * (*localElements / recordElements);
  if (records > std::numeric_limits<int64_t>::max() / recordBytes)
    return std::nullopt;
  const int64_t localBytes = records * recordBytes;
  auto kernel = materialize->getParentOfType<riscv::KernelOp>();
  if (!kernel || localBytes > kernel.getTarget().getMaxPrivateStackBytes()) {
    return std::nullopt;
  }

  return riscv::LocalPackPlanAttr::get(
      builder.getContext(), "interleave", rowAxis, recordAxis,
      interleaveRows, recordBytes, recordElements, localBytes);
}

} // namespace

mlir::LogicalResult weft::materializeRISCVSharedSupply(
    riscv::MaterializeOp operation, mlir::IRRewriter &rewriter) {
  if (operation.getPlacement() == "local")
    return operation.emitError("local shared supply requires its storage materializer");
  if (operation.getPlacement() == "reload") {
    auto load = riscv_internal::sourceLoad(operation.getInput());
    if (!load) return operation.emitError("reload supply requires an existing numerical read");
    rewriter.setInsertionPoint(operation);
    auto staged = riscv::StagedViewOp::create(rewriter, operation.getLoc(),
        load.getRegion().getType(), load.getRegion(), operation.getSchema(),
        operation.getOwnerDomainId(), operation.getBirthId(), operation.getLifetimeEndDomainId());
    copySharedIdentity(operation, staged);
    llvm::SmallVector<riscv::ExtractOp> extracts;
    for (mlir::Operation *user : operation.getResult().getUsers())
      if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
          extract && extract.getInput() == operation.getResult()) extracts.push_back(extract);
    for (riscv::ExtractOp extract : extracts) {
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      auto source = load.getRegion().getType();
      if (!result) return extract.emitError("reload projection requires one shaped physical result");
      llvm::SmallVector<int64_t> strides, origins;
      for (int64_t axis : result.getAxisIds().asArrayRef()) {
        auto found = llvm::find(source.getAxisIds().asArrayRef(), axis);
        if (found == source.getAxisIds().asArrayRef().end())
          return extract.emitError("reload projection introduced an axis absent from its memory descriptor");
        size_t index = found - source.getAxisIds().asArrayRef().begin();
        strides.push_back(source.getStrides()[index]);
        origins.push_back(source.getOrigins()[index]);
      }
      auto descriptor = riscv::MemDescType::get(rewriter.getContext(), source.getEncoding(),
          result.getShape(), result.getAxisIds(), riscv_internal::integers(rewriter, strides),
          riscv_internal::integers(rewriter, origins), source.getAlignment(), "slice", source.getAccess(),
          source.getAliasSet(), source.getLayoutIdentity(), source.getStorageBits(),
          source.getElements(), source.getInterleaveRows());
      rewriter.setInsertionPoint(extract);
      auto slice = riscv::SliceOp::create(rewriter, extract.getLoc(), descriptor,
          staged.getResult(), extract.getIndices(), extract.getSelectors());
      copySharedIdentity(extract, slice);
      auto form = extract.getAccess().getForm();
      auto leaf = riscv_internal::leaf(rewriter, "transfer", "load", ("rvv.load." + form).str(),
          ("rvv.load." + form).str(), 0, 0);
      auto projected = riscv::LoadOp::create(rewriter, extract.getLoc(), result, slice.getResult(),
          mlir::Value(), extract.getAccess(), leaf);
      copySharedIdentity(extract, projected);
      rewriter.replaceOp(extract, projected.getResult());
    }
    llvm::SmallVector<mlir::OpOperand *> uses;
    for (mlir::OpOperand &use : operation.getResult().getUses()) uses.push_back(&use);
    for (mlir::OpOperand *use : uses) {
      rewriter.setInsertionPoint(use->getOwner());
      auto cloned = riscv::LoadOp::create(rewriter, operation.getLoc(), operation.getResult().getType(),
          staged.getResult(), mlir::Value(), load.getAccess(), load.getLeaf());
      copySharedIdentity(load, cloned);
      use->set(cloned.getResult());
    }
    mlir::Value oldInput = operation.getInput();
    rewriter.eraseOp(operation);
    while (auto convert = oldInput.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (!convert.getResult().use_empty() || convert.getConversion().getEffect() != "pure") break;
      oldInput = convert.getInput();
      rewriter.eraseOp(convert);
    }
    if (load.getResult().use_empty()) rewriter.eraseOp(load);
    return mlir::success();
  }
  rewriter.setInsertionPoint(operation);
  mlir::Value input = operation.getInput();
  if (input.getType() != operation.getResult().getType()) {
    auto source = mlir::dyn_cast<riscv::ValueType>(input.getType());
    auto target = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (!source || !target)
      return operation.emitError("register materialization cannot convert a non-shaped representation");
    auto conversion = riscv::ConvertLayoutOp::create(rewriter, operation.getLoc(), target, input,
        riscv_internal::layoutConversion(rewriter, source.getLayout(), target.getLayout()),
        riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
    copySharedIdentity(operation, conversion);
    input = conversion.getResult();
  }
  auto materialized = riscv::RegisterMaterializeOp::create(rewriter, operation.getLoc(),
      operation.getResult().getType(), input, operation.getOwnerDomainId(), operation.getBirthId(),
      operation.getLifetimeEndDomainId(), "share");
  copySharedIdentity(operation, materialized);
  rewriter.replaceOp(operation, materialized.getResult());
  return mlir::success();
}

mlir::LogicalResult weft::planRISCVResidency(mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  bool invalid = false;
  module.walk([&](riscv::MaterializeOp materialize) {
    if (materialize.getPlacement() != "shared") {
      materialize.emitError(
          "residency planning requires one unresolved shared materialization");
      invalid = true;
      return;
    }
    materialize.setLocalPackPlanAttr({});
    if (auto plan = planEncodedLocalPack(builder, materialize)) {
      materialize.setPlacementAttr(builder.getStringAttr("local"));
      materialize.setSchemaAttr(builder.getStringAttr("encoded-interleave"));
      materialize.setLocalPackPlanAttr(*plan);
      return;
    }
    if (auto value =
            mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
        value && value.getLayout().getCarrier() == "local" &&
        !mlir::isa<kernel::EncodingType>(value.getElementType())) {
      materialize.setPlacementAttr(builder.getStringAttr("local"));
      return;
    }
    auto kernel = materialize->getParentOfType<riscv::KernelOp>();
    if (kernel && requiresAddressableReload(materialize, kernel.getTarget())) {
      auto load = riscv_internal::sourceLoad(materialize.getInput());
      // A stage is a saved numerical value. Addressable supply may repeat its
      // original read only when every consumer remains in that read epoch,
      // including writes after a use in a repeated child region.
      if (load && llvm::all_of(materialize.getResult().getUsers(),
          [&](mlir::Operation *user) { return !riscv::readCrossesWrite(load, user); }))
        materialize.setPlacementAttr(builder.getStringAttr("reload"));
    }
  });
  return mlir::failure(invalid);
}

namespace {
using namespace mlir;

struct ReadWork { uint64_t bytes = 0, transfers = 0; };
using FrequencyKey = SmallVector<std::string>;
using ReadKey = std::pair<FrequencyKey, std::string>;
using ReadPolynomial = std::map<ReadKey, ReadWork>;
using WorkPolynomial = std::map<FrequencyKey, uint64_t>;
using OpaqueWork = std::map<ReadKey, uint64_t>;
struct ProgramWork {
  WorkPolynomial instructions;
  OpaqueWork opaque;
};
struct SupplyEstimate {
  ReadPolynomial work;
  ProgramWork program;
  int64_t registerPeak;
};

// A frequency symbol names the actual SCF bounds, not the loop's position in
// a clone. Adding a preparation loop cannot accidentally rename another loop
// to the baseline symbol. No guessed trip counts or branch probabilities.
std::optional<std::string> indexIdentity(Value value, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  std::string text;
  llvm::raw_string_ostream out(text);
  out << value.getType() << ':';
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (!isa<riscv::KernelOp>(argument.getOwner()->getParentOp())) return std::nullopt;
    out << "argument" << argument.getArgNumber();
    return text;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumRegions() || operation->getNumResults() != 1 ||
      !isMemoryEffectFree(operation)) return std::nullopt;
  out << operation->getName();
  if (auto properties = operation->getPropertiesAsAttribute()) out << properties;
  out << '(';
  for (Value input : operation->getOperands()) {
    auto identity = indexIdentity(input, budget);
    if (!identity) return std::nullopt;
    out << *identity << ';';
  }
  out << ')';
  return text;
}

std::optional<FrequencyKey> frequencyKey(const analysis::ExecutionMultiplicity &frequency) {
  FrequencyKey key;
  for (Operation *operation : frequency.dynamicLoops) {
    auto loop = dyn_cast<scf::ForOp>(operation);
    if (!loop) return std::nullopt;
    unsigned budget = 128;
    auto lower = indexIdentity(loop.getLowerBound(), budget);
    auto upper = indexIdentity(loop.getUpperBound(), budget);
    auto step = indexIdentity(loop.getStep(), budget);
    if (!lower || !upper || !step) return std::nullopt;
    key.push_back((loop.getUnsignedCmp() ? "unsigned(" : "signed(") +
        *lower + ";" + *upper + ";" + *step + ")");
  }
  llvm::sort(key);
  return key;
}

ArrayAttr frequencyAttributes(Builder &b, ArrayRef<std::string> symbols) {
  SmallVector<Attribute> names;
  for (const auto &symbol : symbols) names.push_back(b.getStringAttr(symbol));
  return b.getArrayAttr(names);
}

// Structural work, not a CPU-cycle prediction. Unpriced operations retain their
// full typed signature and frequency. They may cancel only against the same
// physical work; a new opaque operation never receives a zero cost.
std::optional<uint64_t> instructionWork(Operation *operation) {
  if (operation->hasTrait<OpTrait::IsTerminator>() || riscv::isCoordinateIdentity(operation) ||
      isa<arith::ConstantOp, riscv::ConstantOp, riscv::SymbolOp, riscv::MemoryViewOp,
          riscv::LocalViewOp, riscv::LocalBindOp, riscv::StagedViewOp>(operation)) return 0;
  if (isa<riscv::ExtentOp, riscv::ReaxisOp>(operation)) return 0;
  // Slice offsets are ordinary scalar address arithmetic. Index/domain values
  // are already explicit SSA; charge each nontrivial axis translation here.
  if (auto slice = dyn_cast<riscv::SliceOp>(operation)) return 2 * slice.getIndices().size();
  if (auto view = dyn_cast<riscv::SubviewOp>(operation)) return 2 * view.getOffsets().size();
  if (operation->getName().getDialectNamespace() == "arith" &&
      llvm::all_of(operation->getResultTypes(), [](Type type) { return type.isIntOrIndexOrFloat(); })) return 1;
  if (auto materialize = dyn_cast<riscv::RegisterMaterializeOp>(operation)) {
    auto type = dyn_cast<riscv::ValueType>(materialize.getInput().getType());
    if (type && type.getElementType().isIntOrFloat() && type.getLayout().getCarrier() == "rvv") return 0;
  }
  bool simple = isa<riscv::LoadOp, riscv::StoreOp, riscv::BinaryOp, riscv::CompareOp,
                    riscv::SelectOp, riscv::CastOp, riscv::BitcastOp,
                    riscv::WidenOp, riscv::NarrowOp, riscv::RegisterMaterializeOp>(operation);
  if (!simple) return std::nullopt;
  // Each selected numeric carrier is at least one issue; wider LMULs carry
  // proportionally more physical work. Compound/decode/packing operations are
  // opaque until their actual instruction relation is known, never guessed.
  Value value = operation->getNumResults() ? operation->getResult(0) : operation->getOperand(0);
  auto type = dyn_cast<riscv::ValueType>(value.getType());
  if (!type) return value.getType().isIntOrIndexOrFloat() ? std::optional<uint64_t>(1) : std::nullopt;
  if (!type.getElementType().isIntOrFloat()) return std::nullopt;
  auto coordinates = riscv::RegisterCoordinates::get(type.getLayout());
  if (!coordinates || coordinates->parts() < 0) return std::nullopt;
  uint64_t parts = coordinates->parts();
  uint64_t width = std::max<int64_t>(1, type.getLayout().getRegisterGroups());
  if (parts > uint64_t(std::numeric_limits<int64_t>::max()) / width) return std::nullopt;
  return parts * width;
}

std::string physicalWorkSignature(Operation *operation) {
  std::string text;
  llvm::raw_string_ostream out(text);
  out << operation->getName() << '(';
  for (Type type : operation->getOperandTypes()) out << type << ';';
  out << ")->(";
  for (Type type : operation->getResultTypes()) out << type << ';';
  out << ')';
  for (Value operand : operation->getOperands()) {
    Attribute constant;
    if (matchPattern(operand, m_Constant(&constant))) out << "constant:" << constant;
    else out << "variable";
  }
  if (auto properties = operation->getPropertiesAsAttribute()) out << properties;
  // Leaf/implementation/layout/effect attributes determine real work. Source
  // locations are provenance, and cannot establish an instruction equivalence.
  for (NamedAttribute attribute : operation->getAttrs())
    if (attribute.getName() != "source_origin" && attribute.getName() != "canonical_op")
      out << attribute.getName() << '=' << attribute.getValue();
  return text;
}

std::optional<ProgramWork> programWork(Block *scope, std::string &rejection) {
  ProgramWork result;
  auto add = [&](uint64_t &total, uint64_t work, uint64_t frequency) {
    const uint64_t limit = std::numeric_limits<int64_t>::max();
    if (work && frequency > (limit - total) / work) return false;
    total += work * frequency;
    return true;
  };
  bool complete = true;
  scope->walk([&](Operation *operation) {
    if (!complete) return;
    auto frequency = analysis::executionMultiplicityInBlock(operation, scope);
    if (!frequency) {
      rejection = "physical work has conditional or opaque execution frequency";
      complete = false;
      return;
    }
    auto symbols = frequencyKey(*frequency);
    if (!symbols) {
      complete = false; rejection = "physical work has an unrepresentable symbolic frequency"; return;
    }
    auto key = *symbols;
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      // Setup is per loop invocation; compare/advance are per real iteration.
      complete = add(result.instructions[key], 1, frequency->fixed);
      if (auto trips = analysis::staticTripCount(loop)) {
        if (*trips > uint64_t(std::numeric_limits<int64_t>::max()) / 2) complete = false;
        else complete &= add(result.instructions[key], 2 * *trips, frequency->fixed);
      } else {
        auto iterations = *frequency;
        iterations.dynamicLoops.push_back(loop);
        auto symbol = frequencyKey(iterations);
        if (!symbol) complete = false;
        else complete &= add(result.instructions[*symbol], 2, frequency->fixed);
      }
    } else if (auto work = instructionWork(operation)) {
      complete = add(result.instructions[key], *work, frequency->fixed);
    } else {
      complete = add(result.opaque[{key, physicalWorkSignature(operation)}], 1, frequency->fixed);
    }
    if (!complete) rejection = "physical work exceeds its bounded coefficient representation";
  });
  return complete ? std::optional<ProgramWork>(result) : std::nullopt;
}

template <typename Key>
bool noAdditionalWork(const std::map<Key, uint64_t> &after,
                      const std::map<Key, uint64_t> &before) {
  for (const auto &[key, amount] : after) {
    auto found = before.find(key);
    if (amount > (found == before.end() ? 0 : found->second)) return false;
  }
  return true;
}

// Unknown trip counts stay independent symbols; no "inner loop is large"
// assumption can amortize preparation or justify an extra read. The separate
// whole-program ledger below proves whether non-read work can be cancelled.
std::optional<ReadPolynomial> readWork(Block *scope, std::string &rejection) {
  ReadPolynomial result;
  bool complete = true;
  auto reject = [&](Operation *operation, StringRef reason) {
    complete = false;
    if (rejection.empty()) {
      llvm::raw_string_ostream stream(rejection);
      stream << operation->getName() << ": " << reason;
    }
  };
  scope->walk([&](Operation *operation) {
    if (isa<riscv::LoadOp>(operation) || riscv::isCoordinateIdentity(operation) ||
        operation->hasTrait<OpTrait::IsTerminator>() || isMemoryEffectFree(operation)) return;
    auto memory = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!memory) {
      if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        reject(operation, "payload effects are not described");
      return;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    memory.getEffects(effects);
    // Specialized panel/fragment/atomic reads must acquire their own cost
    // relation. They cannot disappear from a generic-load comparison as zero.
    if (llvm::any_of(effects, [](const auto &effect) { return isa<MemoryEffects::Read>(effect.getEffect()); }))
      reject(operation, "read has no transfer cost relation");
  });
  scope->walk([&](riscv::LoadOp load) {
    if (load.getResult().use_empty()) return;
    auto type = dyn_cast<riscv::ValueType>(load.getResult().getType());
    if (!type || !type.getElementType().isIntOrFloat() ||
        type.getLayout().getValidity() != "full") {
      reject(load, "read has no complete numeric active domain"); return;
    }
    auto elements = riscv_internal::staticProduct(type.getShape().asArrayRef());
    auto carriers = riscv::rvvCarrierCount(type.getLayout());
    auto frequency = analysis::executionMultiplicityInBlock(load, scope);
    if (!elements || !carriers || !frequency || *elements < 0 || *carriers < 0) {
      reject(load, "read has no bounded carrier or execution frequency"); return;
    }
    auto key = frequencyKey(*frequency);
    if (!key) { reject(load, "read has an unrepresentable symbolic frequency"); return; }
    auto access = load.getAccess();
    if (!access || access.getForm() == "unassigned") {
      reject(load, "read has no selected memory form"); return;
    }
    // A strided/indexed issue cannot borrow the cost of a contiguous load.
    // Compare each access class independently until a target model can price
    // their relative costs; fewer bytes alone cannot justify a new gather.
    auto &term = result[{*key, access.getForm().str()}];
    const unsigned width = (type.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    const uint64_t limit = std::numeric_limits<int64_t>::max();
    if (uint64_t(*elements) > limit / width) {
      reject(load, "read byte count exceeds the cost representation"); return;
    }
    uint64_t bytesPerRead = uint64_t(*elements) * width;
    if ((bytesPerRead && frequency->fixed > (limit - term.bytes) / bytesPerRead) ||
        (*carriers && frequency->fixed > (limit - term.transfers) / uint64_t(*carriers))) {
      reject(load, "dynamic transfer coefficient exceeds the cost representation"); return;
    }
    term.bytes += bytesPerRead * frequency->fixed;
    term.transfers += uint64_t(*carriers) * frequency->fixed;
  });
  return complete ? std::optional<ReadPolynomial>(result) : std::nullopt;
}

bool reducesTransfers(const ReadPolynomial &candidate, const ReadPolynomial &baseline) {
  bool less = false;
  for (const auto &[key, after] : candidate) {
    auto found = baseline.find(key);
    ReadWork before = found == baseline.end() ? ReadWork{} : found->second;
    if (after.bytes > before.bytes || after.transfers > before.transfers) return false;
    less |= after.bytes < before.bytes || after.transfers < before.transfers;
  }
  for (const auto &[key, before] : baseline)
    if (!candidate.count(key)) less |= before.bytes != 0 || before.transfers != 0;
  return less;
}

std::optional<SupplyEstimate> estimateSupply(riscv::MaterializeOp source, bool reload,
                                            std::string &rejection, bool &invalid) {
  auto kernel = source->getParentOfType<riscv::KernelOp>();
  IRMapping mapping;
  OwningOpRef<ModuleOp> trial(cast<ModuleOp>(kernel->getParentOfType<ModuleOp>()->clone(mapping)));
  auto trialKernel = cast<riscv::KernelOp>(mapping.lookup(kernel.getOperation()));
  auto selected = cast<riscv::MaterializeOp>(mapping.lookup(source.getOperation()));
  Block *scope = selected->getBlock();
  std::optional<int64_t> levelId;
  if (auto level = dyn_cast<riscv::LoopOp>(scope->getParentOp()))
    levelId = level.getDomain().getType().getDomainId();
  if (reload) selected.setPlacementAttr(StringAttr::get(source.getContext(), "reload"));
  ScopedDiagnosticHandler diagnostics(source.getContext(), [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(rejection);
    diagnostic.print(stream);
    return success();
  });
  // Shared births and their owning Levels must cross the same lowering
  // boundary together. Verifying a half-expanded stage/Level pair is invalid.
  PassManager prepare(source.getContext());
  prepare.addPass(createPrepareRISCVCompositesPass());
  if (failed(prepare.run(*trial))) { invalid = true; return {}; }
  if (levelId) {
    scope = nullptr;
    unsigned matches = 0;
    trialKernel.walk([&](scf::ForOp loop) {
      auto level = loop->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
      if (level && level.getDomainId() == *levelId) { scope = loop.getBody(); ++matches; }
    });
    if (matches != 1) {
      rejection = "owning Level no longer has one physical body";
      invalid = true;
      return {};
    }
  }
  // Analyze the resulting graph, including independent live values and the
  // other consumers of the original read. Do not subtract an assumed buffer.
  if (failed(verify(*trial))) { invalid = true; return {}; }
  auto pressure = riscv_internal::physicalRegisterPressure(trialKernel);
  auto work = readWork(scope, rejection);
  auto program = programWork(scope, rejection);
  if (!pressure || !work || !program) {
    if (!pressure) rejection = "unknown register residency";
    return {};
  }
  return SupplyEstimate{*work, *program, pressure->total()};
}

ArrayAttr workAttributes(Builder &b, const ReadPolynomial &work) {
  SmallVector<Attribute> terms;
  for (const auto &[key, term] : work) {
    terms.push_back(b.getDictionaryAttr({
        b.getNamedAttr("dynamic_loops", frequencyAttributes(b, key.first)),
        b.getNamedAttr("memory_form", b.getStringAttr(key.second)),
        b.getNamedAttr("read_bytes", b.getI64IntegerAttr(term.bytes)),
        b.getNamedAttr("transfers", b.getI64IntegerAttr(term.transfers))}));
  }
  return b.getArrayAttr(terms);
}

DictionaryAttr programAttributes(Builder &b, const ProgramWork &work) {
  SmallVector<Attribute> instructions, opaque;
  for (const auto &[key, amount] : work.instructions) {
    instructions.push_back(b.getDictionaryAttr({
        b.getNamedAttr("dynamic_loops", frequencyAttributes(b, key)),
        b.getNamedAttr("instruction_work", b.getI64IntegerAttr(amount))}));
  }
  for (const auto &[key, amount] : work.opaque) {
    opaque.push_back(b.getDictionaryAttr({
        b.getNamedAttr("dynamic_loops", frequencyAttributes(b, key.first)),
        b.getNamedAttr("physical_work", b.getStringAttr(key.second)),
        b.getNamedAttr("instances", b.getI64IntegerAttr(amount))}));
  }
  return b.getDictionaryAttr({b.getNamedAttr("instructions", b.getArrayAttr(instructions)),
                              b.getNamedAttr("unpriced", b.getArrayAttr(opaque))});
}

class SelectSharedSupply : public PassWrapper<SelectSharedSupply, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectSharedSupply)
  StringRef getArgument() const override { return "weft-riscv-select-shared-supply"; }
  StringRef getDescription() const override {
    return "Compare complete shared-read and projected-read graphs with frequency and residency";
  }
  void runOnOperation() override {
    Builder b(&getContext());
    bool invalid = false;
    getOperation().walk([&](riscv::KernelOp kernel) {
      if (kernel.getResourcesMaterialized()) return;
      SmallVector<Attribute> decisions;
      if (auto previous = kernel->getAttrOfType<ArrayAttr>("weft.riscv.residency_choices"))
        llvm::append_range(decisions, previous);
      kernel.walk([&](riscv::MaterializeOp materialize) {
        if (invalid) return;
        if (materialize.getPlacement() != "shared" || !isDenseLoadBacked(materialize)) return;
        auto input = dyn_cast<riscv::ValueType>(materialize.getResult().getType());
        auto load = riscv_internal::sourceLoad(materialize.getInput());
        if (!input || input.getLayout().getCarrier() != "rvv" || !load ||
            load->getBlock() != materialize->getBlock() || load.getSnapshotStorage() ||
            materialize.getResult().use_empty()) return;
        bool legal = llvm::all_of(materialize.getResult().getUsers(), [&](Operation *user) {
          auto extract = dyn_cast<riscv::ExtractOp>(user);
          auto output = extract ? dyn_cast<riscv::ValueType>(extract.getResult().getType()) : riscv::ValueType();
          return extract && output && extract.getInput() == materialize.getResult() &&
              !riscv::readCrossesWrite(load, extract) &&
              llvm::all_of(output.getAxisIds().asArrayRef(), [&](int64_t axis) {
                return llvm::is_contained(load.getRegion().getType().getAxisIds().asArrayRef(), axis);
              });
        });
        if (!legal) return;
        std::string beforeFailure, afterFailure;
        auto before = estimateSupply(materialize, false, beforeFailure, invalid);
        auto after = invalid ? std::optional<SupplyEstimate>() :
            estimateSupply(materialize, true, afterFailure, invalid);
        if (invalid) {
          materialize.emitError("shared supply materialization violated its physical contract: ")
              << (afterFailure.empty() ? beforeFailure : afterFailure);
          return;
        }
        bool selected = before && after && after->registerPeak <= before->registerPeak &&
            after->registerPeak <= kernel.getTarget().getVectorRegisters() &&
            noAdditionalWork(after->program.instructions, before->program.instructions) &&
            noAdditionalWork(after->program.opaque, before->program.opaque) &&
            reducesTransfers(after->work, before->work);
        NamedAttrList record;
        record.set("birth", b.getI64IntegerAttr(materialize.getBirthId()));
        record.set("selected", b.getStringAttr(selected ? "reload" : "register"));
        record.set("evaluated_candidates", b.getI64IntegerAttr(2));
        record.set("cost_basis", b.getStringAttr("physical-program-work-with-symbolic-frequency-and-opaque-terms"));
        record.set("decision", b.getStringAttr(!before || !after ? "incomplete-domain-or-residency" :
            selected ? "no-additional-physical-work-and-residency" : "retain-shared-value"));
        if (!beforeFailure.empty()) record.set("register_unavailable", b.getStringAttr(beforeFailure));
        if (!afterFailure.empty()) record.set("reload_unavailable", b.getStringAttr(afterFailure));
        if (before) {
          record.set("register_read_work", workAttributes(b, before->work));
          record.set("register_peak", b.getI64IntegerAttr(before->registerPeak));
          record.set("register_program_work", programAttributes(b, before->program));
        }
        if (after) {
          record.set("reload_read_work", workAttributes(b, after->work));
          record.set("reload_peak", b.getI64IntegerAttr(after->registerPeak));
          record.set("reload_program_work", programAttributes(b, after->program));
        }
        decisions.push_back(record.getDictionary(&getContext()));
        if (selected) materialize.setPlacementAttr(b.getStringAttr("reload"));
      });
      if (!decisions.empty()) kernel->setAttr("weft.riscv.residency_choices", b.getArrayAttr(decisions));
    });
    if (invalid) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVSharedSupplyPass() {
  return std::make_unique<SelectSharedSupply>();
}
