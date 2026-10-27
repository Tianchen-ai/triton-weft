#include "Verification.h"

using namespace weft::kernel;
using namespace weft::kernel::detail;

mlir::LogicalResult KernelOp::verify() {
  mlir::Block &body = getBody().front();
  if (getArgNames().size() != body.getNumArguments() ||
      getArgAccess().size() != body.getNumArguments() ||
      getArgAliasSets().size() != body.getNumArguments())
    return emitOpError(
        "argument names, access, and alias sets must match kernel entry arguments");
  if (failed(verifyStringArray(*this, getArgNames(), "arg_names")) ||
      failed(verifyStringArray(*this, getArgAccess(), "arg_access")) ||
      failed(verifyStringArray(*this, getShapeSymbols(), "shape_symbols")))
    return mlir::failure();
  llvm::DenseSet<llvm::StringRef> names;
  for (auto [index, nameAttr] : llvm::enumerate(getArgNames())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !names.insert(name).second)
      return emitOpError("kernel argument names must be non-empty and unique");
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(getArgAccess()[index]).getValue();
    mlir::Type type = body.getArgument(index).getType();
    if (access != "none" && access != "read" && access != "write" &&
        access != "readwrite")
      return emitOpError("unknown kernel argument access mode");
    if (mlir::isa<ViewType, PointerType>(type)) {
      if (getArgAliasSets()[index] < 0)
        return emitOpError("View argument requires a non-negative alias set");
    } else if (access != "none" || getArgAliasSets()[index] != -1) {
      return emitOpError(
          "scalar argument has no View access effect or alias set");
    }
  }
  if (!mlir::isa<ReturnOp>(body.getTerminator()))
    return emitOpError("kernel body must end in weft_kernel.return");
  if (failed(weft::verifyStrideABI<ViewType>(*this, body, getArgNames(),
                                            getShapeSymbols(), getStridedArgsAttr())))
    return mlir::failure();

  bool typeFailure = false;
  auto verifyTypes = [&](mlir::Operation *owner, mlir::Block &block) {
    for (mlir::BlockArgument argument : block.getArguments()) {
      if (!isCanonicalProgramType(argument.getType())) {
        owner->emitOpError(
            "region argument type is outside the canonical Weft type set");
        typeFailure = true;
      } else if (failed(verifyEncodingInType(owner, argument.getType()))) {
        typeFailure = true;
      }
    }
  };
  verifyTypes(*this, body);
  getBody().walk([&](mlir::Operation *operation) {
    llvm::StringRef name = operation->getName().getStringRef();
    const bool standardCanonical =
        name == "arith.constant" || name == "arith.ceildivui" ||
        name == "scf.for" || name == "scf.if" || name == "scf.while" ||
        name == "scf.yield" || name == "scf.condition";
    if (operation->getDialect() != getOperation()->getDialect() &&
        !standardCanonical) {
      operation->emitOpError(
          "operation is outside the canonical Weft/arith/SCF contract");
      typeFailure = true;
    }
    for (mlir::Value operand : operation->getOperands())
      if (!isCanonicalProgramType(operand.getType())) {
        operation->emitOpError("operand type is outside the canonical Weft type set");
        typeFailure = true;
      }
    for (mlir::Value result : operation->getResults())
      if (!isCanonicalProgramType(result.getType())) {
        operation->emitOpError("result type is outside the canonical Weft type set");
        typeFailure = true;
      } else if (failed(verifyEncodingInType(operation, result.getType())))
        typeFailure = true;
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &nested : region)
        verifyTypes(operation, nested);
  });
  if (typeFailure)
    return mlir::failure();

  llvm::DenseSet<int64_t> localShapes;
  getBody().walk([&](LocalShapeOp shape) {
    if (!localShapes.insert(shape.getId()).second) {
      shape.emitOpError("local shape identities must be unique within a kernel");
      typeFailure = true;
    }
  });
  auto checkLocalShapeUses = [&](mlir::Operation *owner, mlir::Type type) {
    for (int64_t extent : logicalShape(type)) {
      if (extent >= -static_cast<int64_t>(getShapeSymbols().size())) continue;
      if (extent == std::numeric_limits<int64_t>::min() ||
          !weft::localShapeExtent<LocalShapeOp>(owner, -extent)) {
        owner->emitOpError("local shape reference escapes its SSA definition scope");
        typeFailure = true;
      }
    }
  };
  getBody().walk([&](mlir::Operation *op) {
    for (mlir::Type type : op->getOperandTypes()) checkLocalShapeUses(op, type);
    for (mlir::Type type : op->getResultTypes()) checkLocalShapeUses(op, type);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (auto argument : block.getArguments()) checkLocalShapeUses(op, argument.getType());
  });
  if (typeFailure) return mlir::failure();

  llvm::DenseSet<int64_t> domainIds;
  unsigned roots = 0;
  getBody().walk([&](mlir::Operation *operation) {
    if (auto root = mlir::dyn_cast<RootDomainOp>(operation)) {
      ++roots;
      domainIds.insert(root.getResult().getType().getDomainId());
      return;
    }
    if (auto domain = mlir::dyn_cast<DomainOp>(operation)) {
      if (!domainIds.insert(domain.getResult().getType().getDomainId()).second) {
        domain.emitOpError("domain identity must be unique within one kernel");
        typeFailure = true;
      }
      return;
    }
    if (auto level = mlir::dyn_cast<LevelOp>(operation))
      if (!level.getDomain().getDefiningOp<DomainOp>()) {
        level.emitOpError("Level domain must be produced by a canonical domain op");
        typeFailure = true;
      }
  });
  if (roots != 1) {
    emitOpError("kernel must contain exactly one canonical root domain");
    return mlir::failure();
  }
  if (typeFailure)
    return mlir::failure();

  auto rootArgument = [&](mlir::Value value) -> std::optional<unsigned> {
    while (mlir::Operation *producer = value.getDefiningOp()) {
      if (auto slice = mlir::dyn_cast<SliceOp>(producer)) {
        value = slice.getBase();
        continue;
      }
      if (auto subview = mlir::dyn_cast<SubviewOp>(producer)) {
        value = subview.getBase();
        continue;
      }
      if (auto field = mlir::dyn_cast<FieldOp>(producer)) {
        value = field.getOwner();
        continue;
      }
      return std::nullopt;
    }
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || argument.getOwner() != &body)
      return std::nullopt;
    return argument.getArgNumber();
  };
  getBody().walk([&](mlir::Operation *operation) {
    mlir::Value region;
    llvm::StringRef required;
    if (auto admit = mlir::dyn_cast<AdmitOp>(operation)) {
      region = admit.getRegion();
      required = "read";
    } else if (auto commit = mlir::dyn_cast<CommitOp>(operation)) {
      region = commit.getRegion();
      required = "write";
    } else if (auto atomic = mlir::dyn_cast<AtomicRMWOp>(operation)) {
      region = atomic.getRegion();
      required = "readwrite";
    } else {
      return;
    }
    auto index = rootArgument(region);
    if (!index)
      return;
    llvm::StringRef access =
        mlir::cast<mlir::StringAttr>(getArgAccess()[*index]).getValue();
    const bool allowed = required == "readwrite" ? access == "readwrite"
                         : required == "read"
                             ? access == "read" || access == "readwrite"
                             : access == "write" || access == "readwrite";
    if (!allowed) {
      operation->emitOpError()
          << required << " effect contradicts the kernel argument access mode";
      typeFailure = true;
    }
  });
  if (typeFailure)
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult HelperOp::verify() {
  if (getName().empty() || getBody().front().getNumArguments())
    return emitOpError("helper needs a source name and a closure body");
  bool found = false;
  getBody().walk([&](HelperReturnOp ret) {
    if (ret->getParentOfType<HelperOp>() == *this) found = true;
  });
  if (!found) return emitOpError("helper has no return path");
  return mlir::success();
}

mlir::LogicalResult HelperReturnOp::verify() {
  auto helper = (*this)->getParentOfType<HelperOp>();
  if (!helper || getValues().getTypes() != helper.getResultTypes())
    return emitOpError("helper return values must match their enclosing function signature");
  return mlir::success();
}

void HelperReturnOp::getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), FunctionControlResource::get());
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(getOperation()->getParentOp()))
    return emitOpError("return terminates only a kernel body");
  if (!getValues().empty())
    return emitOpError("current kernels return through commit, not SSA results");
  return mlir::success();
}

mlir::LogicalResult RootDomainOp::verify() {
  DomainType type = getResult().getType();
  if (type.getRelation() != "root" || type.getDomainId() != 0 ||
      type.getParentDomainId() != -1 || type.getAxisId() != 0)
    return emitOpError("root_domain must produce the canonical root domain");
  return mlir::success();
}

mlir::LogicalResult SymbolOp::verify() {
  if (getName().empty() || (getKind() != "shape" && getKind() != "source_auto"))
    return emitOpError("symbol is a named shape or source auto value");
  if (getKind() == "shape" && !getChoices().empty())
    return emitOpError("shape symbol cannot carry source-auto choices");
  if (llvm::any_of(getChoices(),
                   [](int64_t value) { return value <= 0; }))
    return emitOpError("source-auto choices must be positive");
  return mlir::success();
}

mlir::LogicalResult DomainOp::verify() {
  DomainType parent = getParent().getType();
  DomainType child = getResult().getType();
  if (child.getRelation() == "root")
    return emitOpError("child domain cannot be another root");
  if (child.getParentDomainId() != parent.getDomainId())
    return emitOpError("child domain must name its parent domain identity");
  if (child.getAxisId() == parent.getAxisId() &&
      child.getAxisName().split('.').first !=
          parent.getAxisName().split('.').first)
    return emitOpError("a refined logical axis must preserve its axis name");
  auto [axisBase, axisSerial] = child.getAxisName().rsplit('.');
  if (axisBase.empty() || axisSerial != std::to_string(child.getDomainId()))
    return emitOpError(
        "child domain axis name must carry its unique domain identity suffix");
  auto parentProducer = getParent().getDefiningOp();
  if (!mlir::isa_and_nonnull<RootDomainOp, DomainOp>(parentProducer))
    return emitOpError("domain parent must be a canonical root or child domain");
  auto multiplicity = getMultiplicity().getDefiningOp<mlir::arith::CeilDivUIOp>();
  if (!multiplicity || multiplicity.getLhs() != getExtent() ||
      multiplicity.getRhs() != getPartition())
    return emitOpError(
        "domain multiplicity must be ceildiv(extent, partition)");
  for (auto [value, name] :
       {std::pair(getExtent(), llvm::StringRef("extent")),
        std::pair(getPartition(), llvm::StringRef("partition"))}) {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
          integer && integer.getInt() <= 0)
        return emitOpError() << "domain " << name << " must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult LevelOp::verify() {
  if (getCarried().size() != getResults().size())
    return emitOpError("level carried operands and results must match");
  for (auto [operand, result] : llvm::zip(getCarried(), getResults()))
    if (operand.getType() != result.getType())
      return emitOpError("level carried operand/result types must match");
  DomainType domain = getDomain().getType();
  PointType point = PointType::get(getContext(), domain);
  mlir::Block &state = getStateBirths().front();
  mlir::Block &staged = getStagedBirths().front();
  mlir::Block &body = getBody().front();
  if (state.getNumArguments() != 1 || state.getArgument(0).getType() != point ||
      staged.getNumArguments() != 1 || staged.getArgument(0).getType() != point)
    return emitOpError("both births regions take the current level point");
  auto stateYield = mlir::dyn_cast<BirthsYieldOp>(state.getTerminator());
  auto stagedYield = mlir::dyn_cast<BirthsYieldOp>(staged.getTerminator());
  if (!stateYield || !stagedYield)
    return emitOpError("births regions must end in births_yield");
  for (mlir::Value value : stateYield.getValues())
    if (!value.getDefiningOp<NewOp>() ||
        value.getParentRegion() != &getStateBirths())
      return emitOpError("state_births may yield only values defined by new");
  for (mlir::Value value : stagedYield.getValues())
    if (!value.getDefiningOp<MaterializeOp>() ||
        value.getParentRegion() != &getStagedBirths())
      return emitOpError(
          "staged_births may yield only values defined by materialize");

  const size_t expectedArguments = 1 + getCarried().size() +
                                   stateYield.getValues().size() +
                                   stagedYield.getValues().size();
  if (body.getNumArguments() != expectedArguments ||
      body.getArgument(0).getType() != point)
    return emitOpError("level body signature does not match domain and births");
  size_t cursor = 1;
  for (mlir::Value carried : getCarried())
    if (body.getArgument(cursor++).getType() != carried.getType())
      return emitOpError("level body carried argument type mismatch");
  for (mlir::Value birth : stateYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("state birth type mismatch");
  for (mlir::Value birth : stagedYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("staged birth type mismatch");
  auto handoff = mlir::dyn_cast<HandoffOp>(body.getTerminator());
  if (!handoff || handoff.getValues().getTypes() != getResults().getTypes())
    return emitOpError("level body handoff must match level results");
  return mlir::success();
}

mlir::LogicalResult BirthsYieldOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || (getOperation()->getParentRegion() != &level.getStateBirths() &&
                 getOperation()->getParentRegion() != &level.getStagedBirths()))
    return emitOpError("births_yield terminates a level births region");
  return mlir::success();
}

mlir::LogicalResult HandoffOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || getOperation()->getParentRegion() != &level.getBody())
    return emitOpError("handoff terminates a level body");
  return mlir::success();
}
