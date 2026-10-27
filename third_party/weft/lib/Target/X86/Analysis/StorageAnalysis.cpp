#include "Analysis/StorageAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace {
Value viewSource(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>()) return cast.getSource();
  if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
    return view.getViewSource();
  return {};
}

bool borrowedUse(OpOperand &use, SmallVectorImpl<Value> &aliases) {
  if (weft::analysis::outgoingControlAliases(use, aliases)) return true;
  Operation *user = use.getOwner();
  if (isa<memref::CastOp, ViewLikeOpInterface>(user)) {
    if (user->getNumResults() != 1 || viewSource(user->getResult(0)) != use.get())
      return false;
    aliases.push_back(user->getResult(0));
    return true;
  }
  if (isa<memref::LoadOp, memref::StoreOp, memref::DimOp,
          vector::LoadOp, vector::StoreOp, vector::GatherOp, vector::ScatterOp,
          vector::MaskedLoadOp, vector::MaskedStoreOp>(user)) return true;
  if (auto call = dyn_cast<func::CallOp>(user)) {
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    return function && function.isPrivate() && !function.isExternal() &&
           function.getArgAttr(use.getOperandNumber(), "weft.borrowed");
  }
  // Pointer extraction, unknown calls and returns can retain the address.
  return false;
}
}

FailureOr<weft::analysis::StorageUses> weft::x86::privateStorageUses(
    Value root, Operation *iteration) {
  if (!root.getDefiningOp<memref::AllocOp>() &&
      !root.getDefiningOp<memref::AllocaOp>()) return failure();
  DominanceInfo dominance;
  auto contains = [&](Value value) {
    if (value == root) return true;
    // A selected branch can hand its fresh buffer to the rest of the same
    // iteration. It may not retain that address on this iteration's back edge.
    // This explicit owner is only supplied by transformations preserving the
    // complete iteration, rather than promoting the allocation independently.
    if (iteration) {
      Operation *at = value.getDefiningOp();
      if (!at) at = value.getParentBlock()->getParentOp();
      if (!at || !iteration->isProperAncestor(at)) return false;
      for (Operation *scope = root.getDefiningOp()->getParentOp();
           scope && scope != iteration; scope = scope->getParentOp()) {
        if (analysis::structuredStorageControl(scope) != analysis::StorageControl::Repeated) continue;
        if (auto argument = dyn_cast<BlockArgument>(value);
            argument && argument.getOwner()->getParentOp() == scope) return false;
      }
      return true;
    }
    if (!root.getParentRegion()->isAncestor(value.getParentRegion())) return false;
    Operation *at = value.getDefiningOp();
    if (auto argument = dyn_cast<BlockArgument>(value))
      at = argument.getOwner()->getParentOp();
    // A back edge to the allocation's own iteration starts before its birth.
    return at && dominance.dominates(root, at);
  };
  return analysis::collectStorageUses(root, contains, borrowedUse);
}

bool weft::x86::isPrivateStorage(Value value) {
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> seen;
  bool foundAllocation = false;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!seen.insert(current).second) continue;
    if (current.getDefiningOp<memref::AllocOp>() ||
        current.getDefiningOp<memref::AllocaOp>()) {
      if (failed(privateStorageUses(current))) return false;
      foundAllocation = true;
    } else if (Value source = viewSource(current)) {
      pending.push_back(source);
    } else if (!analysis::incomingControlValues(current, pending)) {
      return false;
    }
  }
  return foundAllocation;
}
