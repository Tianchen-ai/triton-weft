#include "Weft/Analysis/StorageLifetime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace weft::analysis;

bool weft::analysis::incomingControlValues(
    Value value, SmallVectorImpl<Value> &incoming) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    unsigned position = argument.getArgNumber();
    if (auto loop = dyn_cast<scf::ForOp>(owner)) {
      if (!position) return false;
      --position;
      incoming.push_back(loop.getInitArgs()[position]);
      incoming.push_back(loop.getBody()->getTerminator()->getOperand(position));
    } else if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
      if (argument.getOwner() == loop.getBeforeBody()) {
        incoming.push_back(loop.getInits()[position]);
        incoming.push_back(loop.getAfterBody()->getTerminator()->getOperand(position));
      } else if (argument.getOwner() == loop.getAfterBody()) {
        incoming.push_back(loop.getConditionOp().getArgs()[position]);
      } else return false;
    } else return false;
    return true;
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result) return false;
  Operation *owner = result.getOwner();
  unsigned position = result.getResultNumber();
  if (auto loop = dyn_cast<scf::ForOp>(owner)) {
    incoming.push_back(loop.getInitArgs()[position]);
    incoming.push_back(loop.getBody()->getTerminator()->getOperand(position));
  } else if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
    incoming.push_back(loop.getConditionOp().getArgs()[position]);
  } else if (auto branch = dyn_cast<scf::IfOp>(owner)) {
    incoming.push_back(branch.thenYield().getOperand(position));
    incoming.push_back(branch.elseYield().getOperand(position));
  } else if (auto select = dyn_cast<arith::SelectOp>(owner)) {
    incoming.push_back(select.getTrueValue());
    incoming.push_back(select.getFalseValue());
  } else return false;
  return true;
}

bool weft::analysis::outgoingControlAliases(
    OpOperand &use, SmallVectorImpl<Value> &aliases) {
  Operation *user = use.getOwner();
  unsigned index = use.getOperandNumber();
  if (auto loop = dyn_cast<scf::ForOp>(user)) {
    unsigned start = loop.getInitArgs().getBeginOperandIndex();
    if (index >= start) {
      index -= start;
      aliases.push_back(loop.getRegionIterArgs()[index]);
      aliases.push_back(loop.getResult(index));
    }
  } else if (auto loop = dyn_cast<scf::WhileOp>(user)) {
    aliases.push_back(loop.getBeforeBody()->getArgument(index));
  } else if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
    if (index) {
      auto loop = cast<scf::WhileOp>(condition->getParentOp());
      aliases.push_back(loop.getAfterBody()->getArgument(index - 1));
      aliases.push_back(loop.getResult(index - 1));
    }
  } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    Operation *parent = yield->getParentOp();
    if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
      aliases.push_back(loop.getBeforeBody()->getArgument(index));
    } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      aliases.push_back(loop.getRegionIterArgs()[index]);
      aliases.push_back(loop.getResult(index));
    } else if (isa<scf::IfOp>(parent)) {
      aliases.push_back(parent->getResult(index));
    } else return false;
  } else if (auto select = dyn_cast<arith::SelectOp>(user)) {
    if (index) aliases.push_back(select.getResult());
  } else if (!isa<scf::IfOp>(user)) {
    return false;
  }
  return true;
}

FailureOr<StorageUses> weft::analysis::collectStorageUses(
    Value root, llvm::function_ref<bool(Value)> contains,
    llvm::function_ref<bool(OpOperand &, SmallVectorImpl<Value> &)> follow) {
  StorageUses result;
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> seen;
  llvm::SmallPtrSet<Operation *, 32> users;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second) continue;
    result.aliases.push_back(value);
    if (value.use_empty()) continue;
    if (!contains(value)) return failure();
    for (OpOperand &use : value.getUses()) {
      if (!follow(use, pending)) return failure();
      if (users.insert(use.getOwner()).second)
        result.users.push_back(use.getOwner());
    }
  }
  return result;
}

StorageControl weft::analysis::structuredStorageControl(Operation *op) {
  if (isa<scf::ForOp, scf::WhileOp>(op)) return StorageControl::Repeated;
  if (isa<scf::IfOp>(op)) return StorageControl::Branch;
  return StorageControl::Opaque;
}

bool weft::analysis::storagePreservedBetween(
    Operation *read, Operation *consumer,
    llvm::function_ref<bool(Operation *)> preserves,
    llvm::function_ref<StorageControl(Operation *)> control) {
  if (read == consumer) return true;
  Block *block = read->getBlock();
  Operation *anchor = block->findAncestorOpInBlock(*consumer);
  if (!anchor || anchor == read || !read->isBeforeInBlock(anchor)) return false;
  for (Operation *op = read->getNextNode(); op != anchor; op = op->getNextNode())
    if (!preserves(op)) return false;
  while (anchor != consumer) {
    switch (control(anchor)) {
    case StorageControl::Repeated:
      return preserves(anchor);
    case StorageControl::Opaque:
      return false;
    case StorageControl::Branch:
      Block *nestedBlock = nullptr;
      Operation *nestedAnchor = nullptr;
      for (Region &region : anchor->getRegions())
        for (Block &nested : region)
          if (Operation *found = nested.findAncestorOpInBlock(*consumer)) {
            nestedBlock = &nested;
            nestedAnchor = found;
          }
      if (!nestedAnchor) return false;
      for (Operation &op : *nestedBlock) {
        if (&op == nestedAnchor) break;
        if (!preserves(&op)) return false;
      }
      anchor = nestedAnchor;
      break;
    }
  }
  return true;
}
