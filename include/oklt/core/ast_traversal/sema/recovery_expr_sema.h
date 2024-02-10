#pragma once

#include <oklt/core/ast_traversal/validate_attributes.h>
#include <clang/AST/AST.h>
#include <memory>

namespace oklt {

class SessionStage;

struct RecoveryExprSema {
  bool beforeTraverse(clang::RecoveryExpr *expr, SessionStage &stage);
  bool afterTraverse(clang::RecoveryExpr *expr, SessionStage &stage);
 private:
  std::unique_ptr<ValidatorResult> _validateResult;
};

}
