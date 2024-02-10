#pragma once

#include <oklt/core/ast_traversal/validate_attributes.h>
#include <clang/AST/AST.h>
#include <memory>

namespace oklt {

class SessionStage;

struct ParamSema {
  bool beforeTraverse(clang::ParmVarDecl *paramDecl, SessionStage &stage);
  bool afterTraverse(clang::ParmVarDecl *paramDecl, SessionStage &stage);
 private:
  std::unique_ptr<ValidatorResult> _validateResult;
};

}
