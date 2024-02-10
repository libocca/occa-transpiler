#pragma once

#include <clang/AST/AST.h>
#include <oklt/core/ast_traversal/validate_attributes.h>
#include <memory>

namespace oklt {

class SessionStage;

struct KernelFunctionSema {
  bool beforeTraverse(clang::FunctionDecl* funDecl, SessionStage& stage);
  bool afterTraverse(clang::FunctionDecl* funDecl, SessionStage& stage);

 private:
  // TODO: make custom null-deleter and just hold const clang::Attr*
  std::unique_ptr<ValidatorResult> _validateResult;
};

}  // namespace oklt
