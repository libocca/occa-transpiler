#pragma once

#include <clang/AST/DeclBase.h>
#include <variant>

namespace oklt {

class SessionStage;

class SemanticASTVisitorBase {
 public:
  explicit SemanticASTVisitorBase(SessionStage& stage);
  virtual ~SemanticASTVisitorBase() = 0;

  virtual bool traverseTranslationUnit(clang::Decl* decl) = 0;

 protected:
  //TODO: tl::expected but with ErrorFired
  struct NoOKLAttrs {};
  struct ErrorFired {};

  using ValidationResult = std::variant<const clang::Attr*,
                                        NoOKLAttrs,
                                        ErrorFired>;

  ValidationResult validateAttribute(const clang::ArrayRef<const clang::Attr *> &attrs);
 protected:
  SessionStage& _stage;
};

}
