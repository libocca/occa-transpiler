#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include "oklt/core/config.h"
#include "oklt/core/transpile_session/error_reporter.h"
#include "oklt/core/attribute_manager/attribute_manager.h"

namespace oklt {

class ASTVisitor;

class TranspileSession {
public:
  explicit TranspileSession(const TranspilerConfig &config,
                            clang::ASTContext &ctx);
  ~TranspileSession() = default;

  clang::Rewriter& getRewriter();
  void setAstVisitor(ASTVisitor *visitor);
  ASTVisitor *getVisitor();
  [[nodiscard]] TRANSPILER_TYPE getBackend() const;
  AttributeManager &getAttrManager();
  [[nodiscard]] const AttributeManager &getAttrManager() const;

  ErrorReporter &getErrorReporter();

  void writeTranspiledSource();

protected:
  //INFO: might be redunt here
  clang::ASTContext &_ctx;
  TranspilerConfig _transpileConfig;
  clang::Rewriter _rewriter;
  ErrorReporter _errorReporter;
  ASTVisitor *_astVisitor;
  AttributeManager &_attrManager;
};
}
