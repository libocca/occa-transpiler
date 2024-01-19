#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include "oklt/core/config.h"
#include "oklt/core/attribute_manager/attribute_manager.h"

#include <any>

namespace oklt {

class ASTVisitor;


struct TranspilerSession {
  explicit TranspilerSession(TRANSPILER_TYPE backend);

  TRANSPILER_TYPE targetBackend;
  std::string transpiledCode;
  //INFO: add fields here
};

//INFO: could hold not the reference to the global AttributeManager
//      but hold the pointer to the AttributeManagerView
//      that is built for current session with set of interested attribute handlers
class SessionStage {
public:
  explicit SessionStage(TranspilerSession &globalSession,
                        clang::ASTContext &ctx);
  ~SessionStage() = default;

  clang::Rewriter& getRewriter();
  void setAstVisitor(ASTVisitor *visitor);
  ASTVisitor *getVisitor();
  [[nodiscard]] TRANSPILER_TYPE getBackend() const;
  AttributeManager &getAttrManager();
  [[nodiscard]] const AttributeManager &getAttrManager() const;
  void writeTranspiledSource();

  //TODO: might need better redesign by design patterns
  void setUserCtx(std::any userCtx);
  std::any &getUserCtx();
  [[nodiscard]] const std::any &getUserCtx() const;

protected:
  TranspilerSession &_globalSession;
  //INFO: might be redunt here
  clang::ASTContext &_ctx;
  clang::Rewriter _rewriter;
  ASTVisitor *_astVisitor;
  AttributeManager &_attrManager;
  std::any _userCtx;
};
}
