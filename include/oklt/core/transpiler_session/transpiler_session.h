#pragma once

#include <clang/Frontend/CompilerInstance.h>
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
  explicit SessionStage(TranspilerSession &session,
                        clang::CompilerInstance &compiler);
  ~SessionStage() = default;

  clang::CompilerInstance &getCompiler();

  clang::Rewriter& getRewriter();
  std::string getRewriterResult();

  [[nodiscard]] TRANSPILER_TYPE getBackend() const;
  AttributeManager &getAttrManager();

  void addDiagnosticMessage(clang::StoredDiagnostic &&message);

  //TODO: might need better redesign by design patterns
  bool setUserCtx(const std::string& key, std::any ctx);
  std::any getUserCtx(const std::string& key);

protected:
  TranspilerSession &_session;

  clang::CompilerInstance &_compiler;
  clang::Rewriter _rewriter;

  llvm::SmallVector<clang::StoredDiagnostic> _diagMessages;

  //XXX discuss key
  std::map<std::string, std::any> _userCtxMap;
};
}
