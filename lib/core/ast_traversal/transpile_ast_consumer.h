#pragma once

#include "core/ast_traversal/ast_visitor.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>

namespace oklt {

class SessionStage;

class TranspileASTConsumer : public clang::ASTConsumer {
   public:
    explicit TranspileASTConsumer(SessionStage& stage);
    void HandleTranslationUnit(clang::ASTContext& context) override;

    SessionStage& getSessionStage();

   private:
    SessionStage& _stage;
};

}  // namespace oklt
