#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <tl/expected.hpp>

namespace oklt {

class SessionStage;
class AstProcessorManager;
struct OklSemaCtx;
struct Error;

class PreorderNlrTraversal : public clang::RecursiveASTVisitor<PreorderNlrTraversal> {
   public:
    explicit PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage);
    bool TraverseDecl(clang::Decl* decl);
    bool TraverseStmt(clang::Stmt* stmt);
    bool TraverseTranslationUnitDecl(clang::TranslationUnitDecl* translationUnitDecl);

    tl::expected<std::string, Error> applyAstProcessor(clang::TranslationUnitDecl*);

   private:
    AstProcessorManager& _procMng;
    SessionStage& _stage;
};

}  // namespace oklt
