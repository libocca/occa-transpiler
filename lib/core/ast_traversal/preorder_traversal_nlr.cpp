#include <oklt/core/ast_processor_manager/ast_processor_manager.h>
#include <oklt/core/ast_traversal/preorder_traversal_nlr.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
#define TRAVERSE_EXPR(EXPR_TYPE, EXPR_VAR, PROC_MNG, STAGE)                                 \
  do {                                                                                      \
    if ((EXPR_VAR) == nullptr) {                                                            \
      return true;                                                                          \
    }                                                                                       \
    auto procType = (STAGE).getAstProccesorType();                                          \
    auto cont = (PROC_MNG).runPreActionNodeHandle(procType, EXPR_VAR, STAGE);               \
    if (!cont) {                                                                            \
      return cont;                                                                          \
    }                                                                                       \
    cont = clang::RecursiveASTVisitor<PreorderNlrTraversal>::Traverse##EXPR_TYPE(EXPR_VAR); \
    if (!cont) {                                                                            \
      return cont;                                                                          \
    }                                                                                       \
    cont = (PROC_MNG).runPostActionNodeHandle(procType, EXPR_VAR, STAGE);                   \
    return true;                                                                            \
  } while (false)

}  // namespace
namespace oklt {

PreorderNlrTraversal::PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage)
    : _procMng(procMng), _stage(stage) {}
bool PreorderNlrTraversal::TraverseDecl(clang::Decl* decl) {
  TRAVERSE_EXPR(Decl, decl, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseStmt(clang::Stmt* stmt) {
  TRAVERSE_EXPR(Stmt, stmt, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseRecoveryExpr(clang::RecoveryExpr* recoveryExpr) {
  TRAVERSE_EXPR(RecoveryExpr, recoveryExpr, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseTranslationUnitDecl(clang::TranslationUnitDecl* translationUnitDecl) {
  TRAVERSE_EXPR(TranslationUnitDecl, translationUnitDecl, _procMng, _stage);
}

}  // namespace oklt
