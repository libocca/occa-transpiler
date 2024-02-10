#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/kernel_info/kernel_info.h>

namespace oklt {

template<class SemanticAnalyzer>
class AstTraversal : public clang::RecursiveASTVisitor<AstTraversal<SemanticAnalyzer>>
{
 public:

  using BaseType = clang::RecursiveASTVisitor<AstTraversal<SemanticAnalyzer>>;
  explicit AstTraversal(SessionStage& stage);
  ~AstTraversal() = default;
  [[nodiscard]] const std::vector<ParsedKernelInfo>& getKernelInfo() const;

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, typename BaseType::DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, typename BaseType::DataRecursionQueue* queue = nullptr);

  bool TraverseFunctionDecl(clang::FunctionDecl *funcDecl);
  bool TraverseParmVarDecl(clang::ParmVarDecl *param);
  bool TraverseAttributedStmt(clang::AttributedStmt *attrStmt, typename BaseType::DataRecursionQueue* queue = nullptr);

 protected:
  SessionStage &_stage;
  std::vector<ParsedKernelInfo> _kernels;
  SemanticAnalyzer _analyzer;
};


template<class SemanticAnalyzer>
AstTraversal<SemanticAnalyzer>::AstTraversal(SessionStage& stage)
    :_stage(stage)
    , _kernels()
    , _analyzer()
{}

template<class SemanticAnalyzer>
const std::vector<ParsedKernelInfo>& AstTraversal<SemanticAnalyzer>::getKernelInfo() const
{
  return _kernels;
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseDecl(clang::Decl* decl)
{
  bool beforeResult = _analyzer.beforeTraverse(decl, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseDecl(decl);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(decl, _stage);
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseStmt(clang::Stmt* stmt,
                                                  typename BaseType::DataRecursionQueue* queue)
{
  bool beforeResult = _analyzer.beforeTraverse(stmt, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseStmt(stmt, queue);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(stmt, _stage);
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseRecoveryExpr(clang::RecoveryExpr* expr,
                                                          typename BaseType::DataRecursionQueue* queue)
{
  bool beforeResult = _analyzer.beforeTraverse(expr, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseRecoveryExpr(expr, queue);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(expr, _stage);
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseFunctionDecl(clang::FunctionDecl *funcDecl)
{
  bool beforeResult = _analyzer.beforeTraverse(funcDecl, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseFunctionDecl(funcDecl);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(funcDecl, _stage);
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseParmVarDecl(clang::ParmVarDecl *param)
{
  bool beforeResult = _analyzer.beforeTraverse(param, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseParmVarDecl(param);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(param, _stage);
}

template<class SemanticAnalyzer>
bool AstTraversal<SemanticAnalyzer>::TraverseAttributedStmt(clang::AttributedStmt *attrStmt,
                                                            typename BaseType::DataRecursionQueue* queue)
{
  bool beforeResult = _analyzer.beforeTraverse(attrStmt, _stage);
  if(!beforeResult) {
    return false;
  }
  bool traverseResult = BaseType::TraverseAttributedStmt(attrStmt, queue);
  if(!traverseResult) {
    return false;
  }
  return _analyzer.afterTraverse(attrStmt, _stage);
}

}  // namespace oklt
