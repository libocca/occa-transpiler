#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <oklt/core/ast_traversal/semantic_category.h>
#include <oklt/core/ast_traversal/semantic_base.h>
#include <oklt/core/kernel_info/kernel_info.h>
#include <variant>

namespace oklt {

class SessionStage;

class SemanticAnalyzer : public clang::RecursiveASTVisitor<SemanticAnalyzer>
                         , public SemanticASTVisitorBase
{
 public:

  using KernelInfoT = decltype(KernelMetadata::metadata);
  
  explicit SemanticAnalyzer(SemanticCategory category,
                            SessionStage& session);
  ~SemanticAnalyzer() override = default;

  bool traverseTranslationUnit(clang::Decl* decl) override;
  KernelInfoT& getKernelInfo();

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);

  bool TraverseFunctionDecl(clang::FunctionDecl *funcDecl);
  bool TraverseAttributedStmt(clang::AttributedStmt *attrStmt, DataRecursionQueue* queue = nullptr);

 protected:

  struct OuterForStmt {
    clang::ForStmt *outer;
    std::vector<clang::ForStmt> inners;
  };

  struct KernelASTInfo {
    clang::FunctionDecl *currentKernel;
    std::vector<OuterForStmt> outers;
  };

  struct FunctionSignature {
    std::string attrs;
    std::string returnType;
    std::string funcName;
    std::vector<std::string> params;
  };
  
  // SessionStage& _stage;
  SemanticCategory _category;
  KernelInfoT _kernels;
  std::list<KernelASTInfo> _astKernels;

};

}  // namespace oklt
