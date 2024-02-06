#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <oklt/core/ast_traversal/semantic_category.h>
#include <oklt/core/kernel_info/kernel_info.h>
// #include <oklt/core/diag/error.h>
#include <variant>

namespace oklt {

class SessionStage;

class SemanticAnalyzer : public clang::RecursiveASTVisitor<SemanticAnalyzer> {
 public:

  using KernelInfoT = decltype(KernelMetadata::metadata);
  
  explicit SemanticAnalyzer(SemanticCategory category,
                            SessionStage& session);
  virtual ~SemanticAnalyzer() = default;

  KernelInfoT& getKernelInfo();

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);

  bool TraverseFunctionDecl(clang::FunctionDecl *funcDecl);
  bool TraverseAttributedStmt(clang::AttributedStmt *attrStmt, DataRecursionQueue* queue = nullptr);

 protected:
  struct NoOKLAttrs {};
  struct ErrorFired {};

  using ValidationResult = std::variant<const clang::Attr*,
                                      NoOKLAttrs,
                                      ErrorFired>;

  ValidationResult validateAttribute(const clang::ArrayRef<const clang::Attr *> &attrs);

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
  
  SemanticCategory _category;
  SessionStage& _stage;
  KernelInfoT _kernels;
  std::list<KernelASTInfo> _astKernels;

};

}  // namespace oklt
