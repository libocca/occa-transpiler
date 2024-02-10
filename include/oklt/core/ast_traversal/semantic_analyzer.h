#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <oklt/core/ast_traversal/semantic_category.h>
#include <oklt/core/ast_traversal/semantic_base.h>
#include <oklt/core/kernel_info/kernel_info.h>
#include <oklt/core/ast_traversal/validate_attributes.h>

namespace oklt {

class SessionStage;

//TODO:
// 1. remove hiearchy, strategy pattern in function style
// 2. handlers with custom return type (tl::expected<std::any, Error>
// 3. remove Variant - report errors directly to stage


class SemanticAnalyzer : public clang::RecursiveASTVisitor<SemanticAnalyzer>
{
 public:

  using KernelInfoT = decltype(KernelMetadata::metadata);
  
  explicit SemanticAnalyzer(SessionStage& session,
                            SemanticCategory category,
                            AttrValidatorFnType attrValidateFunc = validateAttributes);
  ~SemanticAnalyzer() = default;
  KernelInfoT& getKernelInfo();

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);

  bool TraverseFunctionDecl(clang::FunctionDecl *funcDecl);
  bool TraverseParmVarDecl(clang::ParmVarDecl *param);
  bool TraverseAttributedStmt(clang::AttributedStmt *attrStmt, DataRecursionQueue* queue = nullptr);

 protected:

  SessionStage &_stage;
  SemanticCategory _category;
  AttrValidatorFnType _attrValidator;
  KernelInfoT _kernels;
};

}  // namespace oklt
