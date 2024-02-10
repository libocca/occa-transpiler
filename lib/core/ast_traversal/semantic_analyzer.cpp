#include <oklt/core/ast_traversal/semantic_analyzer.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_manager/attributed_type_map.h>
#include <oklt/core/ast_traversal/transpile_types/function_info.h>

namespace oklt {
using namespace clang;


//INFO: simple mapping
inline DatatypeCategory makeDatatypeCategory(const QualType &qt) {
  if(qt->isBuiltinType()) {
    return DatatypeCategory::BUILTIN;
  }
  return DatatypeCategory::CUSTOM;
}

SemanticAnalyzer::SemanticAnalyzer(SessionStage& stage,
                                   SemanticCategory category,
                                   AttrValidatorFnType attrValidateFunc)
    :_stage(stage)
    , _category(category)
    , _attrValidator(attrValidateFunc)
    , _kernels()
{}

SemanticAnalyzer::KernelInfoT& SemanticAnalyzer::getKernelInfo() {
  return _kernels;
}

bool SemanticAnalyzer::TraverseDecl(clang::Decl* decl) {
  return RecursiveASTVisitor<SemanticAnalyzer>::TraverseDecl(decl);
}

bool SemanticAnalyzer::TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue) {
  return RecursiveASTVisitor<SemanticAnalyzer>::TraverseStmt(stmt, queue);
}

bool SemanticAnalyzer::TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue)
{
  auto subExpr = expr->subExpressions();
  if (subExpr.empty()) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseRecoveryExpr(expr, queue);
  }

  auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
  if (!declRefExpr) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseRecoveryExpr(expr, queue);
  }

  auto& ctx = _stage.getCompiler().getASTContext();
  auto& attrTypeMap = _stage.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto validationResult = _attrValidator(attrs, _stage);

  if(!validationResult) {
    return false;
  }

  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseRecoveryExpr(expr, queue);
  }
  auto& attrManager = _stage.getAttrManager();
  if (!attrManager.handleAttr(maybeAttr, expr, _stage)) {
    return false;
  }
  return true;
}

bool SemanticAnalyzer::TraverseFunctionDecl(clang::FunctionDecl *funcDecl)
{
  if(!funcDecl->hasAttrs()) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseFunctionDecl(funcDecl);
  }
  auto &attrs = funcDecl->getAttrs();
  auto validationResult = _attrValidator(attrs, _stage);

  if(!validationResult) {
    return false;
  }

  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseFunctionDecl(funcDecl);
  }

  FunctionInfo functionInfoCtx;
  _stage.setUserCtx(FunctionInfo::STAGE_NAME, &functionInfoCtx);

  //INFO: needs manual travers to make the signature before traversin the body
  for(auto &param: funcDecl->parameters()) {
    bool ret = RecursiveASTVisitor<SemanticAnalyzer>::TraverseParmVarDecl(param);
    if(!ret) {
      return false;
    }
  }
  auto& attrManager = SessionStage::getAttrManager();
  auto handledResult = attrManager.handleAttr(maybeAttr, funcDecl, _stage);

  if(!handledResult) {
    return false;
  }

  //TODO: check double traversing the Params
  bool ret = RecursiveASTVisitor<SemanticAnalyzer>::TraverseFunctionDecl(funcDecl);
  if(ret) {
    auto infos = std::move(functionInfoCtx.makeParsedKernelInfo());
    _kernels.insert(_kernels.end(), infos.begin(), infos.end());
  }
  return ret;
}

bool SemanticAnalyzer::TraverseParmVarDecl(clang::ParmVarDecl *param) {
  auto ctxPtr = _stage.getUserCtx(FunctionInfo::STAGE_NAME);
  if(!ctxPtr) {
    //TODO: internal error
    return false;
  }
  if(!ctxPtr->has_value()) {
    //TODO: internal error
    return false;
  }
  FunctionInfo *ctx = std::any_cast<FunctionInfo*>(*ctxPtr);
  if(!ctx) {
    //TODO: logical error
    return false;
  }

  if(!param->hasAttrs()) {
    bool ret = RecursiveASTVisitor<SemanticAnalyzer>::TraverseParmVarDecl(param);
    if(ret) {
      ctx->parameters.push_back(std::make_shared<OriginalParamInfo>(param));
    }
    return ret;
  }
  auto attrs = param->getAttrs();
  auto validationResult = _attrValidator(attrs, _stage);

  if(!validationResult) {
    return false;
  }

  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    bool ret = RecursiveASTVisitor<SemanticAnalyzer>::TraverseParmVarDecl(param);
    if(ret) {
      ctx->parameters.push_back(std::make_shared<OriginalParamInfo>(param));
    }
    return ret;
  }

  auto& attrManager = SessionStage::getAttrManager();
  auto handledResult = attrManager.handleAttr(maybeAttr, param, _stage);

  if(!handledResult) {
    return false;
  }

}


bool SemanticAnalyzer::TraverseAttributedStmt(clang::AttributedStmt *attrStmt,
                                              DataRecursionQueue* queue)
{
  auto attrs = attrStmt->getAttrs();
  auto validationResult = _attrValidator(attrs, _stage);

  if(!validationResult) {
    return false;
  }

  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticAnalyzer>::TraverseAttributedStmt(attrStmt, queue);
  }

  auto& attrManager = _stage.getAttrManager();
  const Stmt* subStmt = attrStmt->getSubStmt();
  auto handledResult = attrManager.handleAttr(maybeAttr, subStmt, _stage);
  if (!handledResult) {
    return false;
  }
  return RecursiveASTVisitor<SemanticAnalyzer>::TraverseAttributedStmt(attrStmt, queue);
}

}
