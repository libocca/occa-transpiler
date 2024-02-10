#include <oklt/core/ast_traversal/sema/recovery_expr_sema.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/attribute_manager/attributed_type_map.h>
#include <oklt/core/attribute_manager/attribute_manager.h>


namespace oklt {
using namespace clang;

bool RecoveryExprSema::beforeTraverse(clang::RecoveryExpr *expr, SessionStage &stage)
{
  auto subExpr = expr->subExpressions();
  if (subExpr.empty()) {
    return true;
  }

  auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
  if (!declRefExpr) {
    return true;
  }

  auto& ctx = stage.getCompiler().getASTContext();
  auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto validationResult = validateAttributes(attrs, stage);

  if(!validationResult) {
    return false;
  }
  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return true;
  }
  _validateResult = std::make_unique<ValidatorResult>(validationResult);
  return true;
}

bool RecoveryExprSema::afterTraverse(clang::RecoveryExpr *expr, SessionStage &stage)
{
  if(!_validateResult) {
    return true;
  }
  auto& attrManager = stage.getAttrManager();
  const Attr* attr = _validateResult.release()->value();
  if (!attrManager.handleAttr(attr, expr, stage)) {
    return false;
  }
  return true;
}
}
