#include <oklt/core/ast_traversal/sema/params_sema.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/ast_traversal/transpile_types/function_info.h>
#include <oklt/core/ast_traversal/validate_attributes.h>
#include <oklt/core/attribute_manager/attribute_manager.h>

namespace oklt {
using namespace clang;

bool ParamSema::beforeTraverse(ParmVarDecl *paramDecl, SessionStage &stage)
{
  //INFO: nothing to do with it
  if(!paramDecl->hasAttrs()) {
    return true;
  }

  auto &attrs = paramDecl->getAttrs();
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

bool ParamSema::afterTraverse(ParmVarDecl *paramDecl, SessionStage &stage)
{
  if(!paramDecl->hasAttrs()) {
    auto anyValue = stage.getUserCtx(FunctionInfo::STAGE_NAME);
    //INFO: not OKL function at all
    if(!anyValue) {
      return true;
    }
    if(!anyValue->has_value()) {
      //TODO: internal error
      return false;
    }
    FunctionInfo &funcCtx = std::any_cast<FunctionInfo&>(*anyValue);
    funcCtx.parameters.push_back(std::make_shared<OriginalParamInfo>(paramDecl));
    return true;
  }

  if(!_validateResult) {
    auto anyValue = stage.getUserCtx(FunctionInfo::STAGE_NAME);
    //INFO: not OKL function at all
    if(!anyValue) {
      return true;
    }
    if(!anyValue->has_value()) {
      //TODO: internal error
      return false;
    }
    FunctionInfo &funcCtx = std::any_cast<FunctionInfo&>(*anyValue);
    funcCtx.parameters.push_back(std::make_shared<OriginalParamInfo>(paramDecl));
    return true;
  }

  //INFO: OKL dependent parameter must be pushed by handler
  auto& attrManager = SessionStage::getAttrManager();
  const Attr *attr = _validateResult.release()->value();
  auto handledResult = attrManager.handleAttr(attr, paramDecl, stage);
  if(!handledResult) {
    return false;
  }

  return true;
}

}
