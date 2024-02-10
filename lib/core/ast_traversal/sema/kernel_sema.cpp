#include <oklt/core/ast_traversal/sema/kernel_sema.h>
#include <oklt/core/ast_traversal/transpile_types/function_info.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

#include <oklt/core/ast_traversal/validate_attributes.h>
#include <any>

namespace oklt {
using namespace clang;

bool KernelFunctionSema::beforeTraverse(clang::FunctionDecl* funcDecl, SessionStage& stage) {
  // INFO: all convertsion are done in afterTraverse
  if (!funcDecl->hasAttrs()) {
    return true;
  }

  auto& attrs = funcDecl->getAttrs();
  auto validationResult = validateAttributes(attrs, stage);
  if (!validationResult) {
    return false;
  }

  auto maybeAttr = validationResult.value();
  if (!maybeAttr) {
    return true;
  }
  // INFO: set validateResult only in the case
  //       that it will be processed in afterTraverse

  _validateResult = std::make_unique<ValidatorResult>(validationResult);
  std::any ctx{FunctionInfo{funcDecl}};
  stage.setUserCtx(FunctionInfo::STAGE_NAME, ctx);
  return true;
}

bool KernelFunctionSema::afterTraverse(clang::FunctionDecl* funcDecl, SessionStage& stage) {
  if (!funcDecl->hasAttrs()) {
    // TODO: call implicit attr manager handlers
    //  auto &attrManager = stage.getAttrManager();
    //  attrManager.handleImplicit(...)
    return true;
  }

  if (!_validateResult) {
    // TODO: call implicit attr manager handlers
    //  auto &attrManager = stage.getAttrManager();
    //  attrManager.handleImplicit(...)
    return true;
  }

  auto anyValue = stage.getUserCtx(FunctionInfo::STAGE_NAME);
  if (!anyValue) {
    // TODO: internal error
    return false;
  }
  if (!anyValue->has_value()) {
    // TODO: internal error
    return false;
  }

  FunctionInfo& funcCtx = std::any_cast<FunctionInfo&>(*anyValue);
  auto& attrManager = stage.getAttrManager();
  const Attr* attr = _validateResult.release()->value();
  if (!attrManager.handleAttr(attr, funcDecl, stage)) {
    return false;
  }

  if (!funcCtx.isValid()) {
    // TODO: logical error
    return false;
  }

  // TODO:
  // 1. make signature
  // 2. fill all checkpoints
  stage.removeUserCtx(FunctionInfo::STAGE_NAME);
  return true;
}

}  // namespace oklt
