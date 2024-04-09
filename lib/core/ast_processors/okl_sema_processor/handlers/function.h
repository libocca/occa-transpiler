#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct FunctionDecl;
struct ParmVarDecl;
struct Attr;
}  // namespace clang
   //
namespace oklt {
class SessionStage;
class OklSemaCtx;

HandleResult preValidateOklKernel(SessionStage& stage,
                                  const clang::FunctionDecl& fd,
                                  const clang::Attr& attr);
HandleResult postValidateOklKernel(SessionStage& stage,
                                   const clang::FunctionDecl& fd,
                                   const clang::Attr& attr);

}  // namespace oklt
   // namespace oklt
