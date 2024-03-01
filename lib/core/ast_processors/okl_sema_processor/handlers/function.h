#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct FunctionDecl;
struct ParmVarDecl;
struct Attr;
}  // namespace clang
   //
namespace oklt {
struct SessionStage;
struct OklSemaCtx;

HandleResult preValidateOklKernel(const clang::Attr& attr,
                                  const clang::FunctionDecl& fd,
                                  OklSemaCtx& sema,
                                  SessionStage& stage);
HandleResult postValidateOklKernel(const clang::Attr& attr,
                                   const clang::FunctionDecl& fd,
                                   OklSemaCtx& sema,
                                   SessionStage& stage);
HandleResult preValidateOklKernelAttrArg(const clang::Attr& attr,
                                         const clang::ParmVarDecl& parm,
                                         OklSemaCtx& sema,
                                         SessionStage& stage);
HandleResult postValidateOklKernelAttrArg(const clang::Attr& attr,
                                          const clang::ParmVarDecl& parm,
                                          OklSemaCtx& sema,
                                          SessionStage& stage);
HandleResult preValidateOklKernelParam(const clang::ParmVarDecl& parm,
                                       OklSemaCtx& sema,
                                       SessionStage& stage);
HandleResult postValidateOklKernelParam(const clang::ParmVarDecl& parm,
                                        OklSemaCtx& sema,
                                        SessionStage& stage);

}  // namespace oklt
   // namespace oklt
