#pragma once

namespace clang {
struct FunctionDecl;
struct ParmVarDecl;
}  // namespace clang
   //
namespace oklt {
struct SessionStage;
struct OklSemaCtx;

bool preValidateOklKernelSema(const clang::FunctionDecl& fd, SessionStage& stage, OklSemaCtx&);
bool postValidateOklKernelSema(const clang::FunctionDecl& fd, SessionStage& stage, OklSemaCtx&);
bool preValidateOklKernelParamSema(const clang::ParmVarDecl& parm,
                                   SessionStage& stage,
                                   OklSemaCtx&);
bool postValidateOklKernelParamSema(const clang::ParmVarDecl& parm,
                                    SessionStage& stage,
                                    OklSemaCtx&);

}  // namespace oklt
   // namespace oklt
