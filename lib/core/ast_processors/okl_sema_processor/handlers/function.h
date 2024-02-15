#pragma once

namespace clang {
struct FunctionDecl;
struct ParmVarDecl;
}  // namespace clang
   //
namespace oklt {
struct SessionStage;

bool prepareOklKernelFunction(const clang::FunctionDecl* fd, SessionStage& stage);
bool transpileOklKernelFunction(const clang::FunctionDecl* fd, SessionStage& stage);
bool prepareOklKernelParam(const clang::ParmVarDecl* parm, SessionStage& stage);
bool transpileOklKernelParam(const clang::ParmVarDecl* parm, SessionStage& stage);

}  // namespace oklt
   // namespace oklt
