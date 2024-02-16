#pragma once

namespace clang {
struct AttributedStmt;
}  // namespace clang
   //
namespace oklt {
struct SessionStage;

bool prepareOklForStmt(const clang::AttributedStmt*, SessionStage&);
bool transpileOklForStmt(const clang::AttributedStmt*, SessionStage&);

}  // namespace oklt
   // namespace oklt
