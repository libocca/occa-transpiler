#pragma once

#include <memory>
#include <sstream>
#include "clang/AST/RecursiveASTVisitor.h"

namespace oklt {

class SessionStage;
class LauncherKernelGenerator;

class LauncherASTVisitor : public clang::RecursiveASTVisitor<LauncherASTVisitor> {
  using Base = RecursiveASTVisitor<LauncherASTVisitor>;
 public:
  LauncherASTVisitor(SessionStage& stage);

  bool TraverseTranslationUnitDecl(clang::TranslationUnitDecl* D);
  bool TraverseFunctionDecl(clang::FunctionDecl* D);
  bool TraverseAttributedStmt(clang::AttributedStmt* S, DataRecursionQueue* Queue = nullptr);

  static std::unique_ptr<LauncherASTVisitor> Create(SessionStage& stage);

 protected:
  SessionStage& _stage;
  std::stringstream _source;
  LauncherKernelGenerator* _generator;
};

}  // namespace oklt
