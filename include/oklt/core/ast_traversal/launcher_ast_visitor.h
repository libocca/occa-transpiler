#pragma once

#include <memory>
#include <sstream>
#include "clang/AST/RecursiveASTVisitor.h"

namespace oklt {

class SessionStage;
struct KernelMetadata;
class LauncherKernelGenerator;

class LauncherASTVisitor : public clang::RecursiveASTVisitor<LauncherASTVisitor> {
  using Base = RecursiveASTVisitor<LauncherASTVisitor>;
 public:
  LauncherASTVisitor(SessionStage& stage);
  virtual ~LauncherASTVisitor() = default;

  bool TraverseTranslationUnitDecl(clang::TranslationUnitDecl* D);
  bool TraverseFunctionDecl(clang::FunctionDecl* D);
  bool TraverseAttributedStmt(clang::AttributedStmt* S, DataRecursionQueue* Queue = nullptr);

 protected:
  SessionStage& _stage;
  std::stringstream _source;
  std::unique_ptr<LauncherKernelGenerator> _generator;
};

}  // namespace oklt
