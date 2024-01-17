#pragma once

#include "oklt/core/config.h"
#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <ostream>

namespace oklt {
class TranspileFrontendAction: public clang::ASTFrontendAction {
public:
  explicit TranspileFrontendAction(TRANSPILER_TYPE backendType,
                                   std::ostream &output);
  ~TranspileFrontendAction() override = default;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &compiler, llvm::StringRef in_file) override;
private:
  TRANSPILER_TYPE _backend;
  std::ostream &_output;
};
}
