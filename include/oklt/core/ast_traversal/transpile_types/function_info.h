#pragma once

#include <oklt/core/ast_traversal/transpile_types/attribute_info.h>
#include <oklt/core/ast_traversal/transpile_types/param_info.h>
#include <oklt/core/ast_traversal/transpile_types/for_stmt_info.h>
#include <oklt/core/kernel_info/kernel_info.h>
#include <vector>

namespace oklt {


struct FunctionInfo {
  static constexpr const char STAGE_NAME[] = "function_info_ctx";
  clang::FunctionDecl *astNode;
  std::vector<AttributeInfoPtr> attrs;
  clang::QualType returnType;
  std::string name;
  std::vector<ParamInfoPtr> parameters;
  std::vector<OuterForStmt> outForStmts;
  [[nodiscard]] std::string getFunctionSignature() const;
  [[nodiscard]] std::vector<ParsedKernelInfo> makeParsedKernelInfo() const;
  [[nodiscard]] bool isValid() const;
};
}
