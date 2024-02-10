#pragma once

#include <oklt/core/ast_traversal/sema/composite_sema.hpp>
#include <oklt/core/ast_traversal/sema/default_sema.hpp>
#include <oklt/core/ast_traversal/sema/attr_stmt_sema.h>
#include <oklt/core/ast_traversal/sema/kernel_sema.h>
#include <oklt/core/ast_traversal/sema/params_sema.h>
#include <oklt/core/ast_traversal/sema/recovery_expr_sema.h>

namespace oklt {


using SemanticProcessor = CompositeSema<
  DefaultTraverseSema<clang::Decl>,
  DefaultTraverseSema<clang::Stmt>,
  RecoveryExprSema,
  KernelFunctionSema,
  ParamSema,
  AttrStmtSema
  >;

}
