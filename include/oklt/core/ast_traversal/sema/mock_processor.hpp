#pragma once

#include <clang/AST/AST.h>

#include <oklt/core/ast_traversal/sema/composite_sema.hpp>
#include <oklt/core/ast_traversal/sema/default_sema.hpp>

namespace oklt {

using SemanticMockProcessor = CompositeSema<DefaultTraverseSema<clang::Decl>,
                                            DefaultTraverseSema<clang::Stmt>,
                                            DefaultTraverseSema<clang::RecoveryExpr>,
                                            DefaultTraverseSema<clang::FunctionDecl>,
                                            DefaultTraverseSema<clang::ParmVarDecl>,
                                            DefaultTraverseSema<clang::AttributedStmt> >;

}
