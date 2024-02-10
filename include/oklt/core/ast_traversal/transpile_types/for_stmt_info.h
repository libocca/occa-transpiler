#pragma once

#include <clang/AST/AST.h>

namespace oklt {


struct OuterForStmt {
  clang::AttributedStmt *outerForStmt;
  std::vector<clang::AttributedStmt*> innerForStmts;
};

}
