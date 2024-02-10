#pragma once

#include <clang/AST/AST.h>

namespace oklt {

class SessionStage;

struct AttrStmtSema {
  bool beforeTraverse(clang::AttributedStmt *stmt, SessionStage &stage);
  bool afterTraverse(clang::AttributedStmt *stmt, SessionStage &stage);
};

}
