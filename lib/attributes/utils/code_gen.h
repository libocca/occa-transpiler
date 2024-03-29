#pragma once

#include <oklt/core/kernel_metadata.h>
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

#include <string>

namespace oklt {
std::string getCondCompStr(const BinOp& bo);
std::string getUnaryStr(const UnOp& uo, const std::string& var);
std::string buildCloseScopes(int& openedScopeCounter);
HandleResult replaceAttributedLoop(const clang::Attr& a,
                                   const clang::ForStmt& f,
                                   const std::string& prefixCode,
                                   const std::string& suffixCode,
                                   SessionStage& s,
                                   bool insertInside = false);
HandleResult replaceAttributedLoop(const clang::Attr& a,
                                   const clang::ForStmt& f,
                                   const std::string& prefixCode,
                                   const std::string& suffixCode,
                                   const std::string& afterRBraceCode,
                                   SessionStage& s,
                                   bool insertInside = false);
}  // namespace oklt
