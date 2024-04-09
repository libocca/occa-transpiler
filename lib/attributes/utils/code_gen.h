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
HandleResult replaceAttributedLoop(SessionStage& s,
                                   const clang::ForStmt& f,
                                   const clang::Attr& a,
                                   const std::string& suffixCode,
                                   const std::string& prefixCode,
                                   bool insertInside);
HandleResult replaceAttributedLoop(SessionStage& s,
                                   const clang::ForStmt& f,
                                   const clang::Attr& a,
                                   const std::string& suffixCode,
                                   const std::string& afterRBraceCode,
                                   const std::string& prefixCode,
                                   bool insertInside);
}  // namespace oklt
