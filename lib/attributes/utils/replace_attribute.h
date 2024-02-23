#pragma once

#include <string>
#include <tl/expected.hpp>
#include <any>
#include <oklt/core/error.h>

namespace clang {
class Decl;
class Stmt;
class Attr;
}  // namespace clang

namespace oklt {

class SessionStage;

tl::expected<std::any, Error> handleGlobalConstant(const clang::Decl* decl,
                                                   SessionStage& s,
                                                   const std::string& qualifier);
tl::expected<std::any, Error> handleGlobalFunction(const clang::Decl* decl,
                                                   SessionStage& s,
                                                   const std::string& funcQualifier);
}  // namespace oklt
