#pragma once

#include <oklt/core/error.h>
#include "core/attribute_manager/result.h"

#include <tl/expected.hpp>

#include <any>
#include <string>

namespace clang {
class Decl;
class Stmt;
class Attr;
}  // namespace clang

namespace oklt {

class SessionStage;

HandleResult handleGlobalConstant(const clang::Decl* decl,
                                  SessionStage& s,
                                  const std::string& qualifier);
HandleResult handleGlobalFunction(const clang::Decl* decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier);

HandleResult handleTranslationUnit(const clang::Decl* decl,
                                   SessionStage& s,
                                   const std::string& include);
}  // namespace oklt
