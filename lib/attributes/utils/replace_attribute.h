#pragma once

#include <oklt/core/error.h>
#include "core/attribute_manager/result.h"

#include <tl/expected.hpp>
#include <clang/AST/Decl.h>

#include <any>
#include <string>

namespace clang {
class FunctionDecl;
class VarDecl;
}  // namespace clang

namespace oklt {

class SessionStage;

HandleResult handleGlobalConstant(const clang::VarDecl& decl,
                                  SessionStage& s,
                                  const std::string& qualifier);
HandleResult handleGlobalFunction(const clang::FunctionDecl& decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier);

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   const std::string& include);
}  // namespace oklt
