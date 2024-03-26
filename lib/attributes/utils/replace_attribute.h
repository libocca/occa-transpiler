#pragma once

#include <oklt/core/error.h>
#include "core/attribute_manager/result.h"

#include <clang/AST/Decl.h>
#include <tl/expected.hpp>

#include <string>

namespace clang {
class TranslationUnitDecl;
class CXXRecordDecl;
class FunctionDecl;
class VarDecl;
class Decl;
}  // namespace clang

namespace oklt {

class SessionStage;

HandleResult handleGlobalConstant(const clang::VarDecl& decl,
                                  SessionStage& s,
                                  const std::string& qualifier);
HandleResult handleGlobalFunction(const clang::FunctionDecl& decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier);
HandleResult handleCXXRecord(const clang::CXXRecordDecl&,
                                  SessionStage&,
                                  const std::string&);
HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   std::string_view includes);
}  // namespace oklt
