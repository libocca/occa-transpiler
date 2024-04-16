#pragma once

#include <oklt/core/error.h>
#include "core/attribute_manager/result.h"

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

HandleResult handleCXXRecord(const clang::CXXRecordDecl&, SessionStage&, const std::string&);

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   std::vector<std::string_view> headers,
                                   std::vector<std::string_view> ns = {});
}  // namespace oklt
