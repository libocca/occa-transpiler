#pragma once

#include <oklt/core/error.h>
#include "core/attribute_manager/result.h"

#include <tl/expected.hpp>

#include <string>

namespace clang {
class TranslationUnitDecl;
class CXXRecordDecl;
class ClassTemplatePartialSpecializationDecl;
class FunctionDecl;
class VarDecl;
class Decl;
}  // namespace clang

namespace oklt {

class SessionStage;

HandleResult handleGlobalConstant(SessionStage& s,
                                  const clang::VarDecl& decl,
                                  const std::string& qualifier);


HandleResult handleGlobalFunction(SessionStage& s,
                                  const clang::FunctionDecl& decl,
                                  const std::string& funcQualifier);

HandleResult handleCXXRecord(SessionStage&, const clang::CXXRecordDecl&, const std::string&);
HandleResult handleCXXRecord(SessionStage&,
                             const clang::ClassTemplatePartialSpecializationDecl&,
                             const std::string&);

HandleResult handleTranslationUnit(SessionStage& s,
                                   const clang::TranslationUnitDecl& decl,
                                   std::vector<std::string_view> headers,
                                   std::vector<std::string_view> ns = {});
}  // namespace oklt
