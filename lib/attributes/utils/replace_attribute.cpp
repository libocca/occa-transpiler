#include "attributes/utils/replace_attribute.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>

#include <spdlog/spdlog.h>

namespace {

using namespace clang;

template <typename T>
oklt::HandleResult handleCXXRecordImpl(const T& node,
                                       oklt::Rewriter& r,
                                       const std::string& modifier_) {
    auto modifier = modifier_ + " ";
    // for all explicit constructors/methods add qualifier
    for (const auto& method : node.methods()) {
        if (method->isImplicit()) {
            continue;
        }
        auto loc = method->getBeginLoc();
        r.InsertTextBefore(loc, modifier);
    }

    // for all templated constructors/methods add qualifier
    for (const auto& decl : node.decls()) {
        if (!isa<FunctionTemplateDecl>(decl)) {
            continue;
        }
        auto funcTemplate = dyn_cast<FunctionTemplateDecl>(decl);
        if (funcTemplate->isImplicit()) {
            continue;
        }
        auto loc = funcTemplate->getAsFunction()->getBeginLoc();
        r.InsertTextBefore(loc, modifier);
    }

    return {};
}
}  // namespace

namespace oklt {
using namespace clang;

HandleResult handleGlobalConstant(const clang::VarDecl& decl,
                                  SessionStage& s,
                                  const std::string& qualifier) {
    if (!isGlobalConstVariable(decl)) {
        return {};
    }

    auto typeStr = decl.getType().getAsString();
    auto declname = decl.getDeclName().getAsString();
    SPDLOG_DEBUG(
        "Found constant global variable declaration: type: {}, name: {}", typeStr, declname);

    std::string newDeclStr;
    if (isArray(decl)) {
        newDeclStr = getNewDeclStrArray(decl, qualifier);
    } else if (isPointerToConst(decl)) {
        newDeclStr = getNewDeclStrPointerToConst(decl, qualifier);
    } else {
        newDeclStr = getNewDeclStrVariable(decl, qualifier);
    }

    if (decl.hasExternalStorage()) {
        newDeclStr = "extern " + newDeclStr;
    }

    // INFO: volatile const int var_const = 0;
    //       ^                          ^
    //      start_loc                  end_loc
    auto startLoc = decl.getBeginLoc();
    auto endLoc = decl.getLocation();
    auto range = SourceRange(startLoc, endLoc);

    s.getRewriter().ReplaceText(range, newDeclStr);

    return {};
}

HandleResult handleGlobalFunction(const clang::FunctionDecl& decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier) {
    // INFO: Check if function is not attributed with OKL attribute
    auto loc = decl.getSourceRange().getBegin();
    auto spacedModifier = funcQualifier + " ";

    SPDLOG_DEBUG("Handle global function '{}' at {}",
                 decl.getNameAsString(),
                 decl.getLocation().printToString(s.getCompiler().getSourceManager()));

    s.getRewriter().InsertTextBefore(loc, spacedModifier);

    return {};
}

HandleResult handleCXXRecord(const clang::CXXRecordDecl& cxxRecord,
                             SessionStage& s,
                             const std::string& modifier) {
    if (cxxRecord.isImplicit()) {
        return {};
    }

    return handleCXXRecordImpl(cxxRecord, s.getRewriter(), modifier);
}

HandleResult handleCXXRecord(const clang::ClassTemplatePartialSpecializationDecl& cxxRecord,
                             SessionStage& s,
                             const std::string& modifier) {
    return handleCXXRecordImpl(cxxRecord, s.getRewriter(), modifier);
}

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   std::vector<std::string_view> headers,
                                   std::vector<std::string_view> ns) {
    SPDLOG_DEBUG("Handle translation unit");

    auto& deps = s.tryEmplaceUserCtx<HeaderDepsInfo>();
    for (auto header : headers) {
        deps.backendHeaders.emplace_back("#include " + std::string(header) + "\n");
    }

    for (auto n : ns) {
        deps.backendNss.emplace_back("using namespace " + std::string(n) + ";\n\n");
    }

    return {};
}
}  // namespace oklt
