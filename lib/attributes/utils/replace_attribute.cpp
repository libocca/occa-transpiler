#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>

#include <spdlog/spdlog.h>

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
                             const std::string& qualifier) {
    auto spacedModifier = qualifier + " ";

    // for all explicit constructors/methods add qualifier
    for (const auto& method : cxxRecord.methods()) {
        if (method->isImplicit()) {
            continue;
        }
        auto loc = method->getBeginLoc();
        s.getRewriter().InsertTextBefore(loc, spacedModifier);
    }

    // for all templated constructors/methods add qualifier
    for (const auto& decl : cxxRecord.decls()) {
        if (!isa<FunctionTemplateDecl>(decl)) {
            continue;
        }
        auto funcTemplate = dyn_cast<FunctionTemplateDecl>(decl);
        if (funcTemplate->isImplicit()) {
            continue;
        }
        auto loc = funcTemplate->getAsFunction()->getBeginLoc();
        s.getRewriter().InsertTextBefore(loc, spacedModifier);
    }

    return {};
}

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   std::string_view include) {
    auto& sourceManager = s.getCompiler().getSourceManager();
    auto mainFileId = sourceManager.getMainFileID();
    auto loc = sourceManager.getLocForStartOfFile(mainFileId);

    SPDLOG_DEBUG("Handle translation unit");

    // s.getRewriter().InsertTextBefore(loc, "#include " + std::string(include) + "\n");
    s.tryEmplaceUserCtx<HeaderDepsInfo>().backendDeps.emplace_back("#include " +
                                                                   std::string(include) + "\n");

    return {};
}
}  // namespace oklt
