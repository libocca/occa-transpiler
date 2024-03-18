#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

HandleResult handleGlobalConstant(const clang::VarDecl& decl,
                                  SessionStage& s,
                                  const std::string& qualifier) {
    if (!isGlobalConstVariable(decl)) {
        return {};
    }

#ifdef TRANSPILER_DEBUG_LOG
    auto type_str = decl.getType().getAsString();
    auto declname = decl.getDeclName().getAsString();

    llvm::outs() << "[DEBUG] Found constant global variable declaration:" << " type: " << type_str
                 << ", name: " << declname << "\n";
#endif

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
    auto start_loc = decl.getBeginLoc();
    auto end_loc = decl.getLocation();
    auto range = SourceRange(start_loc, end_loc);

    s.getRewriter().ReplaceText(range, newDeclStr);

    return {};
}

HandleResult handleGlobalFunction(const clang::FunctionDecl& decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier) {
    // INFO: Check if function is not attributed with OKL attribute
    auto loc = decl.getSourceRange().getBegin();
    auto spacedModifier = funcQualifier + " ";

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle global function '" << decl.getNameAsString() << "'\n";
#endif

    s.getRewriter().InsertTextBefore(loc, spacedModifier);

    return {};
}

HandleResult handleCXXRecord(const clang::CXXRecordDecl& cxxRecord,
                             SessionStage& s,
                             const std::string& qualifier) {
    const auto& sm = s.getCompiler().getSourceManager();
    // TODO move the logic to ast traversal to be common for all handlers
    // skip system headers
    if (SrcMgr::isSystem(sm.getFileCharacteristic(cxxRecord.getLocation()))) {
        return {};
    }

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
        if (!isa<FunctionTemplateDecl>(decl))
        {
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

#ifdef TRANSPILER_DEBUG_LOG
    auto offset = sourceManager.getFileOffset(decl.getLocation());
    llvm::outs() << "[DEBUG] Found translation unit, offset: " << offset << "\n";
#endif

    // s.getRewriter().InsertTextBefore(loc, "#include " + std::string(include) + "\n");
    s.tryEmplaceUserCtx<HeaderDepsInfo>().backendDeps.emplace_back("#include " +
                                                                   std::string(include) + "\n");

    return {};
}
}  // namespace oklt
