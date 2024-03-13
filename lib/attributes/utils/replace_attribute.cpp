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
    if (isConstantSizeArray(decl)) {
        newDeclStr = getNewDeclStrConstantArray(decl, qualifier);
    } else if (isPointerToConst(decl)) {
        newDeclStr = getNewDeclStrPointerToConst(decl, qualifier);
    } else {
        newDeclStr = getNewDeclStrVariable(decl, qualifier);
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
    auto& am = s.getAttrManager();

    auto loc = decl.getSourceRange().getBegin();
    auto spacedModifier = funcQualifier + " ";

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle global function '" << decl.getNameAsString() << "'\n";
#endif

    s.getRewriter().InsertTextBefore(loc, spacedModifier);

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
