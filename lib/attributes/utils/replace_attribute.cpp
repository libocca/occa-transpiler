#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

HandleResult handleGlobalConstant(const clang::VarDecl& decl,
                                  SessionStage& s,
                                  const std::string& qualifier) {
    if (!isGlobalConstVariable(decl)) {
        return true;
    }

#ifdef TRANSPILER_DEBUG_LOG
    auto type_str = decl.getType().getAsString();
    auto declname = decl.getDeclName().getAsString();

    llvm::outs() << "[DEBUG] Found constant global variable declaration:"
                 << " type: " << type_str << ", name: " << declname << "\n";
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

    auto& rewriter = s.getRewriter();
    rewriter.ReplaceText(range, newDeclStr);
    return true;
}

HandleResult handleGlobalFunction(const clang::FunctionDecl& decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier) {
    // INFO: Check if function is not attributed with OKL attribute
    auto& am = s.getAttrManager();
    if ((decl.hasAttrs()) && (am.checkAttrs(decl.getAttrs(), decl, s))) {
        return true;
    }

    auto& rewriter = s.getRewriter();
    auto loc = decl.getSourceRange().getBegin();
    auto spacedModifier = funcQualifier + " ";
    rewriter.InsertTextBefore(loc, spacedModifier);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle global function '" << decl.getNameAsString() << "'\n";
#endif
    return true;
}

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl,
                                   SessionStage& s,
                                   const std::string& include) {
    auto& sourceManager = s.getCompiler().getSourceManager();
    auto mainFileId = sourceManager.getMainFileID();
    auto loc = sourceManager.getLocForStartOfFile(mainFileId);
    auto& rewriter = s.getRewriter();
    rewriter.InsertTextBefore(loc, include + "\n");

#ifdef TRANSPILER_DEBUG_LOG
    auto offset = sourceManager.getFileOffset(decl.getLocation());
    llvm::outs() << "[DEBUG] Found translation unit, offset: " << offset << "\n";
#endif

    return true;
}

}  // namespace oklt
