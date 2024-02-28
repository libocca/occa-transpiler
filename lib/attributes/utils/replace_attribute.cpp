#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

HandleResult handleGlobalConstant(const clang::Decl* decl,
                                  SessionStage& s,
                                  const std::string& qualifier) {
    // Should be variable declaration
    if (!isa<VarDecl>(decl)) {
        return {};
    }
    auto var = dyn_cast<VarDecl>(decl);

    if (!isGlobalConstVariable(var)) {
        return {};
    }

#ifdef TRANSPILER_DEBUG_LOG
    auto type_str = var->getType().getAsString();
    auto declname = var->getDeclName().getAsString();

    llvm::outs() << "[DEBUG] Found constant global variable declaration:" << " type: " << type_str
                 << ", name: " << declname << "\n";
#endif

    std::string newDeclStr;
    if (isConstantSizeArray(var)) {
        newDeclStr = getNewDeclStrConstantArray(var, qualifier);
    } else if (isPointerToConst(var)) {
        newDeclStr = getNewDeclStrPointerToConst(var, qualifier);
    } else {
        newDeclStr = getNewDeclStrVariable(var, qualifier);
    }

    // INFO: volatile const int var_const = 0;
    //       ^                          ^
    //      start_loc                  end_loc
    auto start_loc = var->getBeginLoc();
    auto end_loc = var->getLocation();
    auto range = SourceRange(start_loc, end_loc);

    auto& rewriter = s.getRewriter();
    // rewriter.ReplaceText(range, newDeclStr);

    return TranspilationBuilder(s.getCompiler().getSourceManager(), decl->getDeclKindName(), 1)
        .addReplacement(OKL_TRANSPILED_ATTR, range, newDeclStr)
        .build();
}

HandleResult handleGlobalFunction(const clang::Decl* decl,
                                  SessionStage& s,
                                  const std::string& funcQualifier) {
    // INFO: Check if function
    if (!isa<FunctionDecl>(decl)) {
        return {};
    }

    // INFO: Check if function is not attributed with OKL attribute
    auto& am = s.getAttrManager();
    if ((decl->hasAttrs()) && (am.checkAttrs(decl->getAttrs(), decl, s))) {
        return {};
    }

    auto& rewriter = s.getRewriter();
    auto loc = decl->getSourceRange().getBegin();
    auto spacedModifier = funcQualifier + " ";
    // rewriter.InsertTextBefore(loc, spacedModifier);

#ifdef TRANSPILER_DEBUG_LOG
    auto func = dyn_cast<FunctionDecl>(decl);
    llvm::outs() << "[DEBUG] Handle global function '" << func->getNameAsString() << "'\n";
#endif

    return TranspilationBuilder(s.getCompiler().getSourceManager(), decl->getDeclKindName(), 1u)
        .addReplacement(OKL_TRANSPILED_ATTR, loc, spacedModifier)
        .build();
}

HandleResult handleTranslationUnit(const clang::Decl* decl,
                                   SessionStage& s,
                                   std::string_view includes) {
    auto& sourceManager = s.getCompiler().getSourceManager();
    auto mainFileId = sourceManager.getMainFileID();
    auto loc = sourceManager.getLocForStartOfFile(mainFileId);

#ifdef TRANSPILER_DEBUG_LOG
    auto offset = sourceManager.getFileOffset(decl->getLocation());
    llvm::outs() << "[DEBUG] Found translation unit, offset: " << offset << "\n";
#endif

    return TranspilationBuilder(sourceManager, decl->getDeclKindName(), 1)
        .addInclude(includes)
        .build();
}

}  // namespace oklt
