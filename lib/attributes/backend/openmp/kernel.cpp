#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPKernelAttribute(const clang::Attr& attr,
                                         const clang::FunctionDecl& decl,
                                         SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    removeAttribute(attr, stage);
    auto& rewriter = stage.getRewriter();

    static std::string_view outerText = "extern \"C\"\n";
    rewriter.InsertText(decl.getBeginLoc(), outerText, false, true);

    auto& ctx = stage.getCompiler().getASTContext();
    for (const auto param : decl.parameters()) {
        if (!param || !param->getType().getTypePtrOrNull()) {
            continue;
        }

        auto t = param->getType();
        if (!t->isPointerType()) {
            auto locRange = param->DeclaratorDecl::getSourceRange();
            rewriter.InsertText(locRange.getEnd(), " &");
        }
    }

    return true;
}

__attribute__((constructor)) void registerOPENMPKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, KERNEL_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
