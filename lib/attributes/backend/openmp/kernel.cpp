#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";

HandleResult handleOPENMPKernelAttribute(const Attr& a, const FunctionDecl& decl, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    auto& rewriter = s.getRewriter();

    // Add 'extern "C"'
    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.ReplaceText(attrRange, externC);

    // Convert ann non-pointer params to references
    auto& ctx = s.getCompiler().getASTContext();
    for (const auto param : decl.parameters()) {
        if (!param || !param->getType().getTypePtrOrNull()) {
            continue;
        }

        auto t = param->getType();
        if (!t->isPointerType()) {
            auto locRange = param->DeclaratorDecl::getSourceRange();
            rewriter.InsertTextAfter(locRange.getEnd(), " &");
        }
    }

    return {};
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
