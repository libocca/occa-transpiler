#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleOPENMPKernelAttribute(const clang::Attr* attr,
                                 const clang::Decl* d,
                                 SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif

    auto functionDecl = dyn_cast_or_null<FunctionDecl>(d);
    if (!functionDecl) {
        return false;
    }

    removeAttribute(attr, stage);

    std::string outerText = "#pragma omp parallel for\n";
    return stage.getRewriter().InsertText(functionDecl->getBeginLoc(), outerText, false, true);

    return true;
}

__attribute__((constructor)) void registerOPENMPKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, KERNEL_ATTR_NAME}, AttrDeclHandler{handleOPENMPKernelAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
