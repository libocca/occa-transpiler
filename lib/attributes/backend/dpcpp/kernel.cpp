#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";
const std::string dpccAdditionalArguments = "sycl::queue * queue_,sycl::nd_range<3> * range_";

HandleResult handleKernelAttribute(const clang::Attr* a,
                                   const clang::FunctionDecl* func,
                                   SessionStage& s) {
    auto& rewriter = s.getRewriter();

    // 1. Add 'extern "C"`
    SourceRange arange;
    arange.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    arange.setEnd(a->getRange().getEnd().getLocWithOffset(2));
    rewriter.ReplaceText(arange, externC);

    // 2. Rename function
    auto oldFunctionName = func->getNameAsString();
    SourceRange frange(func->getNameInfo().getSourceRange());
    auto newFunctionName = "_occa_" + oldFunctionName + "_0";  // TODO: use correct dim
    rewriter.ReplaceText(frange, newFunctionName);

    // 3. Update function arguments
    if (func->getNumParams() > 0) {
        rewriter.InsertText(func->getFunctionTypeLoc().getRParenLoc(), dpccAdditionalArguments + ",");
    } else {
        rewriter.InsertText(func->getFunctionTypeLoc().getRParenLoc(), dpccAdditionalArguments);
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute (DPCPP backend): return type: "
                 << func->getReturnType().getAsString() << ", old kernel name: " << oldFunctionName
                 << '\n';
#endif
    return true;
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, KERNEL_ATTR_NAME}, makeSpecificAttrHandle(handleKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
