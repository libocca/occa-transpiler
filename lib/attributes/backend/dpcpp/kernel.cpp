#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";
const std::string dpccAdditionalArguments = "sycl::queue * queue_,sycl::nd_range<3> * range_";
const std::string submitQueue = \
R"(queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {)";

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
        rewriter.InsertText(func->getFunctionTypeLoc().getLParenLoc().getLocWithOffset(sizeof("(") - 1), dpccAdditionalArguments + ",");
    } else {
        rewriter.InsertText(func->getFunctionTypeLoc().getLParenLoc().getLocWithOffset(sizeof("(") - 1), dpccAdditionalArguments);
    }

    // 4. Add submission of kernel in the queue:
    auto* body = dyn_cast<CompoundStmt>(func->getBody());
    rewriter.InsertText(body->getLBracLoc().getLocWithOffset(sizeof("{") - 1), submitQueue);
    // Close two new scopes
    rewriter.InsertText(body->getRBracLoc(), "});});");


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
