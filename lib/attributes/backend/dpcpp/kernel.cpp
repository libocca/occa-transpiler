#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";
const std::string dpcppAdditionalArguments = "sycl::queue * queue_,sycl::nd_range<3> * range_";
const std::string submitQueue =
    R"(queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {)";

HandleResult handleKernelAttribute(const clang::Attr& a,
                                   const clang::FunctionDecl& func,
                                   SessionStage& s) {
    auto trans =
        TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 5);

    // 1. Add 'extern "C"`
    SourceRange attr_range;
    attr_range.setBegin(a.getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    attr_range.setEnd(a.getRange().getEnd().getLocWithOffset(2));
    trans.addReplacement(OKL_TRANSPILED_ATTR, attr_range, externC);

    // 2. Rename function
    auto oldFunctionName = func.getNameAsString();
    SourceRange fname_range(func.getNameInfo().getSourceRange());
    auto newFunctionName = "_occa_" + oldFunctionName + "_0";
    trans.addReplacement(OKL_TRANSPILED_NAME, fname_range, newFunctionName);
    // rewriter.ReplaceText(fname_range, newFunctionName);

    // 3. Update function arguments
    auto insertedArgs = dpcppAdditionalArguments;
    if (func.getNumParams() > 0) {
        insertedArgs += ",";
    }
    trans.addReplacement(OKL_TRANSPILED_ARG,
                         func.getFunctionTypeLoc().getLParenLoc().getLocWithOffset(sizeof("(") - 1),
                         insertedArgs);

    // 4. Add submission of kernel in the queue:
    auto* body = dyn_cast<CompoundStmt>(func.getBody());
    // rewriter.InsertText(body->getLBracLoc().getLocWithOffset(sizeof("{") - 1), submitQueue);
    trans.addReplacement(
        OKL_FUNCTION_PROLOGUE, body->getLBracLoc().getLocWithOffset(sizeof("{") - 1), submitQueue);
    // Close two new scopes
    // rewriter.InsertText(body->getRBracLoc(), "});});");
    trans.addReplacement(OKL_FUNCTION_EPILOGUE, body->getRBracLoc(), "});});");

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute (DPCPP backend): return type: "
                 << func.getReturnType().getAsString() << ", old kernel name: " << oldFunctionName
                 << '\n';
#endif

    return trans.build();
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, KERNEL_ATTR_NAME}, makeSpecificAttrHandle(handleKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
