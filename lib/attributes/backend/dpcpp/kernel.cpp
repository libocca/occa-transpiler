#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "oklt/util/string_utils.h"
#include "pipeline/stages/transpiler/error_codes.h"
#include "tl/expected.hpp"

namespace {
using namespace oklt;
using namespace clang;

const std::string EXTERN_C = "extern \"C\"";
const std::string DPCPP_ADDITIONAL_ARGUMENTS = "sycl::queue * queue_,sycl::nd_range<3> * range_";
const std::string INNER_SIZES_FMT = "[[sycl::reqd_work_group_size({},{},{})]]";
const std::string SUBMIT_QUEUE =
    R"(queue_->submit(
    [&](sycl::handler & handler_) {
      handler_.parallel_for(
        *range_,
        [=](sycl::nd_item<3> item_) {)";

HandleResult handleKernelAttribute(const clang::Attr& a,
                                   const clang::FunctionDecl& func,
                                   SessionStage& s) {
    auto& rewriter = s.getRewriter();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.getParsingKernelInfo() && sema.getParsingKernelInfo()->kernInfo) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::AT_LEAST_ONE_OUTER_REQUIRED,
                                         "[@kernel] requires at least one [@outer] for-loop"});
    }

    // TODO: Add  "[[sycl::reqd_work_group_size(x, y, z]]"
    // 1. Add 'extern "C" [[sycl::reqd_work_group_size(z,y,x)]]`
    auto kernelPrefix = EXTERN_C;
    auto sizes = loopInfo->getInnerSizes();
    if (!sizes.hasNullOpts()) {
        while (sizes.size() < 3) {
            sizes.emplace_front(1);
        }
        kernelPrefix +=
            " " +
            util::fmt(INNER_SIZES_FMT, *sizes[0], *sizes[1], *sizes[2]).value();
    }
    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.ReplaceText(attrRange, kernelPrefix);

    // 2. Rename function
    auto oldFunctionName = func.getNameAsString();
    SourceRange fnameRange(func.getNameInfo().getSourceRange());
    auto newFunctionName = "_occa_" + oldFunctionName + "_0";
    rewriter.ReplaceText(fnameRange, newFunctionName);

    sema.getParsingKernelInfo()->kernInfo->name = newFunctionName;

    // 3. Update function arguments
    auto insertedArgs = DPCPP_ADDITIONAL_ARGUMENTS;
    if (func.getNumParams() > 0) {
        insertedArgs += ",";
    }
    rewriter.InsertText(func.getFunctionTypeLoc().getLParenLoc().getLocWithOffset(sizeof("(") - 1),
                        insertedArgs);

    // 4. Add submission of kernel in the queue:
    auto* body = dyn_cast<CompoundStmt>(func.getBody());
    rewriter.InsertText(body->getLBracLoc().getLocWithOffset(sizeof("{") - 1), SUBMIT_QUEUE);

    // Close two new scopes
    rewriter.InsertText(body->getRBracLoc(), "});});");

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute (DPCPP backend): return type: "
                 << func.getReturnType().getAsString() << ", old kernel name: " << oldFunctionName
                 << '\n';
#endif

    return {};
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, KERNEL_ATTR_NAME}, makeSpecificAttrHandle(handleKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
