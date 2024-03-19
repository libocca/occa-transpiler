#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "oklt/util/string_utils.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace oklt::cuda_subset {
using namespace clang;

const std::string CUDA_KERNEL_DEFINITION = "extern \"C\" __global__";
const std::string LAUNCH_BOUND_FMT = "__launch_bounds__({})";

HandleResult handleKernelAttribute(const clang::Attr& a,
                                   const clang::FunctionDecl& func,
                                   SessionStage& s) {
    // TODO: add __launch_bounds__

    // Replace attribute with cuda kernel definition
    SourceRange arange = getAttrFullSourceRange(a);

    // Rename function
    auto oldFunctionName = func.getNameAsString();
    auto newFunctionName = "_occa_" + oldFunctionName + "_0";
    SourceRange frange(func.getNameInfo().getSourceRange());

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.getParsingKernelInfo() && sema.getParsingKernelInfo()->kernInfo) {
        return tl::make_unexpected(Error{OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL,
                                         "handleKernelAttribute"});
    }
    sema.getParsingKernelInfo()->kernInfo->name = newFunctionName;

    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "[@kernel] requires at least one [@outer] for-loop"});
    }

    std::string kernelPrefix = CUDA_KERNEL_DEFINITION;
    auto sizes = loopInfo->getInnerSizes();
    if (!sizes.hasNullOpts()) {
        auto prod = sizes.product();
        kernelPrefix += " " + util::fmt(LAUNCH_BOUND_FMT, prod).value();
    }

    //    // Basic idea for the future split
    //    // INFO: must be outer nodes
    //    for(auto &elem: sema.getParsingKernelInfo()->children) {
    //        OklLoopInfo * outer = &elem;
    //        if(elem.isRegular()) {
    //            auto *ptr = elem.getFirstAttributedChild();
    //            if(!ptr) {
    //                continue;
    //            }
    //            outer = ptr;
    //        }
    //        //rewriter
    //        outer->stmt
    //        //end function
    //    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute: return type: "
                 << func.getReturnType().getAsString() << ", old kernel name: " << oldFunctionName
                 << '\n';
#endif

    s.getRewriter().ReplaceText(arange, kernelPrefix);
    s.getRewriter().ReplaceText(frange, newFunctionName);

    return {};
}

}  // namespace oklt::cuda_subset
