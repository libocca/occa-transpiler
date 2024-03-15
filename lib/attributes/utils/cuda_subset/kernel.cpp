#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace oklt::cuda_subset {
using namespace clang;

const std::string CUDA_KERNEL_DEFINITION = "extern \"C\" __global__";

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
        return tl::unexpected<Error>(makeError(
            OkltTranspilerErrorCode::INTERNAL_ERROR_KERNEL_INFO_NULL, "handleKernelAttribute"));
    }
    sema.getParsingKernelInfo()->kernInfo->name = newFunctionName;

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

    s.getRewriter().ReplaceText(arange, CUDA_KERNEL_DEFINITION);
    s.getRewriter().ReplaceText(frange, newFunctionName);

    return {};
}

}  // namespace oklt::cuda_subset
