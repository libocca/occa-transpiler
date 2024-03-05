#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

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
