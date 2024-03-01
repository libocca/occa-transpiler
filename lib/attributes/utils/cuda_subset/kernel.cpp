#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
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

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 2u)
        .addReplacement(OKL_TRANSPILED_ATTR, arange, CUDA_KERNEL_DEFINITION)
        .addReplacement(OKL_TRANSPILED_NAME, frange, newFunctionName)
        .build();
}

}  // namespace oklt::cuda_subset
