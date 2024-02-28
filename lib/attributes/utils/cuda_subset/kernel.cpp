#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"

namespace oklt::cuda_subset {
using namespace clang;

const std::string CUDA_KERNEL_DEFINITION = "extern \"C\" __global__";

HandleResult handleKernelAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
    auto func = dyn_cast<FunctionDecl>(d);

    // TODO: add __launch_bounds__

    // Replace attribute with cuda kernel definition
    SourceRange arange;
    arange.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    arange.setEnd(a->getRange().getEnd().getLocWithOffset(2));

    // Rename function
    auto oldFunctionName = func->getNameAsString();
    auto newFunctionName = "_occa_" + oldFunctionName + "_0";  // TODO: use correct dim
    SourceRange frange(func->getNameInfo().getSourceRange());

    // rewriter.ReplaceText(arange, CUDA_KERNEL_DEFINITION);
    // auto& rewriter = s.getRewriter();
    // rewriter.ReplaceText(frange, newFunctionName);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @kernel attribute: return type: "
                 << func->getReturnType().getAsString() << ", old kernel name: " << oldFunctionName
                 << '\n';
#endif

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a->getNormalizedFullName(), 2u)
        .addReplacement(OKL_TRANSPILED_ATTR, arange, CUDA_KERNEL_DEFINITION)
        .addReplacement(OKL_TRANSPILED_NAME, frange, newFunctionName)
        .build();
}

}  // namespace oklt::cuda_subset
