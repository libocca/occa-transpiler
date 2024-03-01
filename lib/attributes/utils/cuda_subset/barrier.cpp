#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include "attributes/utils/cuda_subset/handle.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace oklt::cuda_subset {

oklt::HandleResult handleBarrierAttribute(const clang::Attr& attr,
                                          const clang::Stmt& stmt,
                                          const oklt::AttributedBarrier* params,
                                          SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    if (!params) {
        return tl::make_unexpected(
            makeError(OkltTranspilerErrorCode::INTERNAL_ERROR_PARAMS_NULL_OBJ,
                      "params is null object in handleBarrierAttribute"));
    }
    std::string replacement = "__syncthreads()";
    if (params->type == BarrierType::syncWarp) {
        replacement = "__syncwarp()";
    }
    auto range = getAttrFullSourceRange(attr);
    return TranspilationBuilder(
               stage.getCompiler().getSourceManager(), attr.getNormalizedFullName(), 1)
        .addReplacement(OKL_BARRIER, range.getBegin(), range.getEnd(), replacement)
        .build();
}
}  // namespace oklt::cuda_subset
