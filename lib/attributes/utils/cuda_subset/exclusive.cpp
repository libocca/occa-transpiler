#include "core/attribute_manager/result.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(const clang::Attr& attr,
                                      const clang::Decl& decl,
                                      SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif

    return TranspilationBuilder(
               stage.getCompiler().getSourceManager(), attr.getNormalizedFullName(), 1)
        .addReplacement(OKL_EXCLUSIVE, getAttrFullSourceRange(attr), "")
        .build();
}
}  // namespace oklt::cuda_subset
