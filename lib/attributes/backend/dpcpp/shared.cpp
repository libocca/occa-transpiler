#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleSharedAttribute(const Attr& a, const VarDecl& var, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @shared.\n";
#endif

    auto varName = var.getNameAsString();
    auto typeStr = var.getType().getLocalUnqualifiedType().getAsString();

    auto newDeclaration =
        util::fmt(
            "auto & {} = "
            "*(sycl::ext::oneapi::group_local_memory_for_overwrite<{}>(item_.get_group()))",
            varName,
            typeStr)
            .value();

    SourceRange range(getAttrFullSourceRange(a).getBegin(), var.getSourceRange().getEnd());

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ATTR, range, newDeclaration)
        .build();
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME}, makeSpecificAttrHandle(handleSharedAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
