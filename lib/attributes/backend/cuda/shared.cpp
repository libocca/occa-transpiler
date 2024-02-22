#include "attributes/attribute_names.h"
#include "attributes/utils/handle_shared.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleCUDASharedAttribute(const Attr* a, const Decl* d, SessionStage& s) {
    return handleSharedAttribute(a, d, s, "__shared__");
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, SHARED_ATTR_NAME}, AttrDeclHandler{handleCUDASharedAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
