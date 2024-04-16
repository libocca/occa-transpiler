#include "pipeline/core/stage_action_registry.h"

LLVM_INSTANTIATE_REGISTRY(oklt::StagePluginRegistry);
namespace oklt {

std::unique_ptr<StageAction> instantiateStageAction(clang::StringRef stageName) {
    for (const auto& it : StagePluginRegistry::entries()) {
        if (it.getName() == stageName) {
            return it.instantiate();
        }
    }

    return nullptr;
}

}  // namespace oklt
