#pragma once

#include <llvm/Support/Registry.h>
#include "pipeline/core/stage_action.h"

namespace oklt {

using StagePluginRegistry = llvm::Registry<StageAction>;

std::unique_ptr<StageAction> instantiateStageAction(clang::StringRef stageName);
}  // namespace oklt
