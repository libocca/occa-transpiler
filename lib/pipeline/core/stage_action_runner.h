#pragma once

#include "core/transpiler_session/transpiler_session.h"

namespace oklt {
using SharedTranspilerSessionResult = tl::expected<SharedTranspilerSession, std::vector<Error>>;

SharedTranspilerSessionResult runStageAction(llvm::StringRef stageName,
                                             SharedTranspilerSession session);

SharedTranspilerSessionResult runPipeline(const std::vector<std::string>& pipeline,
                                          SharedTranspilerSession session);

}  // namespace oklt
