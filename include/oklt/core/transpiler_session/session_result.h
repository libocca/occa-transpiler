#pragma once
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <tl/expected.hpp>

namespace oklt {
struct Error;

using TranspilerSessionResult = tl::expected<SharedTranspilerSession, std::vector<Error>>;

inline UserResult toUserResult(SharedTranspilerSession session) {
  return std::move(session->output);
}
}  // namespace oklt
