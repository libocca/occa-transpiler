#pragma once

#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

namespace oklt {
UserResult normalize(UserInput input);

UserResult normalize_ex(UserInput input);
}  // namespace oklt
