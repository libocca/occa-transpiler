#pragma once

#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

namespace oklt {
UserResult transpile(UserInput input);
UserResult transpile_ex(UserInput input);
}  // namespace oklt
