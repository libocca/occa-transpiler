#pragma once

#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

namespace oklt {
/**
 * @brief Transpiles the user input.
 *
 * @param input The user input to transpile.
 * @return UserResult The result of the transpilation.
 */
UserResult transpile(UserInput input);
}  // namespace oklt
