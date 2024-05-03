#pragma once

#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

namespace oklt {
/**
 * @brief Normalizes the user input.
 *
 * @param input The user input to normalize.
 * @return UserResult The result of the normalization.
 */
UserResult normalize(UserInput input);
}  // namespace oklt
