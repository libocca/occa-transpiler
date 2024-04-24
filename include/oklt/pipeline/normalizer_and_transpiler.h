#pragma once

#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

namespace oklt {
/**
 * @brief Normalizes the input and then transpiles it.
 *
 * @param input UserInput
 * @return UserResult The result of the transpilation
 */
UserResult normalizeAndTranspile(UserInput input);
}  // namespace oklt
