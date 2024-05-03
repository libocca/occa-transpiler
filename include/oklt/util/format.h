#pragma once

#include <string_view>

namespace oklt {
// INFO: can't be used from the Shared Library in some cases
//  double free occurs, needs deeper investigation

/**
 * @brief Format source code in a string.
 *
 * @param str The source code string to format.
 * @return std::string The formatted code.
 */
std::string format(std::string_view);
}  // namespace oklt
