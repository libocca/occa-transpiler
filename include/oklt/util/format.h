#pragma once

#include <string_view>

namespace oklt {
// INFO: can't be used from the Shared Library in some cases
//  double free occurs, needs deeper investigation
std::string format(std::string_view);
}  // namespace oklt
