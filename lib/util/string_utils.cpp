#include <oklt/util/string_utils.h>
#include <algorithm>

namespace oklt::util {
std::string toLower(const std::string &str) {
  std::string result;
  result.reserve(str.size());
  std::transform(str.begin(), str.end(), std::back_inserter(result), ::tolower);
  return result;
}
}  // namespace oklt::util
