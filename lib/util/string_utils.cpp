#include <oklt/util/string_utils.h>

namespace oklt::util {
std::string toLower(llvm::StringRef str) {
  return str.lower();
}
}  // namespace oklt::util
