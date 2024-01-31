#include <oklt/util/string_utils.h>
#include <llvm/ADT/StringRef.h>

namespace oklt::util {
std::string toLower(const std::string &str) {
  llvm::StringRef ref(str);
  return ref.lower();
}
}  // namespace oklt::util
