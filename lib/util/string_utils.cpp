#include <oklt/util/string_utils.h>
#include <algorithm>

namespace oklt::util {
std::string toLower(const std::string &str) {
  std::string result;
  result.reserve(str.size());
  std::transform(str.begin(), str.end(), std::back_inserter(result), ::tolower);
  return result;
}

std::string toCamelCase(std::string str) {
  std::size_t res_ind = 0;
  for (int i = 0; i < str.length(); i++) {
    // check for spaces in the sentence
    if (str[i] == ' ' || str[i] == '_') {
    // conversion into upper case
      str[i + 1] = ::toupper(str[i + 1]);
      continue;
    }
    // If not space, copy character
    else {
      str[res_ind++] = str[i];
    }
  }
          // return string to main
  return str.substr(0, res_ind);
}

}  // namespace oklt::util
