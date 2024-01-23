#pragma once

#include <clang/Basic/SourceLocation.h>

namespace oklt {
struct OklAttribute {
  std::string raw;
  std::string name;
  std::string params;
  clang::SourceLocation begin_loc;
  std::vector<size_t> tok_indecies;
};

}
