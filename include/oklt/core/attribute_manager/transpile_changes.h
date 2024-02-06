#pragma once

#include <clang/Basic/SourceLocation.h>
#include <functional>
#include <vector>

namespace oklt {


struct TranspileChanges {
  std::string from;
  std::string to;
  clang::SourceRange range;
};

using Changes = std::vector<TranspileChanges>;
using HandledChanges = std::function<void(const Changes &changes)>;

}
