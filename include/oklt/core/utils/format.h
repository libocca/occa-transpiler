#pragma once

#include <llvm/ADT/StringRef.h>
#include <string>

namespace oklt {
// INFO: can't be used from the Shared Library in some cases
//  double free occurs, needs deeper investigation
std::string format(llvm::StringRef Code);
}  // namespace oklt
