#pragma once

#include <llvm/ADT/StringRef.h>
#include <string>

namespace oklt::util {
std::string toLower(llvm::StringRef str);
}
