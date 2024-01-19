#pragma once

#include <string>
#include <llvm/ADT/StringRef.h>

namespace oklt {
    //INFO: can't be used from the Shared Library in some cases
    // double free occurs, needs deeper investigation
    std::string format(llvm::StringRef Code);
}
