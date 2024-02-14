#pragma once

#include <clang/Basic/SourceLocation.h>
#include "okl_attribute.h"

using namespace clang;
namespace oklt {

struct OklAttrMarker {
    OklAttribute attr;
    SourceLocation loc;
};

}  // namespace oklt
