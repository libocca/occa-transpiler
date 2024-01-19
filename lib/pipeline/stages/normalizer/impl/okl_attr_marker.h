#pragma once

#include "okl_attribute.h"
#include <clang/Basic/SourceLocation.h>

using namespace clang;
namespace oklt {

struct OklAttrMarker {
  OklAttribute attr;
  SourceLocation loc;
};

}  // namespace oklt
