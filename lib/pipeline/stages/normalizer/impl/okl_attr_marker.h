#pragma once

#include <clang/Basic/SourceLocation.h>
#include <clang/AST/ParentMapContext.h>
#include "okl_attribute.h"

using namespace clang;
namespace oklt {

struct OklAttrMarker {
    OklAttribute attr;
    struct {
        uint32_t line;
        uint32_t col;
    } loc;
};

}  // namespace oklt
