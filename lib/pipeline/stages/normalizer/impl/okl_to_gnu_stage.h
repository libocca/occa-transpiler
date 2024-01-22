#pragma once

#include "okl_attr_marker.h"
#include <oklt/core/transpiler_session/transpiler_session.h>

#include <list>
#include <tl/expected.hpp>

using namespace clang;
namespace oklt {

struct OklToGnuStageOutput {
  std::string gnuCppSrc;
  std::list<OklAttrMarker> gnuMarkers;
  std::list<OklAttrMarker> recoveryMarkers;
};

struct OklToGnuStageInput {
  std::string oklCppSrc;
};

tl::expected<OklToGnuStageOutput, int> convertOklToGnuAttribute(OklToGnuStageInput input,
                                                                TranspilerSession &session);
} // namespace oklt

